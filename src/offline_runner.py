from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch

from src.offline_gp import fit_gp, GPFitState
from src.offline_oracle import ReplicateOracle, ObservationStore
from src.offline_policy import PolicyConfig, choose_action


@dataclass
class RunConfig:
    budget: int = 200              
    n_init: int = 24
    init_reps: int = 3             
    init_noise_var: float = 25.0    
    batch_size: int = 1
    max_reps_per_cond: int = 4
    noise_floor: float = 1e-4

    refit_every: int = 4
    fit_maxiter: int = 150

    lambda_di: float = 0.0

    mode: str = "repeats"


def _true_objective_map(T: pd.DataFrame, lam: float) -> Dict[str, float]:
    need = {"cond_id", "y_mean_mono", "y_mean_di"}
    missing = [c for c in need if c not in T.columns]
    if missing:
        raise KeyError(f"T missing columns for true objective: {missing}")
    out = {}
    for _, r in T.iterrows():
        cid = str(r["cond_id"])
        out[cid] = float(r["y_mean_mono"] - lam * r["y_mean_di"])
    return out


def _cond_s2_map(obs: ObservationStore, lam: float, noise_floor: float) -> Dict[str, float]:
    
    out: Dict[str, float] = {}
    for cid, pairs in obs.obs.items():
        if len(pairs) >= 2:
            arr = np.array(pairs, dtype=float)  # (n,2)
            f = arr[:, 0] - float(lam) * arr[:, 1]
            s2 = float(np.var(f, ddof=1))
            if np.isfinite(s2) and s2 >= 0:
                out[cid] = float(max(s2, noise_floor))
    return out


def _estimate_global_s2(obs: ObservationStore, lam: float, fallback: float) -> float:
    
    s2s = []
    for _, pairs in obs.obs.items():
        if len(pairs) >= 2:
            arr = np.array(pairs, dtype=float)
            f = arr[:, 0] - float(lam) * arr[:, 1]
            s2 = float(np.var(f, ddof=1))
            if np.isfinite(s2) and s2 >= 0:
                s2s.append(s2)
    if not s2s:
        return float(max(fallback, 1e-12))
    med = float(np.median(s2s))
    return float(max(med, fallback, 1e-12))


def _build_training_replicate_level(
    *,
    obs: ObservationStore,
    idx_of: Dict[str, int],
    lam: float,
    noise_floor: float,
    global_s2: float,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    
    s2_by_cid = _cond_s2_map(obs, lam=lam, noise_floor=noise_floor)

    train_idx: List[int] = []
    y: List[float] = []
    yvar: List[float] = []

    for cid, pairs in obs.obs.items():
        if not pairs:
            continue

        xi = int(idx_of[cid])
        s2 = float(s2_by_cid.get(cid, global_s2))
        s2 = float(max(s2, noise_floor))

        for (mono, di) in pairs:
            f = float(mono) - float(lam) * float(di)
            train_idx.append(xi)
            y.append(f)
            yvar.append(s2)

    if not train_idx:
        return [], np.array([], dtype=float), np.array([], dtype=float)

    return train_idx, np.array(y, dtype=float), np.array(yvar, dtype=float)


def run_one(
    *,
    T: pd.DataFrame,
    raw_map: Dict[str, List[Tuple[float, float]]],
    df_feat: pd.DataFrame,
    X_all: torch.Tensor,
    policy_cfg: PolicyConfig,
    cfg: RunConfig,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if device is None:
        device = X_all.device
    if dtype is None:
        dtype = X_all.dtype

    lam = float(cfg.lambda_di)
    rng = np.random.default_rng(seed)
    
    torch.manual_seed(seed)

    cond_ids = df_feat["cond_id"].astype(str).tolist()
    idx_of = {cid: i for i, cid in enumerate(cond_ids)}
    cid_of = {i: cid for cid, i in idx_of.items()}

    oracle = ReplicateOracle(raw_map, max_reps_per_cond=int(cfg.max_reps_per_cond))
    oracle.reset(seed=seed)
    obs = ObservationStore()

    if cfg.mode not in ("repeats", "norepeats"):
        raise ValueError(f"cfg.mode must be repeats|norepeats, got {cfg.mode}")
    policy_cfg.allow_repeats = (cfg.mode == "repeats")

    true_f = _true_objective_map(T, lam=lam)
    global_best_true = max(true_f.values())

    def cap_available(cid: str) -> int:
        return min(len(raw_map.get(cid, [])), int(cfg.max_reps_per_cond))

    eligible_init = [cid for cid in cond_ids if cap_available(cid) >= int(cfg.init_reps)]
    if len(eligible_init) < int(cfg.n_init):
        raise RuntimeError(
            f"Not enough conditions with >= init_reps={cfg.init_reps}. "
            f"Have {len(eligible_init)} eligible, need {cfg.n_init}."
        )

    init_cids = rng.choice(eligible_init, size=int(cfg.n_init), replace=False).tolist()

    seen_idx: Set[int] = set()
    tried_cids: Set[str] = set()

    n_seed_meas = 0
    for cid in init_cids:
        for _ in range(int(cfg.init_reps)):
            y = oracle.draw(cid)
            if y is None:
                raise RuntimeError(f"Oracle ran out during seeding for cid={cid}")
            obs.add(cid, *y)
            n_seed_meas += 1
        seen_idx.add(idx_of[cid])
        tried_cids.add(cid)

    if n_seed_meas != int(cfg.n_init) * int(cfg.init_reps):
        raise RuntimeError("Seeding count mismatch.")

    def best_observed_f() -> float:
        best = -np.inf
        for _, pairs in obs.obs.items():
            arr = np.array(pairs, dtype=float)
            f = arr[:, 0] - lam * arr[:, 1]
            m = float(np.mean(f))
            if m > best:
                best = m
        return float(best) if np.isfinite(best) else float("nan")

    def best_true_f_tried() -> float:
        if not tried_cids:
            return float("nan")
        return float(max(true_f[c] for c in tried_cids))

    hist_rows: List[Dict[str, Any]] = []
    t = n_seed_meas

    hist_rows.append({
        "experiments": t,
        "best_observed_f": best_observed_f(),
        "best_true_f": best_true_f_tried(),
        "picked_kind": "seed",
        "picked_cid": "",
        "global_best_true": global_best_true,

        "n_new_candidates": np.nan,
        "n_rep_candidates": np.nan,
        "best_new_score_raw": np.nan,
        "best_rep_score_raw": np.nan,
        "picked_score_raw": np.nan,
    })

    gp_state = GPFitState()
    rep_actions = 0
    new_actions = 0

    while t < int(cfg.budget):
        global_s2 = _estimate_global_s2(obs, lam=lam, fallback=float(cfg.init_noise_var))

        train_idx, y_rep, yvar_rep = _build_training_replicate_level(
            obs=obs,
            idx_of=idx_of,
            lam=lam,
            noise_floor=float(cfg.noise_floor),
            global_s2=float(global_s2),
        )
        if len(train_idx) == 0:
            raise RuntimeError("No training data available (unexpected).")

        X_train = X_all[train_idx].to(device=device, dtype=dtype)
        Y_train = torch.tensor(y_rep, device=device, dtype=dtype)
        Yvar_train = torch.tensor(yvar_rep, device=device, dtype=dtype)

        gp, gp_state = fit_gp(
            X_train, Y_train, Yvar_train,
            state=gp_state,
            refit_every=int(cfg.refit_every),
            fit_maxiter=int(cfg.fit_maxiter),
            noise_floor=float(cfg.noise_floor),
            device=device,
            dtype=dtype,
        )

        n_new_in_batch = 0
        for _ in range(int(cfg.batch_size)):
            if t >= int(cfg.budget):
                break

            kind, cid, dbg = choose_action(
                gp,
                X_all=X_all.to(device=device, dtype=dtype),
                idx_of=idx_of,
                cid_of=cid_of,
                seen_idx=seen_idx,
                obs=obs,
                oracle=oracle,
                cfg=policy_cfg,
                rng=rng,
                n_new_in_batch=n_new_in_batch,
            )

            if kind == "rep" and oracle.remaining(cid) <= 0:
                kind = "new"

            if kind == "new":
                n_new_in_batch += 1

            y = oracle.draw(cid)
            if y is None:
                raise RuntimeError(f"Oracle returned None for cid={cid} (kind={kind}).")

            obs.add(cid, *y)
            tried_cids.add(cid)

            if kind == "new":
                seen_idx.add(idx_of[cid])
                new_actions += 1
            else:
                rep_actions += 1

            t += 1

            hist_rows.append({
                "experiments": t,
                "best_observed_f": best_observed_f(),
                "best_true_f": best_true_f_tried(),
                "picked_kind": kind,
                "picked_cid": cid,
                "global_best_true": global_best_true,

                "n_new_candidates": dbg.get("n_new_candidates"),
                "n_rep_candidates": dbg.get("n_rep_candidates"),
                "best_new_score_raw": dbg.get("best_new_score_raw"),
                "best_rep_score_raw": dbg.get("best_rep_score_raw"),
                "picked_score_raw": dbg.get("picked_score_raw"),
            })


    n_all = obs.n_total()
    n_rep_data = obs.n_repeats()
    data_rep_share = (n_rep_data / n_all) if n_all else 0.0

    action_total = rep_actions + new_actions
    action_rep_share = (rep_actions / action_total) if action_total else 0.0

    stats = {
        "seed": int(seed),
        "mode": cfg.mode,
        "lambda_di": lam,
        "budget": int(cfg.budget),
        "n_init": int(cfg.n_init),
        "init_reps": int(cfg.init_reps),
        "init_noise_var": float(cfg.init_noise_var),
        "max_reps_per_cond": int(cfg.max_reps_per_cond),
        "n_total_meas": int(n_all),
        "n_unique_conds": int(obs.n_unique()),
        "n_rep_meas": int(n_rep_data),
        "replicate_share_data": float(data_rep_share),
        "rep_actions": int(rep_actions),
        "new_actions": int(new_actions),
        "replicate_share_actions": float(action_rep_share),
        "max_meas_per_cond": int(obs.max_per_cond()),
    }

    curve_df = pd.DataFrame(hist_rows)
    return curve_df, stats