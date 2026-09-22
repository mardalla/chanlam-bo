from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.offline_data import load_tidy, load_raw_map, load_features, align_universe
from src.offline_gp import fit_gp, GPFitState
from src.offline_oracle import ReplicateOracle, ObservationStore
from src.offline_acq import AcqConfig, score_candidates, pick_best
from src.offline_runner import (
    _true_objective_map,
    _estimate_global_s2,
    _build_training_replicate_level,
)

from run_mandatory_reps import _recommendations, _observed_argmax


def run_forced_repeat(
    *,
    T: pd.DataFrame,
    raw_map: Dict[str, List[Tuple[float, float]]],
    df_feat: pd.DataFrame,
    X_all: torch.Tensor,
    budget: int = 200,
    n_init: int = 24,
    repeat_share: float = 0.0,
    init_noise_var: float = 25.0,
    noise_floor: float = 1e-4,
    refit_every: int = 4,
    fit_maxiter: int = 150,
    lambda_di: float = 0.0,
    mc_samples: int = 64,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    init_cids: Optional[List[str]] = None,
    track_rec_every: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lam = float(lambda_di)

    cond_ids = df_feat["cond_id"].astype(str).tolist()
    idx_of = {cid: i for i, cid in enumerate(cond_ids)}

    oracle = ReplicateOracle(raw_map, max_reps_per_cond=4)
    oracle.reset(seed=seed)
    obs = ObservationStore()

    true_f = _true_objective_map(T, lam=lam)
    global_best_true = max(true_f.values())

    acq_cfg = AcqConfig(acq_mode="ei", mc_samples=mc_samples, cand_chunk=256)

    if init_cids is None:
        eligible = [cid for cid in cond_ids if len(raw_map.get(cid, [])) >= 1]
        init_cids = rng.choice(eligible, size=n_init, replace=False).tolist()

    seen_idx: Set[int] = set()
    tried_cids: Set[str] = set()

    obs_log: List[Dict[str, Any]] = []

    for cid in init_cids:
        y = oracle.draw(cid)
        if y is None:
            raise RuntimeError(f"Oracle empty during seeding for {cid}")
        obs.add(cid, *y)
        seen_idx.add(idx_of[cid])
        tried_cids.add(cid)
        obs_log.append({
            "step": len(obs_log) + 1, "cond_id": cid, "kind": "seed",
            "y_mono": float(y[0]), "y_di": float(y[1]),
            "y_obj": float(y[0] - lam * y[1]),
            "true_f": float(true_f.get(cid, np.nan)),
        })

    t = n_init

    def best_observed_f() -> float:
        _, v = _observed_argmax(obs, lam)
        return v

    def best_true_f_tried() -> float:
        return float(max(true_f[c] for c in tried_cids)) if tried_cids else float("nan")

    hist: List[Dict[str, Any]] = [{
        "experiments": t,
        "best_observed_f": best_observed_f(),
        "best_true_f": best_true_f_tried(),
        "picked_kind": "seed",
        "picked_cid": None,
        "y_obj": np.nan,
        "global_best_true": global_best_true,
        "best_new_score_raw": np.nan,
        "best_rep_score_raw": np.nan,
    }]

    gp_state = GPFitState()
    gp_last = None
    rep_actions = 0
    new_actions = 0

    while t < budget:
        global_s2 = _estimate_global_s2(obs, lam=lam, fallback=init_noise_var)

        train_idx, y_arr, yvar_arr = _build_training_replicate_level(
            obs=obs, idx_of=idx_of, lam=lam,
            noise_floor=noise_floor, global_s2=global_s2,
        )
        X_train = X_all[train_idx].to(device=device, dtype=torch.float64)
        Y_train = torch.tensor(y_arr, device=device, dtype=torch.float64)
        Yvar_train = torch.tensor(yvar_arr, device=device, dtype=torch.float64)

        gp, gp_state = fit_gp(
            X_train, Y_train, Yvar_train,
            state=gp_state, refit_every=refit_every,
            fit_maxiter=fit_maxiter, noise_floor=noise_floor,
            device=device, dtype=torch.float64,
        )
        gp_last = gp

        n_all = X_all.shape[0]
        mask_new = np.ones(n_all, dtype=bool)
        mask_new[list(seen_idx)] = False
        cand_new_idx = [i for i in np.where(mask_new)[0].tolist()
                        if oracle.remaining(cond_ids[i]) > 0]

        cand_rep_idx = [idx_of[cid] for cid in obs.obs
                        if oracle.remaining(cid) > 0]

        base_idx = sorted(seen_idx)
        X_base = X_all[base_idx].to(device=device, dtype=torch.float64)

        best_new_idx, best_new_score = None, None
        if cand_new_idx:
            scores_new = score_candidates(
                gp, X_all[cand_new_idx].to(device=device, dtype=torch.float64),
                X_base, acq_cfg,
            )
            best_new_idx, best_new_score = pick_best(cand_new_idx, scores_new)

        best_rep_idx, best_rep_score = None, None
        if cand_rep_idx:
            scores_rep = score_candidates(
                gp, X_all[cand_rep_idx].to(device=device, dtype=torch.float64),
                X_base, acq_cfg,
            )
            best_rep_idx, best_rep_score = pick_best(cand_rep_idx, scores_rep)

        do_repeat = (rng.random() < repeat_share) and (best_rep_idx is not None)

        if do_repeat:
            pick_idx = best_rep_idx
            kind = "rep"
        elif best_new_idx is not None:
            pick_idx = best_new_idx
            kind = "new"
        elif best_rep_idx is not None:
            pick_idx = best_rep_idx
            kind = "rep"
        else:
            break

        cid = cond_ids[pick_idx]

        if kind == "rep" and oracle.remaining(cid) <= 0:
            if best_new_idx is not None:
                pick_idx = best_new_idx
                cid = cond_ids[pick_idx]
                kind = "new"
            else:
                break

        y = oracle.draw(cid)
        if y is None:
            break

        obs.add(cid, *y)
        tried_cids.add(cid)

        if kind == "new":
            seen_idx.add(pick_idx)
            new_actions += 1
        else:
            rep_actions += 1

        t += 1

        obs_log.append({
            "step": len(obs_log) + 1, "cond_id": cid, "kind": kind,
            "y_mono": float(y[0]), "y_di": float(y[1]),
            "y_obj": float(y[0] - lam * y[1]),
            "true_f": float(true_f.get(cid, np.nan)),
        })

        row = {
            "experiments": t,
            "best_observed_f": best_observed_f(),
            "best_true_f": best_true_f_tried(),
            "picked_kind": kind,
            "picked_cid": cid,
            "y_obj": float(y[0] - lam * y[1]),
            "global_best_true": global_best_true,
            "best_new_score_raw": best_new_score,
            "best_rep_score_raw": best_rep_score,
        }
        if track_rec_every and (t % track_rec_every == 0):
            row.update(_recommendations(
                gp=gp_last, X_all=X_all, cond_ids=cond_ids, obs=obs,
                true_f=true_f, lam=lam, seen_idx=seen_idx,
                global_best_true=global_best_true, device=device,
            ))
        hist.append(row)

    global_s2 = _estimate_global_s2(obs, lam=lam, fallback=init_noise_var)
    train_idx, y_arr, yvar_arr = _build_training_replicate_level(
        obs=obs, idx_of=idx_of, lam=lam,
        noise_floor=noise_floor, global_s2=global_s2,
    )
    gp_final, _ = fit_gp(
        X_all[train_idx].to(device=device, dtype=torch.float64),
        torch.tensor(y_arr, device=device, dtype=torch.float64),
        torch.tensor(yvar_arr, device=device, dtype=torch.float64),
        state=GPFitState(), refit_every=1, fit_maxiter=fit_maxiter,
        noise_floor=noise_floor, device=device, dtype=torch.float64,
    )
    final_rec = _recommendations(
        gp=gp_final, X_all=X_all, cond_ids=cond_ids, obs=obs,
        true_f=true_f, lam=lam, seen_idx=seen_idx,
        global_best_true=global_best_true, device=device,
    )

    curve_df = pd.DataFrame(hist)
    obs_df = pd.DataFrame(obs_log)
    stats = {
        "seed": seed,
        "repeat_share_target": repeat_share,
        "lambda_di": lam,
        "budget": budget,
        "n_init": n_init,
        "n_total_meas": obs.n_total(),
        "n_unique_conds": obs.n_unique(),
        "rep_actions": rep_actions,
        "new_actions": new_actions,
        "actual_rep_share": rep_actions / max(rep_actions + new_actions, 1),
        "max_meas_per_cond": obs.max_per_cond(),
        "final_best_true_f": best_true_f_tried(),
        "final_best_obs_f": best_observed_f(),
        "global_best_true": global_best_true,
    }
    stats.update(final_rec)
    return curve_df, stats, obs_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--n_init", type=int, default=24)
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=37)
    parser.add_argument("--mc_samples", type=int, default=64)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--repeat_shares", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--track_rec_every", type=int, default=0,
                        help="0 = end-of-run nomination only; N>0 also logs one "
                             "every N measurements, giving a regret trajectory.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="results_forced")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    T = load_tidy(root / "data/processed/chanlam_tidy.csv")
    raw_map = load_raw_map(root / "data/processed/chanlam_raw.csv")
    feat_path = root / "data/processed/features.parquet"
    if not feat_path.exists():
        feat_path = root / "data/processed/features.csv"
    df_feat, meta, X_all = load_features(feat_path, root / "models/encodings.json")
    data = align_universe(T, raw_map, df_feat, X_all, meta)

    device = torch.device(args.device)
    X_tensor = torch.as_tensor(data.X_all, dtype=torch.float64).to(device)

    print(f"Universe: {len(data.universe)} conditions, {X_tensor.shape[1]} features")

    repeat_shares = list(args.repeat_shares)
    lambdas = list(args.lambdas)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    cond_ids = data.df_feat["cond_id"].astype(str).tolist()
    eligible = [cid for cid in cond_ids if len(data.raw_map.get(cid, [])) >= 1]
    seed_to_init = {
        seed: np.random.default_rng(seed).choice(eligible, size=args.n_init, replace=False).tolist()
        for seed in seeds
    }

    all_curves: List[pd.DataFrame] = []
    all_stats: List[Dict[str, Any]] = []
    all_obs: List[pd.DataFrame] = []
    total = len(lambdas) * len(repeat_shares) * len(seeds)
    run_i = 0

    for lam in lambdas:
        for rs in repeat_shares:
            for seed in seeds:
                run_i += 1
                print(f"  [{run_i}/{total}] λ={lam}, repeat_share={rs:.1f}, seed={seed}", end=" ")
                t0 = time.time()
                curve, stats, obs_df = run_forced_repeat(
                    T=data.T, raw_map=data.raw_map,
                    df_feat=data.df_feat, X_all=X_tensor,
                    budget=args.budget, n_init=args.n_init,
                    repeat_share=rs, lambda_di=lam,
                    mc_samples=args.mc_samples,
                    seed=seed, device=device,
                    init_cids=seed_to_init[seed],
                    track_rec_every=args.track_rec_every,
                )
                elapsed = time.time() - t0
                for frame in (curve, obs_df):
                    frame["repeat_share_target"] = rs
                    frame["seed"] = seed
                    frame["lambda_di"] = lam
                stats["elapsed_sec"] = round(elapsed, 1)
                all_curves.append(curve)
                all_stats.append(stats)
                all_obs.append(obs_df)
                print(f"→ {elapsed:.0f}s | visited={stats['final_best_true_f']:.1f} | "
                      f"rec_obs={stats['rec_obs_true']:.1f} | "
                      f"rec_post={stats['rec_post_true']:.1f} | "
                      f"unique={stats['n_unique_conds']} | "
                      f"rep={stats['rep_actions']} ({stats['actual_rep_share']:.0%})")

    curves_df = pd.concat(all_curves, ignore_index=True)
    stats_df = pd.DataFrame(all_stats)
    obs_all = pd.concat(all_obs, ignore_index=True)
    curves_df.to_csv(outdir / "curves_forced.csv", index=False)
    stats_df.to_csv(outdir / "stats_forced.csv", index=False)
    obs_all.to_csv(outdir / "observations_forced.csv", index=False)

    for lam in lambdas:
        obj = "yield only" if lam == 0 else "penalised yield"
        print(f"\n{'=' * 84}\n  {obj}\n{'=' * 84}")
        print(f"  {'share':>7} | {'visited':>16} | {'rec (observed)':>16} | "
              f"{'rec (posterior)':>16} | {'realised':>8}")
        for rs in repeat_shares:
            sub = stats_df[(stats_df["lambda_di"] == lam)
                           & (stats_df["repeat_share_target"] == rs)]
            print(f"  {rs:>6.0%} | "
                  f"{sub['final_best_true_f'].mean():7.2f} ± {sub['final_best_true_f'].std():5.2f} | "
                  f"{sub['rec_obs_true'].mean():7.2f} ± {sub['rec_obs_true'].std():5.2f} | "
                  f"{sub['rec_post_true'].mean():7.2f} ± {sub['rec_post_true'].std():5.2f} | "
                  f"{sub['actual_rep_share'].mean():7.1%}")

    print(f"\nDone! Results in {outdir}/")


if __name__ == "__main__":
    main()
