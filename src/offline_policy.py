from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Tuple, List

import numpy as np
import torch

from src.offline_acq import AcqConfig, score_candidates, pick_best
from src.offline_oracle import ObservationStore, ReplicateOracle


@dataclass
class PolicyConfig:

    allow_repeats: bool = True
    max_cand: int = 4096              # subsample NEW candidates for speed
    min_new_per_batch: int = 0        # e.g. for batch_size>1, force at least k NEW per batch
    acq: AcqConfig = AcqConfig()      # use acq_mode="ei" for qLogNEI/qNEI


def choose_action(
    gp,
    *,
    X_all: torch.Tensor,                 # (N,d)
    idx_of: Dict[str, int],              # cond_id -> global index
    cid_of: Dict[int, str],              # global index -> cond_id
    seen_idx: Set[int],                  # unique evaluated condition indices
    obs: ObservationStore,
    oracle: ReplicateOracle,
    cfg: PolicyConfig,
    rng: np.random.Generator,
    n_new_in_batch: int = 0,
) -> Tuple[str, str, Dict[str, Any]]:


    if not seen_idx:
        raise RuntimeError("seen_idx is empty; need at least 1 evaluated condition for qLogNEI baseline.")
    base_idx = sorted(seen_idx)
    X_base = X_all[base_idx]

    n_all = int(X_all.shape[0])
    mask_new = np.ones(n_all, dtype=bool)
    if seen_idx:
        mask_new[list(seen_idx)] = False
    cand_new_idx = np.where(mask_new)[0].astype(int).tolist()

    cand_new_idx = [i for i in cand_new_idx if oracle.remaining(cid_of[i]) > 0]

    if len(cand_new_idx) > int(cfg.max_cand):
        cand_new_idx = rng.choice(cand_new_idx, size=int(cfg.max_cand), replace=False).astype(int).tolist()

    cand_rep_idx: List[int] = []
    if cfg.allow_repeats:
        for cid in obs.obs.keys():
            if oracle.remaining(cid) > 0:
                cand_rep_idx.append(int(idx_of[cid]))

    best_new_idx: Optional[int] = None
    best_new_score: Optional[float] = None
    if cand_new_idx:
        X_cand_new = X_all[cand_new_idx]
        scores_new = score_candidates(gp, X_cand_new, X_base, cfg=cfg.acq)
        best_new_idx, best_new_score = pick_best(cand_new_idx, scores_new)

    best_rep_idx: Optional[int] = None
    best_rep_score: Optional[float] = None
    if cand_rep_idx:
        X_cand_rep = X_all[cand_rep_idx]
        scores_rep = score_candidates(gp, X_cand_rep, X_base, cfg=cfg.acq)
        best_rep_idx, best_rep_score = pick_best(cand_rep_idx, scores_rep)

    must_new = (n_new_in_batch < int(cfg.min_new_per_batch))

    pick_kind: Optional[str] = None
    pick_idx: Optional[int] = None
    pick_score_raw: Optional[float] = None

    if must_new:
        if best_new_idx is None:
            if best_rep_idx is None:
                raise RuntimeError("No NEW candidates and no REPEAT candidates available.")
            pick_kind, pick_idx, pick_score_raw = "rep", best_rep_idx, float(best_rep_score)
        else:
            pick_kind, pick_idx, pick_score_raw = "new", best_new_idx, float(best_new_score)
    else:
        if best_new_idx is None and best_rep_idx is None:
            raise RuntimeError("No NEW candidates and no REPEAT candidates available.")
        if best_new_idx is None:
            pick_kind, pick_idx, pick_score_raw = "rep", best_rep_idx, float(best_rep_score)
        elif best_rep_idx is None:
            pick_kind, pick_idx, pick_score_raw = "new", best_new_idx, float(best_new_score)
        else:
            if float(best_rep_score) > float(best_new_score):
                pick_kind, pick_idx, pick_score_raw = "rep", best_rep_idx, float(best_rep_score)
            else:
                pick_kind, pick_idx, pick_score_raw = "new", best_new_idx, float(best_new_score)

    cid = cid_of[int(pick_idx)]
    if pick_kind == "rep" and oracle.remaining(cid) <= 0:
        if best_new_idx is None:
            raise RuntimeError("Picked REPEAT but no remaining replicate and no NEW candidates exist.")
        pick_kind, pick_idx = "new", best_new_idx
        cid = cid_of[int(pick_idx)]
        pick_score_raw = float(best_new_score) if best_new_score is not None else None

    debug = {
        "picked_kind": pick_kind,
        "picked_idx": int(pick_idx),
        "picked_cid": cid,
        "picked_score_raw": pick_score_raw,

        "n_new_candidates": int(len(cand_new_idx)),
        "n_rep_candidates": int(len(cand_rep_idx)),

        "best_new_idx": best_new_idx,
        "best_new_score_raw": best_new_score,
        "best_rep_idx": best_rep_idx,
        "best_rep_score_raw": best_rep_score,

        "must_new": must_new,
    }
    return str(pick_kind), str(cid), debug