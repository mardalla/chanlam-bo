from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch


@dataclass
class AcqConfig:
    acq_mode: str = "ei"      # "ei" | "lcb" | "kg"
    mc_samples: int = 64
    kappa: float = 2.0
    kg_chunk: int = 4096      # used for kg only
    cand_chunk: int = 256     # chunk candidate scoring to avoid CUDA OOM


def _build_qlognei(gp, X_base: torch.Tensor, cfg: AcqConfig):
    try:
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement as NEI
    except Exception:
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as NEI

    from botorch.sampling.normal import SobolQMCNormalSampler

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(cfg.mc_samples)]))
    acq = NEI(model=gp, X_baseline=X_base, sampler=sampler, prune_baseline=True)
    return acq


def _build_lcb(gp, X_base: torch.Tensor, cfg: AcqConfig):
    try:
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement as NEI
    except Exception:
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as NEI

    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.acquisition.objective import GenericMCObjective

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(cfg.mc_samples)]))
    kappa = float(cfg.kappa)

    def _obj(samples: torch.Tensor, **_):
        mu = samples.mean(dim=0, keepdim=True)
        sd = samples.std(dim=0, keepdim=True)
        lcb = (mu - kappa * sd).expand_as(samples).squeeze(-1)
        return lcb

    acq = NEI(
        model=gp,
        X_baseline=X_base,
        sampler=sampler,
        objective=GenericMCObjective(_obj),
        prune_baseline=True,
    )
    return acq


def _build_kg(gp, X_base: torch.Tensor, cfg: AcqConfig):
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.sampling.normal import SobolQMCNormalSampler

    nf = int(cfg.mc_samples)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([nf]))
    acq = qKnowledgeGradient(model=gp, num_fantasies=nf, sampler=sampler)
    return acq


def build_acq(gp, X_base: torch.Tensor, cfg: AcqConfig):
    mode = str(cfg.acq_mode).lower()
    if mode == "kg":
        return _build_kg(gp, X_base, cfg)
    if mode == "lcb":
        return _build_lcb(gp, X_base, cfg)
    # default: qLogNEI/qNEI
    return _build_qlognei(gp, X_base, cfg)


@torch.no_grad()
def score_candidates(gp, X_cand: torch.Tensor, X_base: torch.Tensor, cfg: AcqConfig) -> torch.Tensor:
    n = int(X_cand.shape[0])
    if n == 0:
        return torch.empty((0,), device=X_cand.device, dtype=X_cand.dtype)

    acq = build_acq(gp, X_base, cfg)

    chunk = int(cfg.cand_chunk) if int(cfg.cand_chunk) > 0 else n
    mode = str(cfg.acq_mode).lower()

    vals_list: List[torch.Tensor] = []

    if mode != "kg":
        for i in range(0, n, chunk):
            Xci = X_cand[i:i + chunk]
            vals_i = acq(Xci.unsqueeze(1)).reshape(-1)
            vals_list.append(vals_i.detach())
        return torch.cat(vals_list, dim=0)

    nf = int(cfg.mc_samples)
    for i in range(0, n, chunk):
        Xci = X_cand[i:i + chunk].unsqueeze(1)          # (c,1,d)
        X_f = Xci.repeat(1, nf, 1)                      # (c,nf,d)
        X_aug = torch.cat([Xci, X_f], dim=1)            # (c,1+nf,d)
        vals_i = acq(X_aug).reshape(-1)
        vals_list.append(vals_i.detach())

    return torch.cat(vals_list, dim=0)


def pick_best(cand_idx: List[int], scores: torch.Tensor) -> Tuple[int, float]:
    if len(cand_idx) == 0:
        raise ValueError("cand_idx empty")
    j = int(torch.argmax(scores).item())
    return int(cand_idx[j]), float(scores[j].item())