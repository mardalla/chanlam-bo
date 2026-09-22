from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import warnings
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


DEFAULT_NOISE_FLOOR = 1e-4


@dataclass
class GPFitState:
    hyper_state: Optional[Dict[str, Any]] = None
    calls: int = 0


def _filter_hyper_state(sd: Dict[str, Any]) -> Dict[str, Any]:
    keep_prefixes = ("covar_module.", "mean_module.")
    out = {}
    for k, v in sd.items():
        if k.startswith(keep_prefixes):
            out[k] = v
    return out


def _as_col(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected (n,) or (n,1); got {tuple(x.shape)}")


def _maybe_to(x: torch.Tensor, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> torch.Tensor:
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x


def fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    train_Yvar: torch.Tensor,
    *,
    state: Optional[GPFitState] = None,
    refit_every: int = 4,
    fit_maxiter: int = 150,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    suppress_warnings: bool = True,
) -> Tuple[torch.nn.Module, GPFitState]:
    
    if train_X.ndim != 2:
        raise ValueError(f"train_X must be (n,d); got {tuple(train_X.shape)}")

    train_Y = _as_col(train_Y)
    train_Yvar = _as_col(train_Yvar)

    if train_X.shape[0] != train_Y.shape[0] or train_Y.shape != train_Yvar.shape:
        raise ValueError(
            f"Shape mismatch: X={tuple(train_X.shape)} Y={tuple(train_Y.shape)} Yvar={tuple(train_Yvar.shape)}"
        )

    if device is None:
        device = train_X.device
    if dtype is None:
        dtype = train_X.dtype

    train_X = _maybe_to(train_X, device, dtype)
    train_Y = _maybe_to(train_Y, device, dtype)
    train_Yvar = _maybe_to(train_Yvar, device, dtype)

    train_Yvar = torch.clamp(train_Yvar, min=float(noise_floor))


    like = FixedNoiseGaussianLikelihood(
        noise=train_Yvar.squeeze(-1),
        learn_additional_noise=False
    ).to(device=device, dtype=dtype)

    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=like,
        input_transform=Normalize(d=train_X.shape[-1]).to(device=device, dtype=dtype),
        outcome_transform=Standardize(m=1).to(device=device, dtype=dtype),
    ).to(device=device, dtype=dtype)

    if state is None:
        state = GPFitState()

    if state.hyper_state is not None:
        gp.load_state_dict(state.hyper_state, strict=False)

    do_refit = (state.calls % max(int(refit_every), 1) == 0) or (state.hyper_state is None)

    if do_refit:
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll, options={"maxiter": int(fit_maxiter)})
        else:
            fit_gpytorch_mll(mll, options={"maxiter": int(fit_maxiter)})

        state.hyper_state = _filter_hyper_state(gp.state_dict())


    gp.eval()
    state.calls += 1
    return gp, state


@torch.no_grad()
def posterior_mean_std(gp: torch.nn.Module, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    post = gp.posterior(X)
    mu = post.mean.squeeze(-1)
    var = post.variance.squeeze(-1).clamp_min(1e-12)
    return mu, var.sqrt()