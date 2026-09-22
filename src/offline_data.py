from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

import json
import numpy as np
import pandas as pd
import torch


@dataclass
class OfflineData:
    T: pd.DataFrame
    raw_map: Dict[str, List[Tuple[float, float]]]
    df_feat: pd.DataFrame
    X_all: torch.Tensor
    meta: dict
    universe: Set[str]


def load_tidy(tidy_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(tidy_csv)

    needed = ["cond_id",
              "mean_yield_mono", "var_yield_mono", "n_mono",
              "mean_yield_di", "var_yield_di", "n_di"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Tidy missing columns: {missing}\nHave: {list(df.columns)}")

    T = df.rename(columns={
        "mean_yield_mono": "y_mean_mono",
        "var_yield_mono":  "y_var_mono",
        "n_mono":          "n_reps_mono",
        "mean_yield_di":   "y_mean_di",
        "var_yield_di":    "y_var_di",
        "n_di":            "n_reps_di",
    }).loc[:, ["cond_id", "y_mean_mono", "y_var_mono", "n_reps_mono",
               "y_mean_di", "y_var_di", "n_reps_di"]].copy()

    for c in ["y_mean_mono", "y_mean_di", "y_var_mono", "y_var_di"]:
        T[c] = pd.to_numeric(T[c], errors="coerce")

    T = T.dropna(subset=["y_mean_mono", "y_mean_di"]).copy()
    T["y_mean_mono"] = T["y_mean_mono"].clip(0, 100)
    T["y_mean_di"]   = T["y_mean_di"].clip(0, 100)

    vm_med = float(np.nanmedian(T["y_var_mono"])) if np.isfinite(np.nanmedian(T["y_var_mono"])) else 1e-4
    vd_med = float(np.nanmedian(T["y_var_di"]))   if np.isfinite(np.nanmedian(T["y_var_di"]))   else 1e-4
    T["y_var_mono"] = T["y_var_mono"].fillna(vm_med).clip(lower=1e-6)
    T["y_var_di"]   = T["y_var_di"].fillna(vd_med).clip(lower=1e-6)

    T["cond_id"] = T["cond_id"].astype(str)
    return T


def load_raw_map(raw_csv: Path) -> Dict[str, List[Tuple[float, float]]]:
    df = pd.read_csv(raw_csv)
    needed = ["cond_id", "yield_mono", "yield_di"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Raw missing columns: {missing}\nHave: {list(df.columns)}")

    df["cond_id"] = df["cond_id"].astype(str)
    df["yield_mono"] = pd.to_numeric(df["yield_mono"], errors="coerce")
    df["yield_di"]   = pd.to_numeric(df["yield_di"], errors="coerce")
    df = df.dropna(subset=["cond_id", "yield_mono", "yield_di"]).copy()

    raw_map: Dict[str, List[Tuple[float, float]]] = {}
    for cid, g in df.groupby("cond_id", sort=False):
        pairs = list(zip(g["yield_mono"].astype(float).tolist(),
                         g["yield_di"].astype(float).tolist()))
        raw_map[cid] = pairs
    return raw_map


def load_features(features_parquet: Path, meta_json: Path, dtype=np.float64) -> tuple[pd.DataFrame, dict, torch.Tensor]:
    df = pd.read_parquet(features_parquet)
    with open(meta_json, "r") as f:
        meta = json.load(f)

    if "cond_id" not in df.columns:
        raise KeyError("features.parquet must include cond_id.")

    feat_cols = meta.get("feature_columns", None)
    if not feat_cols:
        raise KeyError("encodings.json must contain feature_columns")

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"features.parquet missing {len(missing)} feature cols. Example: {missing[:10]}")

    df["cond_id"] = df["cond_id"].astype(str)
    X = torch.tensor(df[feat_cols].to_numpy(dtype=dtype))
    return df, meta, X


def align_universe(T: pd.DataFrame,
                   raw_map: Dict[str, List[Tuple[float, float]]],
                   df_feat: pd.DataFrame,
                   X_all: torch.Tensor,
                   meta: dict) -> OfflineData:
    tidy_set = set(T["cond_id"].astype(str))
    raw_set  = set(raw_map.keys())
    feat_set = set(df_feat["cond_id"].astype(str))
    universe = tidy_set & raw_set & feat_set
    if not universe:
        raise RuntimeError("Empty universe after intersecting tidy/raw/features cond_id sets.")

    T2 = T[T["cond_id"].isin(universe)].reset_index(drop=True)

    mask = df_feat["cond_id"].isin(universe).to_numpy()
    df2 = df_feat.loc[mask].reset_index(drop=True)
    X2  = X_all[mask]

    raw2 = {cid: pairs for cid, pairs in raw_map.items() if cid in universe}

    return OfflineData(T=T2, raw_map=raw2, df_feat=df2, X_all=X2, meta=meta, universe=universe)


def print_quick_stats(T: pd.DataFrame, raw_map: Dict[str, List[Tuple[float, float]]], df_feat: pd.DataFrame, X_all: torch.Tensor):
    print(f"[stats] tidy conditions: {len(T)}")
    print(f"[stats] raw_map conditions: {len(raw_map)}")
    print(f"[stats] features conditions: {len(df_feat)} | X shape={tuple(X_all.shape)}")

    counts = pd.Series([len(v) for v in raw_map.values()]).value_counts().sort_index()
    print("[stats] replicate counts per condition:")
    print(counts.to_string())

    X = X_all.detach().cpu().numpy()
    rs = X.sum(axis=1)
    print(f"[stats] feature row_sum min/mean/max: {rs.min():.1f} / {rs.mean():.1f} / {rs.max():.1f}")