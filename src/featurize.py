from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

INFILE   = Path("data/processed/chanlam_tidy.csv")
OUTFILE  = Path("data/processed/features.parquet")
META_DIR = Path("models"); META_DIR.mkdir(parents=True, exist_ok=True)
META_JSON = META_DIR / "encodings.json"

CATEGORICAL = ["s0", "s1", "b0", "b1", "bs", "bl", "c"]
CONTINUOUS = []

VAR_FLOOR = 1e-4

def minmax_scale_nonmissing(series: pd.Series):
    """Scale using only non-missing values; return (scaled_series, vmin, vmax).
       Missing values remain NaN (we will fill to 0 later but add a has_* flag)."""
    s = pd.to_numeric(series, errors="coerce")
    nonmiss = s.dropna()
    if nonmiss.empty:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index), np.nan, np.nan
    vmin = float(nonmiss.min()); vmax = float(nonmiss.max())
    if np.isclose(vmax, vmin):
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index), vmin, vmax
    scaled = (s - vmin) / (vmax - vmin)
    return scaled, vmin, vmax

def main(infile: Path = INFILE, outfile: Path = OUTFILE):
    df = pd.read_csv(infile)

    factor_cols = [c for c in (CATEGORICAL + CONTINUOUS + ["cond_id"]) if c in df.columns]
    meta_df = df[factor_cols].copy()

    cats_present = [c for c in CATEGORICAL if c in df.columns]
    df_cats = df[cats_present].copy()
    for c in cats_present:
        df_cats[c] = df_cats[c].fillna("__NA__").astype(str)

    dummies = pd.get_dummies(df_cats, prefix=cats_present, prefix_sep="=")
    cat_levels = {}
    for c in cats_present:
        cols = sorted([col for col in dummies.columns if col.startswith(f"{c}=")])
        dummies = dummies.reindex(
            columns=[*(x for x in dummies.columns if not x.startswith(f"{c}=")), *cols],
            fill_value=0,
        )
        cat_levels[c] = [col.split("=", 1)[1] for col in cols]

    cont_present = [c for c in CONTINUOUS if c in df.columns]
    df_cont = pd.DataFrame(index=df.index, dtype=float)
    cont_mins_maxs = {}
    for c in cont_present:
        has_flag = f"has_{c}"
        df_cont[has_flag] = df[c].notna().astype(float)
        scaled, vmin, vmax = minmax_scale_nonmissing(df[c])
        df_cont[f"{c}_scaled"] = scaled.fillna(0.0).astype(float)  # keep 0 for missing
        cont_mins_maxs[c] = {"min": vmin, "max": vmax}

    X = pd.concat([dummies, df_cont], axis=1).astype(np.float64)
    feature_cols = list(X.columns)

    mono_mean = pd.to_numeric(df.get("mean_yield_mono", pd.Series(np.nan)), errors="coerce").clip(lower=0, upper=100)
    mono_var  = pd.to_numeric(df.get("var_yield_mono",  pd.Series(np.nan)), errors="coerce")
    mono_n    = pd.to_numeric(df.get("n_mono",          pd.Series(np.nan)), errors="coerce").fillna(0).astype(int)

    di_mean = pd.to_numeric(df.get("mean_yield_di", pd.Series(np.nan)), errors="coerce").clip(lower=0, upper=100)
    di_var  = pd.to_numeric(df.get("var_yield_di",  pd.Series(np.nan)), errors="coerce")
    di_n    = pd.to_numeric(df.get("n_di",          pd.Series(np.nan)), errors="coerce").fillna(0).astype(int)

    def _clean_var(v: pd.Series) -> pd.Series:
        if v.isna().all():
            return pd.Series(VAR_FLOOR, index=df.index, dtype=float)
        med = float(np.nanmedian(v))
        fallback = VAR_FLOOR if not np.isfinite(med) else max(med, VAR_FLOOR)
        return v.fillna(fallback).clip(lower=VAR_FLOOR)

    mono_var = _clean_var(mono_var)
    di_var   = _clean_var(di_var)

    out = pd.concat(
        [
            meta_df.reset_index(drop=True),
            pd.DataFrame({
                "mono_mean": mono_mean.values,
                "mono_var":  mono_var.values,
                "mono_n":    mono_n.values,
                "di_mean":   di_mean.values,
                "di_var":    di_var.values,
                "di_n":      di_n.values,
            }),
            X.reset_index(drop=True),
        ],
        axis=1,
    )

    out = out.loc[:, ~out.columns.duplicated()]

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outfile, index=False)

    meta = {
        "categorical_levels": cat_levels,
        "continuous_minmax":  cont_mins_maxs,
        "feature_columns":    feature_cols,
        "targets": {
            "mono_mean": "mono_mean", "mono_var": "mono_var", "mono_n": "mono_n",
            "di_mean":   "di_mean",   "di_var":   "di_var",   "di_n":   "di_n",
        },
        "cond_id_column": "cond_id",
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved features → {outfile.resolve()}")
    print(f"Saved encodings → {META_JSON.resolve()}")
    print(f"n_samples = {len(out):,} | n_features = {len(feature_cols)}")
    n_nan_rows = np.isnan(out[feature_cols].to_numpy()).any(axis=1).sum()
    print(f"rows with any NaN in features: {n_nan_rows}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=Path, default=INFILE)
    p.add_argument("--outfile", type=Path, default=OUTFILE)
    args = p.parse_args()
    main(args.infile, args.outfile)
