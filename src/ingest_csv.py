from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

B0 = 'inputs["Boronic Acid"].components[0].identifiers[0].value'
B1 = 'inputs["Boronic Acid"].components[1].identifiers[0].value'
S0 = 'inputs["Sulfonamide"].components[0].identifiers[0].value'
S1 = 'inputs["Sulfonamide"].components[1].identifiers[0].value'
BS = 'inputs["Base_Solid"].components[0].identifiers[0].value'
BL = 'inputs["Base_Liquid"].components[0].identifiers[0].value'
CAT = 'inputs["Catalyst"].components[0].identifiers[0].value'

YMONO = 'outcomes[0].products[0].measurements[0].percentage.value'
YDI   = 'outcomes[0].products[1].measurements[0].percentage.value'
RID   = "reaction_id"


MISSING = "<NA>"

def _s(x) -> str:
    if pd.isna(x):
        return MISSING
    return str(x).strip()


def _make_cond_id(r: pd.Series) -> str:
    return f"s0={r['s0']}|s1={r['s1']}|b0={r['b0']}|b1={r['b1']}|bs={r['bs']}|bl={r['bl']}|c={r['c']}"

def _var_or_nan(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) >= 2:
        return float(np.var(s.to_numpy(), ddof=1))
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument("--out_raw", type=Path, required=True)
    ap.add_argument("--out_tidy", type=Path, required=True)
    ap.add_argument("--drop_incomplete_pairs", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    
    required = [B0,B1,S0,S1,BS,BL,CAT,YMONO,YDI]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    raw = pd.DataFrame({
        "b0": df.get(B0).map(_s),
        "b1": df.get(B1).map(_s),
        "s0": df.get(S0).map(_s),
        "s1": df.get(S1).map(_s),
        "bs": df.get(BS).map(_s),
        "bl": df.get(BL).map(_s),
        "c":  df.get(CAT).map(_s),
        "yield_mono": pd.to_numeric(df.get(YMONO), errors="coerce"),
        "yield_di":   pd.to_numeric(df.get(YDI),   errors="coerce"),
    })
    if RID in df.columns:
        raw["reaction_id"] = df[RID]

    for col in ("yield_mono", "yield_di"):
        raw.loc[(raw[col] < 0) | (raw[col] > 100), col] = np.nan

    raw["cond_id"] = raw.apply(_make_cond_id, axis=1)

    if args.drop_incomplete_pairs:
        raw = raw.dropna(subset=["yield_mono", "yield_di"]).copy()

    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_raw, index=False)

    tidy = (
        raw.groupby("cond_id", dropna=False)
           .agg(
               s0=("s0", "first"),
               s1=("s1", "first"),
               b0=("b0", "first"),
               b1=("b1", "first"),
               bs=("bs", "first"),
               bl=("bl", "first"),
               c=("c", "first"),

               n_mono=("yield_mono", lambda s: int(s.notna().sum())),
               mean_yield_mono=("yield_mono", "mean"),
               var_yield_mono=("yield_mono", _var_or_nan),

               n_di=("yield_di", lambda s: int(s.notna().sum())),
               mean_yield_di=("yield_di", "mean"),
               var_yield_di=("yield_di", _var_or_nan),
           )
           .reset_index()
    )

    args.out_tidy.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(args.out_tidy, index=False)

    print(f"[raw]  {args.out_raw}  rows={len(raw)}  unique_cond_id={raw['cond_id'].nunique()}")
    print(f"[tidy] {args.out_tidy} rows={len(tidy)}")


if __name__ == "__main__":
    main()