#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TF_DIRS = [
    ("M1", "processed_fx_data"),
    ("M5", "processed_fx_data_M5"),
    ("M15", "processed_fx_data_M15"),
    ("M30", "processed_fx_data_M30"),
]

INDICATORS = [
    "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram",
    "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_Position",
    "Stoch_K", "Stoch_D", "ATR_14", "Williams_R", "CCI"
]

def load_processed_pair(base_dir: Path, pair: str) -> pd.DataFrame:
    pair_dir = base_dir / pair
    if not pair_dir.exists():
        return pd.DataFrame()
    parts = []
    for fname in ["train.parquet", "validation.parquet", "test.parquet"]:
        fpath = pair_dir / fname
        if fpath.exists():
            try:
                parts.append(pd.read_parquet(fpath))
            except Exception:
                pass
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    metrics["rows"] = len(df)
    metrics["cols"] = len(df.columns)

    # NaN ratio across all numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns
    metrics["nan_ratio_num"] = float(df[num_cols].isna().mean().mean()) if len(num_cols) else 0.0

    # Targets
    if "Target_Direction_1" in df.columns:
        td = df["Target_Direction_1"].dropna()
        if len(td):
            metrics["target_dir_1_pct_buy"] = float(td.mean())
        else:
            metrics["target_dir_1_pct_buy"] = np.nan
    if "Target_Return_1" in df.columns:
        tr = df["Target_Return_1"].dropna()
        metrics["target_ret_1_mean"] = float(tr.mean()) if len(tr) else np.nan
        metrics["target_ret_1_std"] = float(tr.std()) if len(tr) else np.nan

    # Indicator availability and basic stats
    for col in INDICATORS:
        if col in df.columns:
            s = df[col]
            metrics[f"{col}_nan"] = float(s.isna().mean())
            metrics[f"{col}_mean"] = float(s.dropna().mean()) if s.notna().any() else np.nan
        else:
            metrics[f"{col}_nan"] = np.nan
            metrics[f"{col}_mean"] = np.nan

    # Range checks
    def ratio_outside(s: pd.Series, lo: float, hi: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return float(((s < lo) | (s > hi)).mean())

    if "RSI_14" in df.columns:
        metrics["RSI_14_out_of_[0,100]"] = ratio_outside(df["RSI_14"], 0.0, 100.0)
    if "Stoch_K" in df.columns:
        metrics["Stoch_K_out_of_[0,100]"] = ratio_outside(df["Stoch_K"], 0.0, 100.0)
    if "BB_Position" in df.columns:
        metrics["BB_Position_out_of_[0,1]"] = ratio_outside(df["BB_Position"], 0.0, 1.0)
    if "ATR_14" in df.columns:
        s = df["ATR_14"].dropna()
        metrics["ATR_14_negative_ratio"] = float((s < 0).mean()) if len(s) else np.nan

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Quick MTF check for processed FX datasets")
    parser.add_argument("--pair", default="EURUSD", help="Par para análise (ex.: EURUSD)")
    args = parser.parse_args()
    pair = args.pair.upper()

    results = {}
    for tf, dirname in TF_DIRS:
        base_dir = Path(dirname)
        if not base_dir.exists():
            continue
        df = load_processed_pair(base_dir, pair)
        if df.empty:
            continue
        results[tf] = compute_metrics(df)

    if not results:
        print(f"Nenhum dataset encontrado para {pair} nas pastas processadas.")
        return

    # Pretty print summary
    print("\n==== Quick MTF Check ====")
    print(f"Pair: {pair}")
    for tf in sorted(results.keys(), key=lambda x: (0 if x=='M1' else int(x[1:]))) :
        m = results[tf]
        print(f"\n[{tf}] rows={m['rows']:,}, cols={m['cols']} | NaN(num)={m['nan_ratio_num']:.3f}")
        if "target_dir_1_pct_buy" in m:
            td = m.get("target_dir_1_pct_buy", np.nan)
            trm = m.get("target_ret_1_mean", np.nan)
            trs = m.get("target_ret_1_std", np.nan)
            print(f" Targets: pct_buy={td:.3f} | ret1_mean={trm:.6f} | ret1_std={trs:.6f}")
        # Range flags
        rsi_flag = m.get("RSI_14_out_of_[0,100]", np.nan)
        stoch_flag = m.get("Stoch_K_out_of_[0,100]", np.nan)
        bbpos_flag = m.get("BB_Position_out_of_[0,1]", np.nan)
        atr_neg = m.get("ATR_14_negative_ratio", np.nan)
        print(" Range checks:", end="")
        print(f" RSI_oor={rsi_flag:.3f}", end="")
        print(f" | StochK_oor={stoch_flag:.3f}", end="")
        print(f" | BBpos_oor={bbpos_flag:.3f}", end="")
        print(f" | ATR_neg={atr_neg:.5f}")

if __name__ == "__main__":
    main()