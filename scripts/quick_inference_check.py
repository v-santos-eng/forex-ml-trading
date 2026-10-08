import os
import json
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_CANDIDATES = [
    "EURJPY",
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCHF",
    "AUDJPY",
    "GBPJPY",
    "USDNOK",
    "USDSEK",
    "XAUUSD",
]


def pick_pair_with_artifacts(base_dir: str, processed_dir_base: str, preferred: str = "") -> str:
    # processed_dir_base deve apontar para diretório base contendo subpastas por par
    if preferred:
        mdir = os.path.join(base_dir, "trained_models", preferred)
        pdir = os.path.join(processed_dir_base, preferred)
        if os.path.isdir(mdir) and os.path.isdir(pdir):
            has_pkl = any(name.endswith(".pkl") for name in os.listdir(mdir))
            has_test = os.path.exists(os.path.join(pdir, "test.parquet"))
            if has_pkl and has_test:
                return preferred
        print(f"WARNING: Par preferido '{preferred}' não possui artefatos necessários no diretório processado informado. Selecionando automaticamente...")
    for p in PAIR_CANDIDATES:
        mdir = os.path.join(base_dir, "trained_models", p)
        pdir = os.path.join(processed_dir_base, p)
        if os.path.isdir(mdir) and os.path.isdir(pdir):
            has_pkl = any(name.endswith(".pkl") for name in os.listdir(mdir))
            has_test = os.path.exists(os.path.join(pdir, "test.parquet"))
            if has_pkl and has_test:
                return p
    return ""


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_parquet_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print("Parquet read failed (default engine):", e)
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as e2:
            print("Parquet read failed with pyarrow:", e2)
            try:
                return pd.read_parquet(path, engine="fastparquet")
            except Exception as e3:
                print("Parquet read failed with fastparquet:", e3)
                raise


def coerce_direction_labels(y: pd.Series) -> pd.Series:
    # Map {-1, 1} -> {0,1}; keep 0 as 0 if present
    uniq = set(pd.unique(y.dropna()))
    if -1 in uniq and 1 in uniq and 0 not in uniq:
        return y.map({-1: 0, 1: 1}).astype(int)
    if -1 in uniq and 1 in uniq and 0 in uniq:
        return y.map({-1: 0, 0: 0, 1: 1}).astype(int)
    # If already 0/1 or {0,1,2}, leave as is but cast to int
    try:
        return y.astype(int)
    except Exception:
        return y


def parse_args():
    parser = argparse.ArgumentParser(description="Quick inference sanity check for FX models.")
    parser.add_argument("--pair", type=str, default="", help="Par de moedas a validar (ex: EURJPY). Se ausente, escolhe automaticamente um par disponível.")
    parser.add_argument("--save_json", type=str, default="", help="Caminho do arquivo JSON para salvar métricas (opcional).")
    parser.add_argument("--processed_dir", type=str, default="", help="Diretório base para dados processados (ex.: processed_fx_data_M5). Se vazio, usa processed_fx_data padrão.")
    return parser.parse_args()


def main():
    args = parse_args()
    base = BASE_DIR

    # Resolve diretório base de dados processados
    if args.processed_dir:
        if os.path.isabs(args.processed_dir):
            pdir_base = args.processed_dir
        else:
            pdir_base = os.path.join(base, args.processed_dir)
    else:
        pdir_base = os.path.join(base, "processed_fx_data")

    pair = pick_pair_with_artifacts(base, pdir_base, args.pair)
    if not pair:
        print("ERROR: No pair found with both PKL model(s) and test.parquet in the provided processed_dir.")
        sys.exit(99)

    print(f"Selected pair: {pair}")
    mdir = os.path.join(base, "trained_models", pair)
    pdir = os.path.join(pdir_base, pair)

    # Prefer specific models in order
    prefer = [
        "xgboost_clf_target_1.pkl",
        "random_forest_clf_target_1.pkl",
        "xgboost_reg_target_1.pkl",
        "random_forest_reg_target_1.pkl",
        "gradient_boosting_reg_target_1.pkl",
    ]

    model_path = None
    for name in prefer:
        cand = os.path.join(mdir, name)
        if os.path.exists(cand):
            model_path = cand
            break
    if model_path is None:
        pkls = sorted(
            [os.path.join(mdir, f) for f in os.listdir(mdir) if f.endswith(".pkl")]
        )
        if pkls:
            model_path = pkls[0]

    if not model_path:
        print("ERROR: No model .pkl found for", pair)
        sys.exit(98)

    print("Loading model:", os.path.basename(model_path))
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Loaded model type:", type(model))

    fname = os.path.basename(model_path).lower()
    model_kind = "classification" if "_clf_" in fname else "regression"

    # Try to load model-specific metadata to get exact feature list
    model_meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
    model_features = None
    if os.path.exists(model_meta_path):
        try:
            mmeta = load_json(model_meta_path)
            for key in [
                "feature_columns",
                "features",
                "feature_names",
                "features_used",
            ]:
                if key in mmeta and isinstance(mmeta[key], list) and mmeta[key]:
                    model_features = list(mmeta[key])
                    break
            print(
                f"Model metadata found: {os.path.basename(model_meta_path)} | "
                f"Features listed: {len(model_features) if model_features else 0}"
            )
        except Exception as e:
            print("Warning: failed reading model metadata:", e)

    # Fallback to processed metadata
    proc_meta_path = os.path.join(pdir, "metadata.json")
    feature_cols = []
    target_cols = []
    timeframe = None
    if os.path.exists(proc_meta_path):
        try:
            meta = load_json(proc_meta_path)
            feature_cols = meta.get("feature_columns") or []
            target_cols = meta.get("target_columns") or []
            timeframe = meta.get("timeframe")
        except Exception as e:
            print("Warning: failed reading processed metadata:", e)

    # Load test set
    test_path = os.path.join(pdir, "test.parquet")
    df = read_parquet_any(test_path)
    print("Test df shape:", df.shape)

    # Decide feature set and build matrix X
    if model_features:
        needed = list(model_features)
    else:
        needed = list(feature_cols)

    if not needed:
        print("ERROR: Could not determine feature set from metadata.")
        # Try heuristic: all non-Target_ numeric columns
        candidates = [
            c
            for c in df.columns
            if not c.startswith("Target_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        if candidates:
            needed = candidates
            print("Heuristic features used:", len(needed))
        else:
            sys.exit(97)

    # Ensure all required columns exist (create missing as zeros)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing {len(missing)} feature columns in test df; filling with zeros.")
        for c in missing:
            df[c] = 0.0

    # Order columns exactly as needed
    X = df[needed].copy()

    # Identify target column
    period = 1
    ycol = None
    if model_kind == "classification":
        # prefer Target_Direction_{period}
        candidates = [c for c in df.columns if c.startswith("Target_Direction_")]
        for c in candidates:
            if c.endswith(f"_{period}"):
                ycol = c
                break
        if ycol is None and candidates:
            ycol = candidates[0]
    else:
        # prefer Target_Return_{period}
        candidates = [c for c in df.columns if c.startswith("Target_Return_")]
        for c in candidates:
            if c.endswith(f"_{period}"):
                ycol = c
                break
        if ycol is None and candidates:
            ycol = candidates[0]

    if not ycol:
        print("ERROR: Could not find suitable target column in test df")
        sys.exit(96)

    y = df[ycol].copy()

    # Drop NAs in X or y
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X2 = X.loc[mask]
    y2 = y.loc[mask]
    print("After NA filter:", X2.shape, y2.shape)

    # Predict
    try:
        y_pred = model.predict(X2)
    except Exception as e:
        print("Model.predict failed:", e)
        if model_kind == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X2)
            if proba.ndim == 2 and proba.shape[1] > 1:
                y_pred = (proba[:, 1] >= 0.5).astype(int)
            else:
                y_pred = (proba.ravel() >= 0.5).astype(int)
        else:
            raise

    # Metrics
    result = {
        "pair": pair,
        "model_path": os.path.relpath(model_path, BASE_DIR),
        "model_kind": model_kind,
        "n_samples": int(len(y2)),
        "metrics": {},
        "processed_dir": os.path.relpath(pdir, BASE_DIR),
    }
    if timeframe:
        result["timeframe"] = timeframe

    if model_kind == "classification":
        y_true = coerce_direction_labels(y2)
        y_hat = np.array(y_pred).astype(int)
        if set(pd.unique(y_true)) - {0, 1}:
            # If still not binary, try to binarize by sign
            y_true = (y2.values > 0).astype(int)
        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_hat, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_hat, average="weighted", zero_division=0)
        print(f"Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        print("Sample preds:", y_hat[:20].tolist())
        result["metrics"] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
        }
    else:
        y_true = y2.values.astype(float)
        y_hat = np.array(y_pred).astype(float)
        mse = mean_squared_error(y_true, y_hat)
        r2 = r2_score(y_true, y_hat)
        print(f"Metrics -> MSE: {mse:.8f} | R2: {r2:.4f}")
        print("Sample preds:", np.round(y_hat[:10], 6).tolist())
        result["metrics"] = {
            "MSE": float(mse),
            "R2": float(r2),
        }

    print(
        "SUCCESS: Inference sanity check completed for",
        pair,
        "using",
        os.path.basename(model_path),
    )

    # Save JSON if requested
    if args.save_json:
        out_path = args.save_json
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                print("WARNING: could not create directory for save_json:", e)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("Saved metrics JSON to:", out_path)
        except Exception as e:
            print("WARNING: failed to save metrics JSON:", e)


if __name__ == "__main__":
    main()