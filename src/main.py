"""Main entry point for NAISC Singtel 2026 challenge pipeline."""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import DriftDetector, DriftMitigator


LIGHTGBM_PARAMS = {
    "objective": "binary",
    "is_unbalance": True,
    "random_state": 42,
    "importance_type": "gain",
    "verbosity": -1,
}

REQUIRED_LIGHTGBM_PARAMS = {
    "objective": "binary",
    "is_unbalance": True,
    "random_state": 42,
    "importance_type": "gain",
    "verbosity": -1,
}


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    target_col = "ChurnStatus"
    id_col = "CustomerID"
    exclude = {id_col, target_col, "Month"}
    train_features = [c for c in train_df.columns if c not in exclude]
    test_features = [c for c in test_df.columns if c not in exclude]
    common_features = sorted(set(train_features).intersection(test_features))
    if not common_features:
        raise ValueError("No common train/test features found.")

    x_train = train_df[common_features].copy()
    x_test = test_df[common_features].copy()
    y_train = train_df[target_col].copy()
    test_ids = test_df[id_col].copy() if id_col in test_df.columns else pd.Series(test_df.index.astype(str), name=id_col)

    if y_train.dtype == "object":
        y_norm = y_train.astype(str).str.lower().str.strip()
        if set(y_norm.dropna().unique()).issubset({"yes", "no"}):
            y_train = (y_norm == "yes").astype(int)
        else:
            y_train = LabelEncoder().fit_transform(y_norm.fillna("missing"))
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)

    for c in common_features:
        if pd.api.types.is_numeric_dtype(x_train[c]):
            med = pd.to_numeric(x_train[c], errors="coerce").median()
            x_train[c] = pd.to_numeric(x_train[c], errors="coerce").fillna(med)
            x_test[c] = pd.to_numeric(x_test[c], errors="coerce").fillna(med)
        else:
            le = LabelEncoder()
            combined = pd.concat([x_train[c], x_test[c]], axis=0).fillna("missing").astype(str)
            le.fit(combined)
            x_train[c] = le.transform(x_train[c].fillna("missing").astype(str))
            x_test[c] = le.transform(x_test[c].fillna("missing").astype(str))

    return x_train, y_train, x_test, test_ids, common_features


def get_raw_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_train = train_df[features].copy()
    raw_test = test_df[features].copy()
    return raw_train, raw_test


def train_lightgbm(x_train: pd.DataFrame, y_train: pd.Series):
    import lightgbm as lgb
    if LIGHTGBM_PARAMS != REQUIRED_LIGHTGBM_PARAMS:
        raise ValueError(
            "LightGBM parameters must exactly match challenge requirements."
        )

    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(x_train, y_train)
    return model


def build_validation_split(
    train_df: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if "Month" in train_df.columns:
        month_series = train_df["Month"].astype(str)
        month_counts = month_series.value_counts().sort_index()
        if len(month_counts) >= 2:
            holdout_month = month_counts.index[-1]
            val_mask = month_series == holdout_month
            if val_mask.sum() > 100 and (~val_mask).sum() > 100:
                return (
                    x_train.loc[~val_mask],
                    y_train.loc[~val_mask],
                    x_train.loc[val_mask],
                    y_train.loc[val_mask],
                )

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    return x_tr, y_tr, x_val, y_val


def get_feature_importance(model, columns: List[str]) -> Dict[str, float]:
    if hasattr(model, "feature_importances_"):
        return {c: float(v) for c, v in zip(columns, model.feature_importances_)}
    return {c: 0.0 for c in columns}


def print_drift_table(drift_table: pd.DataFrame, mitigation_map: Dict[str, str]) -> None:
    table = drift_table.copy()
    table["mitigation"] = table["feature"].map(mitigation_map).fillna("none")
    table = table[["feature", "feature_type", "test_used", "p_value", "psi", "severity", "drift_detected", "mitigation"]]
    print_table(table, title="Columns with Drift")


def print_table(df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    try:
        from tabulate import tabulate

        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    except Exception:
        print(df.to_string(index=False, max_colwidth=40))


def save_outputs(model, test_ids: pd.Series, test_proba: np.ndarray, output_dir: Path = Path(".")) -> None:
    joblib.dump(model, output_dir / "model.joblib")
    pred_df = pd.DataFrame({"CustomerID": test_ids, "probability_score": test_proba})
    pred_df.to_csv(output_dir / "prediction.csv", index=False)


def evaluate_variant(
    variant_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drift_table: pd.DataFrame,
    feature_importance: Dict[str, float],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    use_scaling: bool,
    use_delta: bool,
    use_seasonality: bool,
    use_pruning: bool,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, Dict[str, str], List[str]]:
    mitigator = DriftMitigator()
    x_train_v, x_test_v, mitigation_map_v, dropped_v = mitigator.apply(
        x_train,
        x_test,
        drift_table,
        feature_importance,
        apply_scaling=use_scaling,
        apply_delta_features=use_delta,
        apply_seasonality=use_seasonality,
        apply_pruning=use_pruning,
    )
    model_v = train_lightgbm(x_train_v.loc[x_tr.index], y_tr)
    val_proba_v = model_v.predict_proba(x_train_v.loc[x_val.index])[:, 1]
    val_auprc_v = average_precision_score(y_val, val_proba_v)
    return val_auprc_v, x_train_v, x_test_v, mitigation_map_v, dropped_v


def main():
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 challenge pipeline")
    parser.add_argument(
        "--train_data_filepath",
        type=str,
        default="NAISC-Singtel-2026/public_data/train.csv",
        help="Path to training CSV (default: public_data/train.csv)",
    )
    parser.add_argument(
        "--test_data_filepath",
        type=str,
        default="NAISC-Singtel-2026/public_data/test.csv",
        help="Path to test CSV (default: public_data/test.csv)",
    )
    args = parser.parse_args()

    total_start = time.time()
    train_df, test_df = load_data(args.train_data_filepath, args.test_data_filepath)
    x_train, y_train, x_test, test_ids, features = prepare_features(train_df, test_df)
    x_train_raw, x_test_raw = get_raw_feature_frames(train_df, test_df, features)

    x_tr, y_tr, x_val, y_val = build_validation_split(train_df, x_train, y_train)
    baseline_model = train_lightgbm(x_tr, y_tr)
    importance = get_feature_importance(baseline_model, list(x_tr.columns))

    print("\n" + "=" * 60)
    print("DATA DRIFT DETECTION & MITIGATION")
    print("=" * 60)
    drift_start = time.time()

    detector = DriftDetector(alpha=0.05, psi_threshold=0.1)
    drift_table, drift_info = detector.detect(x_train_raw, x_test_raw, features)
    variants = [
        ("baseline", False, False, False, False),
        ("scaling_only", True, False, False, False),
        ("scaling_plus_delta", True, True, False, False),
        ("full_policy", True, True, True, True),
    ]

    ablation_rows = []
    variant_store = {}
    for name, use_scaling, use_delta, use_seasonality, use_pruning in variants:
        val_score, x_train_v, x_test_v, mitigation_map_v, dropped_v = evaluate_variant(
            name,
            train_df,
            test_df,
            drift_table,
            importance,
            x_train,
            x_test,
            x_tr,
            y_tr,
            x_val,
            y_val,
            use_scaling,
            use_delta,
            use_seasonality,
            use_pruning,
        )
        ablation_rows.append(
            {
                "variant": name,
                "val_auprc": float(val_score),
                "use_scaling": use_scaling,
                "use_delta": use_delta,
                "use_seasonality": use_seasonality,
                "use_pruning": use_pruning,
                "pruned_features_count": len(dropped_v),
            }
        )
        variant_store[name] = (x_train_v, x_test_v, mitigation_map_v, dropped_v)

    ablation_df = pd.DataFrame(ablation_rows).sort_values("val_auprc", ascending=False)
    best_variant = ablation_df.iloc[0]["variant"]
    x_train_m, x_test_m, mitigation_map, dropped = variant_store[best_variant]

    drift_elapsed = time.time() - drift_start
    print("\n[1/3] Detecting data drift...")
    print("[2/3] Drift Detection Summary:")
    print(f"  - Total features analyzed: {drift_info['total_features']}")
    print(f"  - Features with detected drift: {drift_info['features_with_drift']}")
    print(f"  - Drift percentage: {drift_info['drift_percentage']:.2f}%")
    print(f"  - Drift classifier AUC: {drift_info['drift_classifier_auc']:.6f}")
    print("\n[3/3] Applying mitigation strategies...")
    base_val_auprc = float(ablation_df.loc[ablation_df["variant"] == "baseline", "val_auprc"].iloc[0])
    best_val_auprc = float(ablation_df.iloc[0]["val_auprc"])
    print(f"  - Validation AU-PRC (baseline): {base_val_auprc:.6f}")
    print(f"  - Validation AU-PRC (best variant): {best_val_auprc:.6f}")
    print(f"  - Selected training mode: {best_variant}")
    if dropped:
        print(f"  - Pruned features: {', '.join(dropped)}")
    print_drift_table(drift_table, mitigation_map)
    print_table(ablation_df, title="Ablation summary (validation AU-PRC)")

    final_x_train = x_train_m
    final_x_test = x_test_m

    model = train_lightgbm(final_x_train, y_train)
    train_proba = model.predict_proba(final_x_train)[:, 1]
    test_proba = model.predict_proba(final_x_test)[:, 1]
    train_auprc = average_precision_score(y_train, train_proba)

    test_auprc = None
    if "ChurnStatus" in test_df.columns:
        y_test_raw = test_df["ChurnStatus"].copy()
        if y_test_raw.dtype == "object":
            y_test = (y_test_raw.astype(str).str.lower().str.strip() == "yes").astype(int)
        else:
            y_test = pd.to_numeric(y_test_raw, errors="coerce").fillna(0).astype(int)
        test_auprc = average_precision_score(y_test, test_proba)

    save_outputs(model, test_ids, test_proba)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("RUNTIME")
    print("=" * 60)
    runtime_df = pd.DataFrame(
        [
            {"metric": "Time taken for drift detection and mitigation (s)", "value": round(drift_elapsed, 2)},
            {"metric": "Total runtime (s)", "value": round(total_elapsed, 2)},
        ]
    )
    print_table(runtime_df)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    perf_rows = [{"dataset": "Train Set", "AU-PRC": round(float(train_auprc), 6)}]
    perf_rows.append(
        {
            "dataset": "Test Set",
            "AU-PRC": "N/A (test labels not available)"
            if test_auprc is None
            else round(float(test_auprc), 6),
        }
    )
    perf_df = pd.DataFrame(perf_rows)
    print_table(perf_df)

    ablation_df.to_csv("ablation_results.csv", index=False)
    drift_table.to_csv("drift_table.csv", index=False)


if __name__ == "__main__":
    main()
