"""
NAISC Singtel 2026 challenge pipeline — `src/main.py`.

End-to-end flow (see `utils.py` module docstring for algorithm detail):

1. **Detect** — `DriftDetector.detect` on raw train/test feature columns → `drift_table`
   (per-feature `drift_detected`, tests, PSI, structural flags) + drift-classifier AUC.
2. **Quantify** — Each row includes `severity` (`low` / `medium` / `high`); exported in
   `drift_table.csv` and summarized in the printed / CSV challenge drift table via
   `build_challenge_drift_table`.
3. **Mitigate** — `DriftMitigator.apply` runs several strategies (toggleable); this
   script evaluates ablations on a time-based or stratified validation split, selects
   the winning variant by AU-PRC, fits final LightGBM on full training data, and writes
   `prediction.csv`, `prediction.txt`, `model.joblib`, plus CSV/text artifacts:
   `drift_detection_summary.csv`, `drift_table.csv`, `drift_mitigation_table.csv`,
   `drift_mitigation_table.txt`, `ablation_results.csv`, `runtime_summary.csv`,
   `model_performance.csv`.
"""

import argparse
import io
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import DriftDetector, DriftMitigator, _skew_direction_label

# Keep drift + ablation passes bounded on multi-million-row CPU runs (~10 min organiser budget).
IMPORTANCE_FIT_MAX_ROWS = 300_000
ABLATION_FIT_MAX_ROWS = 350_000
ABLATION_VAL_MAX_ROWS = 120_000
SCALABILITY_LOG_MIN_ROWS = 200_000
# Console preview of prediction.csv (organiser example format); full scores only on disk.
PREDICTION_PREVIEW_ROWS = 5
# Each ablation runs full mitigation on all train/test rows then a LightGBM fit — dominant cost at huge n.
ABLATION_FULL_VARIANT_ROW_THRESHOLD = 400_000
SCALABILITY_TIGHT_ROW_THRESHOLD = 800_000
IMPORTANCE_FIT_MAX_ROWS_TIGHT = 150_000
ABLATION_FIT_MAX_ROWS_TIGHT = 200_000
ABLATION_VAL_MAX_ROWS_TIGHT = 80_000
# DriftDetector already subsamples heavy stats; tighten further only for very large frames.
DETECTOR_TIGHT_ROW_THRESHOLD = 1_200_000


def _subsample_index_stratified(idx: np.ndarray, y_aligned: pd.Series, max_n: int, random_state: int) -> np.ndarray:
    if len(idx) <= max_n:
        return idx
    y_sub = y_aligned.reindex(idx)
    sub, _ = train_test_split(idx, train_size=max_n, stratify=y_sub, random_state=random_state)
    return np.sort(sub)


def _ablation_variants(n_train_rows: int) -> Dict[str, dict]:
    """Full grid on typical/public scale; fewer variants when train is huge (mitigation is O(variants×n))."""
    full: Dict[str, dict] = {
        "baseline": {},
        "scaling_only": dict(use_scaling=True),
        "scaling+delta": dict(use_scaling=True, use_delta=True),
        "scaling+delta+binning": dict(use_scaling=True, use_delta=True, use_binning=True),
        "scaling+realignment": dict(use_scaling=True, use_realignment=True),
        "sliding_window": dict(use_scaling=True, use_delta=True, use_sliding_window=True),
        "weighted_decay": dict(use_scaling=True, use_delta=True, use_weighted_decay=True),
        "full_policy": dict(
            use_scaling=True,
            use_delta=True,
            use_seasonality=True,
            use_pruning=True,
        ),
        "full+binning+realign": dict(
            use_scaling=True,
            use_delta=True,
            use_seasonality=True,
            use_pruning=True,
            use_binning=True,
            use_realignment=True,
        ),
    }
    if n_train_rows >= ABLATION_FULL_VARIANT_ROW_THRESHOLD:
        return {
            "baseline": {},
            "scaling_only": dict(use_scaling=True),
            "scaling+delta": dict(use_scaling=True, use_delta=True),
            "scaling+realignment": dict(use_scaling=True, use_realignment=True),
            "weighted_decay": dict(use_scaling=True, use_delta=True, use_weighted_decay=True),
            "full_policy": dict(
                use_scaling=True,
                use_delta=True,
                use_seasonality=True,
                use_pruning=True,
            ),
        }
    return full


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


def train_lightgbm_weighted(x_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray | None = None):
    import lightgbm as lgb
    if LIGHTGBM_PARAMS != REQUIRED_LIGHTGBM_PARAMS:
        raise ValueError("LightGBM parameters must exactly match challenge requirements.")
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def get_feature_importance(model, columns: List[str]) -> Dict[str, float]:
    if hasattr(model, "feature_importances_"):
        return {c: float(v) for c, v in zip(columns, model.feature_importances_)}
    return {c: 0.0 for c in columns}


def _friendly_dtype_name(dtype_name: str) -> str:
    lowered = dtype_name.lower()
    if "int" in lowered:
        return "Int"
    if "float" in lowered:
        return "Float"
    return "Object"


def _row_flag(row: pd.Series, key: str) -> bool:
    v = row[key] if key in row.index else None
    if v is None or pd.isna(v):
        return False
    return bool(v)


# Challenge rubric: short, fixed phrasing like the required-outputs slide (not KS/PSI prose).
TEXT_NEW_LEVELS = (
    "Feature has new set of categorical features in test set compared to training set."
)
TEXT_LEFT_SKEW = (
    "Feature demonstrates greater left-skewness in test set compared to training set."
)
TEXT_RIGHT_SKEW = (
    "Feature demonstrates greater right-skewness in test set compared to training set."
)
TEXT_RANGE_EXPLODE = "Feature ranges explode in test set."


def _skew_tail_sentence(row: pd.Series) -> str:
    """Append sample skewness (train vs test) and left/right shift wording for numeric features."""
    if str(row.get("feature_type")) != "numerical":
        return ""
    raw_tr = row.get("train_skewness")
    raw_te = row.get("test_skewness")
    try:
        s_tr = float(raw_tr)
        s_te = float(raw_te)
    except (TypeError, ValueError):
        return ""
    if np.isnan(s_tr) or np.isnan(s_te):
        return ""
    lbl_tr = _skew_direction_label(s_tr)
    lbl_te = _skew_direction_label(s_te)
    out = (
        f" Train sample skewness ~{s_tr:.3f} ({lbl_tr}); "
        f"test ~{s_te:.3f} ({lbl_te})."
    )
    if s_te < s_tr - 0.2:
        out += (
            " Shape drift: test skewness is lower than train "
            "(distribution relatively less right-heavy or more left-heavy vs train)."
        )
    elif s_te > s_tr + 0.2:
        out += (
            " Shape drift: test skewness is higher than train "
            "(distribution relatively more right-heavy or less left-heavy vs train)."
        )
    else:
        out += " Skewness is broadly comparable between train and test."
    return out


def _challenge_drift_description(row: pd.Series, raw_dtype_map: Dict[str, str]) -> str:
    feat = str(row["feature"])
    ctype = _friendly_dtype_name(raw_dtype_map.get(feat, "object"))

    if row["feature_type"] == "categorical" and _row_flag(row, "unseen_categories"):
        return TEXT_NEW_LEVELS
    if (
        row["feature_type"] == "numerical"
        and ctype == "Int"
        and _row_flag(row, "skew_left_stronger_in_test")
    ):
        return TEXT_LEFT_SKEW + _skew_tail_sentence(row)
    if (
        row["feature_type"] == "numerical"
        and ctype == "Int"
        and _row_flag(row, "skew_right_stronger_in_test")
    ):
        return TEXT_RIGHT_SKEW + _skew_tail_sentence(row)
    if (
        row["feature_type"] == "numerical"
        and ctype == "Float"
        and _row_flag(row, "range_expansion")
    ):
        return TEXT_RANGE_EXPLODE + _skew_tail_sentence(row)

    eff = row.get("effect_size")
    eff_str = ""
    if eff is not None and not (isinstance(eff, float) and np.isnan(eff)):
        eff_str = f" Effect size: {float(eff):.3f}."

    if row["feature_type"] == "categorical":
        return f"Categorical distribution differs between training and test sets.{eff_str}"
    if row["feature_type"] == "numerical":
        return f"Numeric distribution differs between training and test sets.{eff_str}" + _skew_tail_sentence(row)
    return f"Distribution differs between training and test sets.{eff_str}"


def _challenge_drift_mitigation(
    row: pd.Series, mitigation_map: Dict[str, str], raw_dtype_map: Dict[str, str]
) -> str:
    feat = str(row["feature"])
    ctype = _friendly_dtype_name(raw_dtype_map.get(feat, "object"))

    if row["feature_type"] == "categorical" and _row_flag(row, "unseen_categories"):
        return "Drop Feature"
    if (
        row["feature_type"] == "numerical"
        and ctype == "Int"
        and (
            _row_flag(row, "skew_left_stronger_in_test")
            or _row_flag(row, "skew_right_stronger_in_test")
        )
    ):
        return "Seasonality Matching"
    if row["feature_type"] == "numerical" and ctype == "Float":
        return "Feature Scaling"

    return _friendly_mitigation_name(mitigation_map.get(feat, "none"))


def _friendly_mitigation_name(value: str) -> str:
    mapping = {
        "log/robust_scaling": "Feature Scaling",
        "pruned_high_drift_low_importance": "Drop Feature",
        "drop_unseen_categories": "Drop Feature",
        "categorical_monitoring": "Category Monitoring",
        "seasonality_matching": "Seasonality Matching",
        "binning": "Binning / Discretization",
        "input_realignment": "Input Re-Alignment",
        "weighted_decay": "Weighted Decay",
        "sliding_window": "Sliding Window",
        "none": "None",
    }
    return mapping.get(value, value.replace("_", " ").title())


def _ascii_table(df: pd.DataFrame, wrap_map: Dict[str, int]) -> str:
    cols = list(df.columns)
    wrapped_rows = []
    widths = {}
    for c in cols:
        widths[c] = max(len(c), wrap_map.get(c, len(c)))

    for _, row in df.iterrows():
        row_cells = {}
        max_lines = 1
        for c in cols:
            txt = str(row[c])
            wrap_w = wrap_map.get(c, 30)
            lines = textwrap.wrap(txt, width=wrap_w) or [""]
            row_cells[c] = lines
            max_lines = max(max_lines, len(lines))
            widths[c] = max(widths[c], max(len(l) for l in lines))
        wrapped_rows.append((row_cells, max_lines))

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    out_lines = [sep]
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    out_lines.extend([header, sep])
    for row_cells, max_lines in wrapped_rows:
        for i in range(max_lines):
            parts = []
            for c in cols:
                lines = row_cells[c]
                parts.append((lines[i] if i < len(lines) else "").ljust(widths[c]))
            out_lines.append("| " + " | ".join(parts) + " |")
        out_lines.append(sep)
    return "\n".join(out_lines)


def _infer_wrap_map(df: pd.DataFrame, default: int = 40) -> Dict[str, int]:
    wm: Dict[str, int] = {}
    for c in df.columns:
        lens = [len(str(c))]
        for v in df[c].head(200):
            lens.append(len(str(v)))
        mx = max(lens) if lens else default
        mx = max(mx, len(str(c)))
        wm[c] = min(mx, 72)
    return wm


def print_ascii_dataframe(
    df: pd.DataFrame,
    title: str | None = None,
    wrap_map: Dict[str, int] | None = None,
) -> None:
    """Print organiser-style +---+ ASCII tables (no tabulate/grid)."""
    if title:
        print(f"\n{title}")
    wm = wrap_map if wrap_map is not None else _infer_wrap_map(df)
    print(_ascii_table(df, wm))


def build_challenge_drift_table(
    drift_table: pd.DataFrame, mitigation_map: Dict[str, str], raw_dtype_map: Dict[str, str]
) -> pd.DataFrame:
    only_drift = drift_table[drift_table["drift_detected"]].copy()
    if only_drift.empty:
        return pd.DataFrame(
            columns=["Columns with Drift", "Column Type", "Drift Description", "Drift Mitigation"]
        )

    only_drift["Columns with Drift"] = only_drift["feature"]
    only_drift["Column Type"] = only_drift["feature"].map(
        lambda c: _friendly_dtype_name(raw_dtype_map.get(c, "object"))
    )
    only_drift["Drift Description"] = only_drift.apply(
        lambda r: _challenge_drift_description(r, raw_dtype_map), axis=1
    )
    only_drift["Drift Mitigation"] = only_drift.apply(
        lambda r: _challenge_drift_mitigation(r, mitigation_map, raw_dtype_map), axis=1
    )
    out = only_drift[
        ["Columns with Drift", "Column Type", "Drift Description", "Drift Mitigation"]
    ]
    return out


def save_outputs(model, test_ids: pd.Series, test_proba: np.ndarray, output_dir: Path = Path(".")) -> None:
    """Binary + prediction deliverables (organisers: prediction.csv in repo root)."""
    joblib.dump(model, output_dir / "model.joblib")
    pred_df = pd.DataFrame({"CustomerID": test_ids, "probability_score": test_proba})
    pred_df.to_csv(output_dir / "prediction.csv", index=False)
    buf = io.StringIO()
    pred_df.to_csv(buf, index=False, sep="\t", lineterminator="\n")
    (output_dir / "prediction.txt").write_text(buf.getvalue(), encoding="utf-8", newline="\n")


def write_csv_reports(
    output_dir: Path,
    drift_summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    drift_table: pd.DataFrame,
    challenge_drift_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    challenge_drift_ascii: str,
) -> None:
    """
    All tabular run outputs as CSV/text on disk (do not duplicate prediction rows here).

    Mirrors console tables: drift headline stats, ablation, runtime, performance, and the
    challenge drift column table. Full per-feature metrics only on disk (drift_table.csv).
    """
    drift_summary_df.to_csv(output_dir / "drift_detection_summary.csv", index=False)
    ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)
    drift_table.to_csv(output_dir / "drift_table.csv", index=False)
    challenge_drift_df.to_csv(output_dir / "drift_mitigation_table.csv", index=False)
    runtime_df.to_csv(output_dir / "runtime_summary.csv", index=False)
    perf_df.to_csv(output_dir / "model_performance.csv", index=False)
    (output_dir / "drift_mitigation_table.txt").write_text(challenge_drift_ascii, encoding="utf-8")


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
    y_train: pd.Series,
    use_scaling: bool = False,
    use_delta: bool = False,
    use_seasonality: bool = False,
    use_pruning: bool = False,
    use_binning: bool = False,
    use_realignment: bool = False,
    use_sliding_window: bool = False,
    use_weighted_decay: bool = False,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, Dict[str, str], List[str], np.ndarray | None]:
    train_month = train_df["Month"] if "Month" in train_df.columns else None
    test_month = test_df["Month"] if "Month" in test_df.columns else None
    mitigator = DriftMitigator()
    x_train_v, x_test_v, mitigation_map_v, dropped_v, sample_weights = mitigator.apply(
        x_train,
        x_test,
        drift_table,
        feature_importance,
        apply_scaling=use_scaling,
        apply_delta_features=use_delta,
        apply_seasonality=use_seasonality,
        apply_pruning=use_pruning,
        apply_binning=use_binning,
        apply_realignment=use_realignment,
        apply_sliding_window=use_sliding_window,
        apply_weighted_decay=use_weighted_decay,
        train_month=train_month,
        test_month=test_month,
        y_train=y_train,
    )
    tr_idx = x_tr.index.intersection(x_train_v.index)
    val_idx = x_val.index.intersection(x_train_v.index)
    if len(tr_idx) < 100 or len(val_idx) < 50:
        return 0.0, x_train_v, x_test_v, mitigation_map_v, dropped_v, sample_weights

    sw = None
    if sample_weights is not None:
        sw_series = pd.Series(sample_weights, index=x_train_v.index)
        sw = sw_series.loc[tr_idx].to_numpy()

    model_v = train_lightgbm_weighted(x_train_v.loc[tr_idx], y_train.loc[tr_idx], sw)
    val_proba_v = model_v.predict_proba(x_train_v.loc[val_idx])[:, 1]
    val_auprc_v = average_precision_score(y_train.loc[val_idx], val_proba_v)
    return val_auprc_v, x_train_v, x_test_v, mitigation_map_v, dropped_v, sample_weights


def main():
    """See module docstring at top of file for LightGBM params and pipeline stages.

    Console vs disk:
    - **Stdout:** progress lines, drift summary stats, mitigation selection, organiser-style
      ``+---+`` ASCII tables (drift / ablation / time taken / AU-PRC), first N-row prediction
      preview, file path list (full scores only in ``prediction.csv``).
    - **Disk:** all structured tables also written as CSV (and prediction.txt) via
      ``write_csv_reports`` + ``save_outputs``.
    """
    parser = argparse.ArgumentParser(
        description="NAISC Singtel 2026 challenge pipeline",
        epilog=(
            "Official run (from repo root): "
            "python ./src/main.py --train_data_filepath <train_data_filepath> "
            "--test_data_filepath <test_data_filepath>"
        ),
    )
    parser.add_argument(
        "--train_data_filepath",
        type=str,
        default="dataset/train.csv",
        help=(
            "Path to training CSV (organisers always pass this explicitly; "
            "default dataset/train.csv is for local dev only)"
        ),
    )
    parser.add_argument(
        "--test_data_filepath",
        type=str,
        default="dataset/test.csv",
        help=(
            "Path to test CSV (organisers always pass this explicitly; "
            "default dataset/test.csv is for local dev only)"
        ),
    )
    args = parser.parse_args()

    total_start = time.time()
    train_df, test_df = load_data(args.train_data_filepath, args.test_data_filepath)
    x_train, y_train, x_test, test_ids, features = prepare_features(train_df, test_df)
    x_train_raw, x_test_raw = get_raw_feature_frames(train_df, test_df, features)
    raw_dtype_map = {col: str(train_df[col].dtype) for col in features if col in train_df.columns}

    x_tr, y_tr, x_val, y_val = build_validation_split(train_df, x_train, y_train)

    scalability_notes: List[str] = []
    n_train_all = len(train_df)
    imp_cap = IMPORTANCE_FIT_MAX_ROWS
    abl_cap = ABLATION_FIT_MAX_ROWS
    val_cap = ABLATION_VAL_MAX_ROWS
    if n_train_all >= SCALABILITY_TIGHT_ROW_THRESHOLD:
        imp_cap = IMPORTANCE_FIT_MAX_ROWS_TIGHT
        abl_cap = ABLATION_FIT_MAX_ROWS_TIGHT
        val_cap = ABLATION_VAL_MAX_ROWS_TIGHT
        scalability_notes.append("tight_importance_ablation_caps")
    if n_train_all >= ABLATION_FULL_VARIANT_ROW_THRESHOLD:
        scalability_notes.append("reduced_ablation_variants")
    if n_train_all >= DETECTOR_TIGHT_ROW_THRESHOLD:
        scalability_notes.append("drift_detector_tight_caps")

    if len(x_tr) > imp_cap:
        idx_imp = _subsample_index_stratified(x_tr.index.to_numpy(), y_tr, imp_cap, 44)
        importance_model = train_lightgbm(x_tr.loc[idx_imp], y_tr.loc[idx_imp])
        importance = get_feature_importance(importance_model, list(x_tr.columns))
        scalability_notes.append(f"feature_importance_fit_rows={len(idx_imp)}")
    else:
        _importance_model = train_lightgbm(x_tr, y_tr)
        importance = get_feature_importance(_importance_model, list(x_tr.columns))

    if len(x_tr) > abl_cap:
        idx_ab = _subsample_index_stratified(x_tr.index.to_numpy(), y_train, abl_cap, 45)
        x_tr_ab, y_tr_ab = x_tr.loc[idx_ab], y_tr.loc[idx_ab]
        scalability_notes.append(f"ablation_train_fit_rows={len(idx_ab)}")
    else:
        x_tr_ab, y_tr_ab = x_tr, y_tr

    if len(x_val) > val_cap:
        idx_va = _subsample_index_stratified(x_val.index.to_numpy(), y_val, val_cap, 46)
        x_val_ab, y_val_ab = x_val.loc[idx_va], y_val.loc[idx_va]
        scalability_notes.append(f"ablation_val_rows={len(idx_va)}")
    else:
        x_val_ab, y_val_ab = x_val, y_val

    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION AND MITIGATION SUMMARY")
    print("(Drift findings, mitigation, runtime, AU-PRC — organiser-required console format.)")
    print("=" * 70)
    if len(train_df) >= SCALABILITY_LOG_MIN_ROWS or scalability_notes:
        extra = "; ".join(scalability_notes) if scalability_notes else "detector subsampling only"
        print(
            f"[Scalability] Large data path active (train n={len(train_df)}). "
            f"Caps reduce worst-case CPU time; {extra}. See README."
        )
    drift_start = time.time()

    print("\n[1/3] Detecting data drift (train vs test, per feature)...")
    detector = DriftDetector(alpha=0.05, psi_threshold=0.1)
    if n_train_all >= DETECTOR_TIGHT_ROW_THRESHOLD:
        detector.max_rows_per_split_stat_tests = 80_000
        detector.max_rows_skew_and_range = 80_000
        detector.max_rows_domain_classifier_per_split = 100_000
    drift_table, drift_info = detector.detect(x_train_raw, x_test_raw, features)
    print("[2/3] Drift detection summary:")
    print(f"  - Columns with detected drift: {drift_info['features_with_drift']} / {drift_info['total_features']}")
    print(f"  - Drift percentage: {drift_info['drift_percentage']:.2f}%")
    print(f"  - Global drift classifier AUC (train vs test domain): {drift_info['drift_classifier_auc']:.6f}")
    print("\n[3/3] Applying mitigation strategies (ablation + selected pipeline)...")
    variants = _ablation_variants(n_train_all)

    ablation_rows = []
    variant_store = {}
    for name, kwargs in variants.items():
        val_score, x_train_v, x_test_v, mitigation_map_v, dropped_v, sw_v = evaluate_variant(
            name,
            train_df,
            test_df,
            drift_table,
            importance,
            x_train,
            x_test,
            x_tr_ab,
            y_tr_ab,
            x_val_ab,
            y_val_ab,
            y_train,
            **kwargs,
        )
        ablation_rows.append(
            {
                "variant": name,
                "val_auprc": float(val_score),
                "strategies": ", ".join(k.replace("use_", "") for k, v in kwargs.items() if v) or "none",
                "pruned_features_count": len(dropped_v),
            }
        )
        variant_store[name] = (x_train_v, x_test_v, mitigation_map_v, dropped_v, sw_v)

    ablation_df = pd.DataFrame(ablation_rows).sort_values("val_auprc", ascending=False)
    best_variant = ablation_df.iloc[0]["variant"]
    x_train_m, x_test_m, mitigation_map, dropped, best_sw = variant_store[best_variant]

    drift_elapsed = time.time() - drift_start
    base_val_auprc = float(ablation_df.loc[ablation_df["variant"] == "baseline", "val_auprc"].iloc[0])
    best_val_auprc = float(ablation_df.iloc[0]["val_auprc"])
    print(f"  - Validation AU-PRC (baseline): {base_val_auprc:.6f}")
    print(f"  - Validation AU-PRC (best variant): {best_val_auprc:.6f}")
    print(f"  - Selected training mode: {best_variant}")
    if dropped:
        print(f"  - Pruned features: {', '.join(dropped)}")
    challenge_drift_df = build_challenge_drift_table(drift_table, mitigation_map, raw_dtype_map)
    challenge_drift_ascii = _ascii_table(
        challenge_drift_df,
        wrap_map={
            "Columns with Drift": 24,
            "Column Type": 12,
            "Drift Description": 60,
            "Drift Mitigation": 24,
        },
    )
    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION AND MITIGATION")
    print("=" * 70)
    print(challenge_drift_ascii)
    ablation_show = ablation_df.rename(
        columns={
            "variant": "Variant",
            "val_auprc": "AU-PRC",
            "strategies": "Strategies",
            "pruned_features_count": "N pruned",
        }
    ).copy()
    ablation_show["AU-PRC"] = np.round(ablation_show["AU-PRC"].astype(float), 6)
    print_ascii_dataframe(
        ablation_show,
        title="ABLATION SUMMARY (VALIDATION AU-PRC)",
        wrap_map={
            "Variant": 22,
            "AU-PRC": 10,
            "Strategies": 52,
            "N pruned": 8,
        },
    )

    final_x_train = x_train_m
    final_x_test = x_test_m

    final_sw = None
    if best_sw is not None:
        sw_series = pd.Series(best_sw, index=x_train_m.index)
        valid_idx = final_x_train.index.intersection(sw_series.index)
        final_sw = sw_series.loc[valid_idx].to_numpy() if len(valid_idx) == len(final_x_train) else None
    model = train_lightgbm_weighted(final_x_train, y_train, final_sw)
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
    runtime_df = pd.DataFrame(
        [
            {
                "Stage": "Drift detection and mitigation",
                "Time Taken (s)": round(drift_elapsed, 2),
            },
            {"Stage": "Total end-to-end", "Time Taken (s)": round(total_elapsed, 2)},
        ]
    )
    perf_df = pd.DataFrame(
        [
            {"": "Train Set", "AU-PRC": round(float(train_auprc), 3)},
            {
                "": "Test Set",
                "AU-PRC": (
                    "N/A (no labels in test CSV)"
                    if test_auprc is None
                    else round(float(test_auprc), 3)
                ),
            },
        ]
    )
    drift_summary_df = pd.DataFrame(
        [
            {
                "total_features": drift_info["total_features"],
                "features_with_drift": drift_info["features_with_drift"],
                "drift_percentage": round(float(drift_info["drift_percentage"]), 4),
                "drift_classifier_auc": round(float(drift_info["drift_classifier_auc"]), 6),
                "selected_mitigation_variant": best_variant,
                "validation_auprc_baseline": round(float(base_val_auprc), 6),
                "validation_auprc_best": round(float(best_val_auprc), 6),
                "dropped_or_pruned_features": ";".join(dropped) if dropped else "",
            }
        ]
    )
    out_dir = Path(".").resolve()
    write_csv_reports(
        out_dir,
        drift_summary_df,
        ablation_df,
        drift_table,
        challenge_drift_df,
        runtime_df,
        perf_df,
        challenge_drift_ascii,
    )

    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION AND MITIGATION - TIME TAKEN")
    print("=" * 70)
    print_ascii_dataframe(runtime_df, wrap_map=_infer_wrap_map(runtime_df, default=24))

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    perf_ascii = perf_df.copy()

    def _fmt_auprc_cell(x: object) -> str:
        if isinstance(x, str):
            return x
        return f"{float(x):.3f}"

    perf_ascii["AU-PRC"] = perf_ascii["AU-PRC"].map(_fmt_auprc_cell)
    print_ascii_dataframe(
        perf_ascii,
        wrap_map={"": 12, "AU-PRC": 22},
    )

    n_prev = min(PREDICTION_PREVIEW_ROWS, len(test_ids))
    pred_preview = pd.DataFrame(
        {
            "CustomerID": test_ids.astype(str).iloc[:n_prev].to_numpy(),
            "probability_score": np.round(test_proba[:n_prev], 3),
        }
    )
    print("\n" + "=" * 70)
    print(
        f"TEST SET PREDICTED PROBABILITIES (first {n_prev} rows; full file: prediction.csv)"
    )
    print("=" * 70)
    print_ascii_dataframe(
        pred_preview,
        wrap_map={"CustomerID": 36, "probability_score": 18},
    )

    print("\n" + "=" * 70)
    print("SAVED OUTPUT FILES (see also CSV list in main.py docstring)")
    print("=" * 70)
    print(f"  {out_dir / 'prediction.csv'}  [required submit format]")
    print(f"  {out_dir / 'model.joblib'}")
    print(f"  {out_dir / 'prediction.txt'}")
    print(f"  {out_dir / 'drift_mitigation_table.csv'}")
    print(f"  {out_dir / 'drift_mitigation_table.txt'}")
    print(f"  {out_dir / 'drift_detection_summary.csv'}")
    print(f"  {out_dir / 'drift_table.csv'}")
    print(f"  {out_dir / 'ablation_results.csv'}")
    print(f"  {out_dir / 'runtime_summary.csv'}")
    print(f"  {out_dir / 'model_performance.csv'}")


if __name__ == "__main__":
    main()
