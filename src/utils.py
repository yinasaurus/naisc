"""
Utilities for NAISC Singtel 2026 drift-aware pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, RobustScaler


def _psi_numeric(train_s: pd.Series, test_s: pd.Series, bins: int = 10) -> float:
    train_arr = train_s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    test_arr = test_s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(train_arr) < 2 or len(test_arr) < 2:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(train_arr, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    train_hist, _ = np.histogram(train_arr, bins=edges)
    test_hist, _ = np.histogram(test_arr, bins=edges)

    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-6, None)
    test_pct = np.clip(test_hist / max(test_hist.sum(), 1), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def _psi_categorical(train_s: pd.Series, test_s: pd.Series) -> float:
    train_v = train_s.fillna("missing").astype(str)
    test_v = test_s.fillna("missing").astype(str)
    categories = sorted(set(train_v.unique()) | set(test_v.unique()))
    if not categories:
        return 0.0

    train_counts = train_v.value_counts().reindex(categories, fill_value=0).to_numpy()
    test_counts = test_v.value_counts().reindex(categories, fill_value=0).to_numpy()
    train_pct = np.clip(train_counts / max(train_counts.sum(), 1), 1e-6, None)
    test_pct = np.clip(test_counts / max(test_counts.sum(), 1), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def psi_severity(psi: float) -> str:
    if psi < 0.1:
        return "low"
    if psi < 0.25:
        return "medium"
    return "high"


def _integer_like_numeric(series: pd.Series) -> bool:
    """True if numeric column is integer dtype or all non-null values are whole numbers."""
    if pd.api.types.is_integer_dtype(series):
        return True
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    return np.allclose(s.values, np.round(s.values), rtol=0, atol=1e-6)


def _skew_shift_left_stronger_in_test(train_s: pd.Series, test_s: pd.Series, integer_like: bool) -> bool:
    """
    Detect materially stronger left skew in the test slice (challenge scenario for Int columns).
    Left-skewed distributions have negative sample skew; "greater left skew in test" means test skew is more negative.
    """
    if not integer_like:
        return False
    tr = pd.to_numeric(train_s, errors="coerce").dropna()
    te = pd.to_numeric(test_s, errors="coerce").dropna()
    if len(tr) < 50 or len(te) < 50:
        return False
    s_tr = float(skew(tr, bias=False))
    s_te = float(skew(te, bias=False))
    return (s_te < s_tr - 0.2) and (s_te < -0.15)


def _unseen_category_stats(train_s: pd.Series, test_s: pd.Series) -> Tuple[bool, float, int]:
    """Fraction of test rows with values never seen in train; count of unseen levels."""
    tr = train_s.fillna("missing").astype(str)
    te = test_s.fillna("missing").astype(str)
    train_vals = set(tr.unique())
    unseen_mask = ~te.isin(train_vals)
    n_unseen_levels = int(te[unseen_mask].nunique()) if unseen_mask.any() else 0
    frac = float(unseen_mask.mean()) if len(te) else 0.0
    # Meaningful structural drift: any unseen level used in test, or nontrivial row mass
    flag = bool(n_unseen_levels > 0 and frac > 0)
    return flag, frac, n_unseen_levels


def _float_range_expansion_ratio(train_s: pd.Series, test_s: pd.Series, is_float: bool) -> Tuple[bool, float]:
    """
    Compare spread of test vs train (robust IQR ratio) to catch 'range explosion' on float columns.
    """
    if not is_float:
        return False, 1.0
    tr = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    te = pd.to_numeric(test_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 30 or len(te) < 30:
        return False, 1.0
    iqr_tr = float(tr.quantile(0.75) - tr.quantile(0.25))
    iqr_te = float(te.quantile(0.75) - te.quantile(0.25))
    eps = 1e-9
    ratio = iqr_te / (iqr_tr + eps)
    # Also flag extreme max expansion
    span_tr = float(tr.quantile(0.99) - tr.quantile(0.01))
    span_te = float(te.quantile(0.99) - te.quantile(0.01))
    span_ratio = span_te / (span_tr + eps)
    flag = ratio >= 1.45 or span_ratio >= 1.6
    return flag, float(max(ratio, span_ratio))


def _combined_severity(psi: float, structural_high: bool) -> str:
    base = psi_severity(psi)
    if structural_high:
        order = {"low": 1, "medium": 2, "high": 3}
        rev = {1: "low", 2: "medium", 3: "high"}
        return rev[max(order[base], 2)]
    return base


@dataclass
class DriftDetector:
    alpha: float = 0.05
    psi_threshold: float = 0.1

    def detect(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, Dict]:
        rows: List[Dict] = []
        for col in features:
            tr = train_df[col]
            te = test_df[col]
            is_num = pd.api.types.is_numeric_dtype(tr)

            skew_left_test = False
            unseen_cat = False
            unseen_frac = 0.0
            n_unseen_levels = 0
            range_expand = False
            range_ratio = 1.0
            integer_like = False
            is_float_col = False

            if is_num:
                tr_num = pd.to_numeric(tr, errors="coerce")
                te_num = pd.to_numeric(te, errors="coerce")
                ks_stat, p_value = ks_2samp(tr_num.dropna(), te_num.dropna())
                psi = _psi_numeric(tr_num, te_num)
                statistical_drift = (p_value < self.alpha) or (psi >= self.psi_threshold)
                test_name = "KS+PSI+struct"
                drift_type = "numerical"
                integer_like = _integer_like_numeric(tr)
                is_float_col = bool(pd.api.types.is_float_dtype(tr))
                skew_left_test = _skew_shift_left_stronger_in_test(tr, te, integer_like)
                range_expand, range_ratio = _float_range_expansion_ratio(tr, te, is_float_col)
            else:
                tr_cat = tr.fillna("missing").astype(str)
                te_cat = te.fillna("missing").astype(str)
                categories = sorted(set(tr_cat.unique()) | set(te_cat.unique()))
                obs = np.vstack(
                    [
                        tr_cat.value_counts().reindex(categories, fill_value=0).to_numpy(),
                        te_cat.value_counts().reindex(categories, fill_value=0).to_numpy(),
                    ]
                )
                _, p_value, _, _ = chi2_contingency(obs)
                psi = _psi_categorical(tr_cat, te_cat)
                statistical_drift = (p_value < self.alpha) or (psi >= self.psi_threshold)
                ks_stat = np.nan
                test_name = "Chi2+PSI+struct"
                drift_type = "categorical"
                unseen_cat, unseen_frac, n_unseen_levels = _unseen_category_stats(tr, te)

            structural = skew_left_test or unseen_cat or range_expand
            drift_detected = bool(statistical_drift or structural)
            structural_high = unseen_cat or range_expand or skew_left_test
            sev = _combined_severity(float(psi), structural_high and drift_detected)

            rows.append(
                {
                    "feature": col,
                    "feature_type": drift_type,
                    "test_used": test_name,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "test_stat": float(ks_stat) if pd.notna(ks_stat) else np.nan,
                    "psi": float(psi),
                    "severity": sev,
                    "drift_detected": drift_detected,
                    "skew_left_stronger_in_test": skew_left_test,
                    "integer_like": integer_like,
                    "unseen_categories": unseen_cat,
                    "unseen_category_fraction": unseen_frac,
                    "n_unseen_levels": int(n_unseen_levels),
                    "range_expansion": range_expand,
                    "range_expansion_ratio": range_ratio,
                }
            )

        summary_df = pd.DataFrame(rows).sort_values(["drift_detected", "psi"], ascending=[False, False])
        drift_clf_auc = self._drift_classifier_auc(train_df, test_df, features)
        info = {
            "total_features": int(len(features)),
            "features_with_drift": int(summary_df["drift_detected"].sum()),
            "drift_percentage": float(summary_df["drift_detected"].mean() * 100 if len(summary_df) else 0.0),
            "drift_classifier_auc": float(drift_clf_auc),
        }
        return summary_df, info

    def _drift_classifier_auc(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> float:
        x_train = train_df[features].copy()
        x_test = test_df[features].copy()
        x_all = pd.concat([x_train, x_test], axis=0, ignore_index=True)
        y_domain = np.array([0] * len(x_train) + [1] * len(x_test))

        for c in features:
            if pd.api.types.is_numeric_dtype(x_all[c]):
                x_all[c] = pd.to_numeric(x_all[c], errors="coerce").fillna(x_all[c].median())
            else:
                enc = LabelEncoder()
                x_all[c] = enc.fit_transform(x_all[c].fillna("missing").astype(str))

        model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42)
        model.fit(x_all, y_domain)
        proba = model.predict_proba(x_all)[:, 1]
        return roc_auc_score(y_domain, proba)


class DriftMitigator:
    def __init__(self) -> None:
        self.scalers: Dict[str, RobustScaler] = {}

    def apply(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        drift_table: pd.DataFrame,
        feature_importance: Dict[str, float],
        apply_scaling: bool = True,
        apply_delta_features: bool = True,
        apply_seasonality: bool = True,
        apply_pruning: bool = True,
        train_month: Optional[pd.Series] = None,
        test_month: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], List[str]]:
        train_out = train_df.copy()
        test_out = test_df.copy()
        mitigation_map: Dict[str, str] = {str(r["feature"]): "none" for _, r in drift_table.iterrows()}
        dropped_columns: List[str] = []

        def _row_bool(row: pd.Series, key: str) -> bool:
            if key not in row.index:
                return False
            v = row[key]
            return bool(v) if not pd.isna(v) else False

        # Strategy 0: drop categoricals with unseen test levels (challenge mitigation)
        drop_unseen: List[str] = []
        for _, row in drift_table.iterrows():
            if not row["drift_detected"]:
                continue
            if row["feature_type"] != "categorical":
                continue
            if not _row_bool(row, "unseen_categories"):
                continue
            feat = str(row["feature"])
            if feat in train_out.columns:
                drop_unseen.append(feat)
                mitigation_map[feat] = "drop_unseen_categories"
        if drop_unseen:
            train_out = train_out.drop(columns=drop_unseen, errors="ignore")
            test_out = test_out.drop(columns=drop_unseen, errors="ignore")
            dropped_columns.extend(drop_unseen)

        # Strategy 1: robust/log scaling for numeric drifted features
        if apply_scaling:
            for _, row in drift_table.iterrows():
                feature = str(row["feature"])
                if feature not in train_out.columns:
                    continue
                if (not row["drift_detected"]) or row["feature_type"] != "numerical":
                    continue

                med = pd.to_numeric(train_out[feature], errors="coerce").median()
                tr_num = pd.to_numeric(train_out[feature], errors="coerce").fillna(med)
                te_num = pd.to_numeric(test_out[feature], errors="coerce").fillna(med)
                if tr_num.min() >= 0 and te_num.min() >= 0:
                    tr_num = np.log1p(tr_num)
                    te_num = np.log1p(te_num)

                scaler = RobustScaler()
                train_out[feature] = scaler.fit_transform(tr_num.to_frame()).ravel()
                test_out[feature] = scaler.transform(te_num.to_frame()).ravel()
                self.scalers[feature] = scaler

                skew_l = _row_bool(row, "skew_left_stronger_in_test")
                int_like = _row_bool(row, "integer_like")
                range_ex = _row_bool(row, "range_expansion")
                if skew_l and int_like and apply_seasonality:
                    mitigation_map[feature] = "seasonality_matching"
                elif row["feature_type"] == "numerical":
                    mitigation_map[feature] = "log/robust_scaling"
        else:
            for _, row in drift_table.iterrows():
                feature = str(row["feature"])
                if feature not in train_out.columns or not row["drift_detected"]:
                    continue
                if row["feature_type"] != "numerical":
                    continue
                skew_l = _row_bool(row, "skew_left_stronger_in_test")
                int_like = _row_bool(row, "integer_like")
                if skew_l and int_like and apply_seasonality:
                    mitigation_map[feature] = "seasonality_matching"
                else:
                    mitigation_map[feature] = "log/robust_scaling"

        # Strategy 2: delta-based numeric features
        if apply_delta_features:
            numeric_cols = [c for c in train_out.columns if pd.api.types.is_numeric_dtype(train_out[c])]
            for c in numeric_cols:
                median_val = pd.to_numeric(train_out[c], errors="coerce").median()
                train_out[f"{c}__delta_median"] = pd.to_numeric(train_out[c], errors="coerce") - median_val
                test_out[f"{c}__delta_median"] = pd.to_numeric(test_out[c], errors="coerce") - median_val

        # Strategy 3: seasonality overlap feature (Month must be passed — not in encoded X)
        if apply_seasonality and train_month is not None and test_month is not None:
            if len(train_month) == len(train_out) and len(test_month) == len(test_out):
                test_months = set(test_month.astype(str).dropna().unique())
                train_out["__seasonality_match"] = (
                    train_month.astype(str).isin(test_months).astype(int).values
                )
                test_out["__seasonality_match"] = 1

        # Strategy 4: drift-based pruning (high-drift + low-importance)
        pruned: List[str] = []
        if apply_pruning:
            high_drift = set(
                drift_table.loc[drift_table["severity"] == "high", "feature"].astype(str).tolist()
            )
            if feature_importance:
                imp_values = np.array(list(feature_importance.values()))
                threshold = float(np.quantile(imp_values, 0.35))
            else:
                threshold = 0.0
            pruned = [
                f
                for f in high_drift
                if f in train_out.columns and feature_importance.get(f, 0.0) <= threshold
            ]
            if pruned:
                train_out = train_out.drop(columns=pruned, errors="ignore")
                test_out = test_out.drop(columns=pruned, errors="ignore")
                for f in pruned:
                    mitigation_map[f] = "pruned_high_drift_low_importance"
                dropped_columns.extend(pruned)

        for _, row in drift_table.iterrows():
            feat = str(row["feature"])
            if not row["drift_detected"]:
                mitigation_map[feat] = "none"
                continue
            if mitigation_map.get(feat) in ("drop_unseen_categories", "pruned_high_drift_low_importance"):
                continue
            if row["feature_type"] == "numerical":
                if mitigation_map.get(feat) in ("seasonality_matching", "log/robust_scaling"):
                    continue
                skew_l = _row_bool(row, "skew_left_stronger_in_test")
                int_like = _row_bool(row, "integer_like")
                if skew_l and int_like and apply_seasonality:
                    mitigation_map[feat] = "seasonality_matching"
                else:
                    mitigation_map[feat] = "log/robust_scaling"
            elif row["feature_type"] == "categorical" and feat in train_out.columns:
                mitigation_map[feat] = "categorical_monitoring"

        return train_out, test_out, mitigation_map, dropped_columns

