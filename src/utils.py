"""
Utilities for NAISC Singtel 2026 drift-aware pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
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

            if is_num:
                tr_num = pd.to_numeric(tr, errors="coerce")
                te_num = pd.to_numeric(te, errors="coerce")
                ks_stat, p_value = ks_2samp(tr_num.dropna(), te_num.dropna())
                psi = _psi_numeric(tr_num, te_num)
                drift_detected = (p_value < self.alpha) or (psi >= self.psi_threshold)
                test_name = "KS+PSI"
                drift_type = "numerical"
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
                drift_detected = (p_value < self.alpha) or (psi >= self.psi_threshold)
                ks_stat = np.nan
                test_name = "Chi2+PSI"
                drift_type = "categorical"

            rows.append(
                {
                    "feature": col,
                    "feature_type": drift_type,
                    "test_used": test_name,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "test_stat": float(ks_stat) if pd.notna(ks_stat) else np.nan,
                    "psi": float(psi),
                    "severity": psi_severity(float(psi)),
                    "drift_detected": bool(drift_detected),
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], List[str]]:
        train_out = train_df.copy()
        test_out = test_df.copy()
        mitigation_map: Dict[str, str] = {}

        # Strategy 1: robust/log scaling for numeric drifted features
        for _, row in drift_table.iterrows():
            feature = row["feature"]
            if not row["drift_detected"]:
                mitigation_map[feature] = "none"
                continue

            if row["feature_type"] == "numerical":
                # log1p only when all values are non-negative
                tr_num = pd.to_numeric(train_out[feature], errors="coerce").fillna(train_out[feature].median())
                te_num = pd.to_numeric(test_out[feature], errors="coerce").fillna(train_out[feature].median())
                if tr_num.min() >= 0 and te_num.min() >= 0:
                    tr_num = np.log1p(tr_num)
                    te_num = np.log1p(te_num)

                scaler = RobustScaler()
                train_out[feature] = scaler.fit_transform(tr_num.to_frame()).ravel()
                test_out[feature] = scaler.transform(te_num.to_frame()).ravel()
                self.scalers[feature] = scaler
                mitigation_map[feature] = "log/robust_scaling"
            else:
                mitigation_map[feature] = "categorical_monitoring"

        # Strategy 2: delta-based numeric features
        numeric_cols = [c for c in train_out.columns if pd.api.types.is_numeric_dtype(train_out[c])]
        for c in numeric_cols:
            median_val = pd.to_numeric(train_out[c], errors="coerce").median()
            train_out[f"{c}__delta_median"] = pd.to_numeric(train_out[c], errors="coerce") - median_val
            test_out[f"{c}__delta_median"] = pd.to_numeric(test_out[c], errors="coerce") - median_val

        # Strategy 3: seasonality matching marker from Month (if present)
        if "Month" in train_df.columns and "Month" in test_df.columns:
            test_months = set(test_df["Month"].astype(str).dropna().unique())
            train_out["__seasonality_match"] = train_df["Month"].astype(str).isin(test_months).astype(int).values
            test_out["__seasonality_match"] = 1

        # Strategy 4: drift-based pruning (high-drift + low-importance)
        high_drift = set(drift_table.loc[drift_table["severity"] == "high", "feature"].tolist())
        if feature_importance:
            imp_values = np.array(list(feature_importance.values()))
            threshold = float(np.quantile(imp_values, 0.35))
        else:
            threshold = 0.0
        to_drop = [f for f in high_drift if feature_importance.get(f, 0.0) <= threshold]
        if to_drop:
            train_out = train_out.drop(columns=to_drop, errors="ignore")
            test_out = test_out.drop(columns=to_drop, errors="ignore")
            for f in to_drop:
                mitigation_map[f] = "pruned_high_drift_low_importance"

        return train_out, test_out, mitigation_map, to_drop

