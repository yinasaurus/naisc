"""
NAISC Singtel 2026 drift-aware pipeline (core library).

Three-stage design (maps to challenge / report expectations):

1. **Detect drift** — `DriftDetector.detect` compares each train vs test feature:
   - Numeric: KS test + PSI + Cohen's d (effect size) + structural cues (skew, range).
   - Categorical: Chi-square + PSI + Cramér's V (effect size) + unseen categories.
   - Composite drift score per feature (0–1) combining p-value, PSI, and effect size.
   - Drift classifier (logistic regression AUC) for global domain-separability.

2. **Quantify severity** — PSI bands → low / medium / high; structural signals
   elevate severity. Effect size and drift score provide additional ranking.

3. **Mitigate drift** — `DriftMitigator.apply` supports eight strategies:
   - **Robustness**: Log/robust scaling, delta-from-median features,
     binning/discretization.
   - **Recency**: Sliding window (recent N months), weighted decay (sample weights),
     seasonality matching.
   - **Monitoring**: Input re-alignment (center test to training norms),
     drift-based feature pruning.
   The CLI (`main.py`) ablates combinations, picks the best validation AU-PRC,
   then retrains LightGBM on the full mitigated training set.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, RobustScaler


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


def _cohens_d(train_s: pd.Series, test_s: pd.Series) -> float:
    """Effect size for numeric features (pooled std)."""
    tr = pd.to_numeric(train_s, errors="coerce").dropna().to_numpy()
    te = pd.to_numeric(test_s, errors="coerce").dropna().to_numpy()
    if len(tr) < 2 or len(te) < 2:
        return 0.0
    n1, n2 = len(tr), len(te)
    pooled_std = np.sqrt(((n1 - 1) * tr.std() ** 2 + (n2 - 1) * te.std() ** 2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float(abs(tr.mean() - te.mean()) / pooled_std)


def _cramers_v(train_s: pd.Series, test_s: pd.Series) -> float:
    """Effect size for categorical features."""
    tr = train_s.fillna("missing").astype(str)
    te = test_s.fillna("missing").astype(str)
    categories = sorted(set(tr.unique()) | set(te.unique()))
    if len(categories) < 2:
        return 0.0
    obs = np.vstack([
        tr.value_counts().reindex(categories, fill_value=0).to_numpy(),
        te.value_counts().reindex(categories, fill_value=0).to_numpy(),
    ])
    chi2, _, _, _ = chi2_contingency(obs)
    n = obs.sum()
    k = min(obs.shape[0], obs.shape[1])
    if n == 0 or k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


def _drift_score(p_value: float, psi: float, effect_size: float) -> float:
    """Composite per-feature drift score (0–1 scale) combining significance, PSI, and effect size."""
    p_score = min(1.0, max(0.0, 1.0 - p_value))
    psi_score = min(1.0, psi / 0.5)
    eff_score = min(1.0, effect_size / 1.0)
    return float(0.4 * p_score + 0.35 * psi_score + 0.25 * eff_score)


def _sample_1d_numeric_array(values: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Without-replacement subsample for heavy two-sample stats (KS, PSI, Cohen d)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) <= max_n:
        return v
    pick = rng.choice(len(v), size=max_n, replace=False)
    return v[pick]


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


def _skew_direction_label(skewness: float) -> str:
    """Human-readable skew shape (sample skewness: negative => left tail longer)."""
    if np.isnan(skewness):
        return "n/a"
    if skewness < -0.5:
        return "left-skewed"
    if skewness > 0.5:
        return "right-skewed"
    if skewness < -0.2:
        return "mildly left-skewed"
    if skewness > 0.2:
        return "mildly right-skewed"
    return "approximately symmetric"


def _numeric_skew_stats(train_s: pd.Series, test_s: pd.Series, min_n: int = 30) -> Tuple[float, float, bool, bool]:
    """
    Sample skewness for train/test and shift flags (material shift in skewness, any numeric).
    left_stronger_in_test: test more left-skewed than train (more negative skew).
    right_stronger_in_test: test more right-skewed than train.
    """
    tr = pd.to_numeric(train_s, errors="coerce").dropna()
    te = pd.to_numeric(test_s, errors="coerce").dropna()
    if len(tr) < min_n or len(te) < min_n:
        return float("nan"), float("nan"), False, False
    s_tr = float(skew(tr, bias=False))
    s_te = float(skew(te, bias=False))
    left_stronger = s_te < s_tr - 0.2
    right_stronger = s_te > s_tr + 0.2
    return s_tr, s_te, left_stronger, right_stronger


def _skew_shift_left_stronger_in_test(
    train_s: pd.Series, test_s: pd.Series, integer_like: bool, s_tr: float, s_te: float
) -> bool:
    """
    Challenge-style Int drift: materially stronger left skew in test (negative skew deepens in test).
    """
    if not integer_like or np.isnan(s_tr) or np.isnan(s_te):
        return False
    if len(pd.to_numeric(train_s, errors="coerce").dropna()) < 50:
        return False
    if len(pd.to_numeric(test_s, errors="coerce").dropna()) < 50:
        return False
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
    # Large-n safety (challenge ~10 min CPU): subsample heavy per-split computations only.
    max_rows_per_split_stat_tests: int = 100_000
    max_rows_skew_and_range: int = 100_000
    max_rows_domain_classifier_per_split: int = 150_000

    def detect(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, Dict]:
        rng = np.random.default_rng(42)
        rows: List[Dict] = []
        for col in features:
            tr = train_df[col]
            te = test_df[col]
            is_num = pd.api.types.is_numeric_dtype(tr)

            skew_left_test = False
            skew_right_test = False
            train_skewness = float("nan")
            test_skewness = float("nan")
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
                tr_arr = tr_num.dropna().to_numpy(dtype=float)
                te_arr = te_num.dropna().to_numpy(dtype=float)
                max_n = self.max_rows_per_split_stat_tests
                if len(tr_arr) > max_n:
                    tr_arr = _sample_1d_numeric_array(tr_arr, max_n, rng)
                if len(te_arr) > max_n:
                    te_arr = _sample_1d_numeric_array(te_arr, max_n, rng)
                if len(tr_arr) < 2 or len(te_arr) < 2:
                    ks_stat, p_value = 0.0, 1.0
                else:
                    ks_stat, p_value = ks_2samp(tr_arr, te_arr)
                psi = _psi_numeric(tr_num, te_num)
                effect_size = _cohens_d(tr, te)
                statistical_drift = (p_value < self.alpha) or (psi >= self.psi_threshold)
                test_name = "KS+PSI+struct"
                drift_type = "numerical"
                integer_like = _integer_like_numeric(tr)
                is_float_col = bool(pd.api.types.is_float_dtype(tr))
                train_skewness, test_skewness, _, skew_right_shift = _numeric_skew_stats(tr, te)
                skew_left_test = _skew_shift_left_stronger_in_test(
                    tr, te, integer_like, train_skewness, test_skewness
                )
                skew_right_test = bool(
                    integer_like
                    and not np.isnan(train_skewness)
                    and not np.isnan(test_skewness)
                    and skew_right_shift
                    and test_skewness > 0.15
                )
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
                effect_size = _cramers_v(tr, te)
                statistical_drift = (p_value < self.alpha) or (psi >= self.psi_threshold)
                ks_stat = np.nan
                test_name = "Chi2+PSI+struct"
                drift_type = "categorical"
                unseen_cat, unseen_frac, n_unseen_levels = _unseen_category_stats(tr, te)

            structural = skew_left_test or skew_right_test or unseen_cat or range_expand
            drift_detected = bool(statistical_drift or structural)
            structural_high = unseen_cat or range_expand or skew_left_test or skew_right_test
            sev = _combined_severity(float(psi), structural_high and drift_detected)

            p_val = float(p_value) if pd.notna(p_value) else 1.0
            composite = _drift_score(p_val, float(psi), effect_size)

            rows.append(
                {
                    "feature": col,
                    "feature_type": drift_type,
                    "test_used": test_name,
                    "p_value": p_val,
                    "test_stat": float(ks_stat) if pd.notna(ks_stat) else np.nan,
                    "psi": float(psi),
                    "effect_size": float(effect_size),
                    "drift_score": float(composite),
                    "severity": sev,
                    "drift_detected": drift_detected,
                    "skew_left_stronger_in_test": skew_left_test,
                    "skew_right_stronger_in_test": skew_right_test,
                    "train_skewness": train_skewness,
                    "test_skewness": test_skewness,
                    "integer_like": integer_like,
                    "unseen_categories": unseen_cat,
                    "unseen_category_fraction": unseen_frac,
                    "n_unseen_levels": int(n_unseen_levels),
                    "range_expansion": range_expand,
                    "range_expansion_ratio": range_ratio,
                }
            )

        summary_df = pd.DataFrame(rows).sort_values(["drift_detected", "psi"], ascending=[False, False])
        drift_clf_auc = self._drift_classifier_auc(
            train_df,
            test_df,
            features,
            np.random.default_rng(999),
            self.max_rows_domain_classifier_per_split,
        )
        info = {
            "total_features": int(len(features)),
            "features_with_drift": int(summary_df["drift_detected"].sum()),
            "drift_percentage": float(summary_df["drift_detected"].mean() * 100 if len(summary_df) else 0.0),
            "drift_classifier_auc": float(drift_clf_auc),
        }
        return summary_df, info

    def _drift_classifier_auc(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: List[str],
        rng: np.random.Generator,
        max_rows_per_split: int,
    ) -> float:
        x_train = train_df[features].copy()
        x_test = test_df[features].copy()
        if len(x_train) > max_rows_per_split:
            idx = rng.choice(len(x_train), size=max_rows_per_split, replace=False)
            x_train = x_train.iloc[idx].reset_index(drop=True)
        if len(x_test) > max_rows_per_split:
            idx = rng.choice(len(x_test), size=max_rows_per_split, replace=False)
            x_test = x_test.iloc[idx].reset_index(drop=True)
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
        apply_binning: bool = False,
        apply_realignment: bool = False,
        apply_sliding_window: bool = False,
        apply_weighted_decay: bool = False,
        train_month: Optional[pd.Series] = None,
        test_month: Optional[pd.Series] = None,
        y_train: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], List[str], Optional[np.ndarray]]:
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

        # Strategy 4: binning/discretization for high-drift numeric features
        if apply_binning:
            for _, row in drift_table.iterrows():
                feature = str(row["feature"])
                if feature not in train_out.columns:
                    continue
                if not row["drift_detected"] or row["feature_type"] != "numerical":
                    continue
                if row["severity"] not in ("medium", "high"):
                    continue
                try:
                    med = pd.to_numeric(train_out[feature], errors="coerce").median()
                    tr_vals = pd.to_numeric(train_out[feature], errors="coerce").fillna(med).to_frame()
                    te_vals = pd.to_numeric(test_out[feature], errors="coerce").fillna(med).to_frame()
                    kbd_kw: dict = {
                        "n_bins": 10,
                        "encode": "ordinal",
                        "strategy": "quantile",
                        "subsample": None,
                    }
                    if "quantile_method" in inspect.signature(KBinsDiscretizer.__init__).parameters:
                        kbd_kw["quantile_method"] = "averaged_inverted_cdf"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        binner = KBinsDiscretizer(**kbd_kw)
                        train_out[f"{feature}__binned"] = binner.fit_transform(tr_vals).ravel()
                        test_out[f"{feature}__binned"] = binner.transform(te_vals).ravel()
                except Exception:
                    pass

        # Strategy 5: input re-alignment (center test data back to training norms)
        if apply_realignment:
            for _, row in drift_table.iterrows():
                feature = str(row["feature"])
                if feature not in train_out.columns:
                    continue
                if not row["drift_detected"] or row["feature_type"] != "numerical":
                    continue
                tr_vals = pd.to_numeric(train_out[feature], errors="coerce")
                te_vals = pd.to_numeric(test_out[feature], errors="coerce")
                train_mean = tr_vals.mean()
                train_std = tr_vals.std()
                test_mean = te_vals.mean()
                test_std = te_vals.std()
                if test_std > 1e-9 and train_std > 1e-9:
                    test_out[feature] = (te_vals - test_mean) / test_std * train_std + train_mean

        # Strategy 6: sliding window (keep only most recent N months of training data)
        sample_weights = None
        if apply_sliding_window and train_month is not None:
            month_vals = train_month.astype(str)
            unique_months = sorted(month_vals.unique())
            if len(unique_months) > 2:
                keep_months = set(unique_months[-3:])
                keep_mask = month_vals.isin(keep_months)
                if keep_mask.sum() > 500:
                    train_out = train_out.loc[keep_mask].copy()
                    if y_train is not None:
                        y_train = y_train.loc[keep_mask].copy()

        # Strategy 7: weighted decay (older data gets lower sample weight)
        if apply_weighted_decay and train_month is not None:
            month_vals = train_month.astype(str)
            unique_months = sorted(month_vals.unique())
            if len(unique_months) >= 2:
                month_to_rank = {m: i for i, m in enumerate(unique_months)}
                ranks = month_vals.map(month_to_rank).fillna(0).astype(float)
                max_rank = ranks.max()
                if max_rank > 0:
                    decay = 0.5 + 0.5 * (ranks / max_rank)
                    valid_idx = train_out.index.intersection(decay.index)
                    sample_weights = decay.loc[valid_idx].to_numpy()

        # Strategy 8: drift-based pruning (high-drift + low-importance)
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

        return train_out, test_out, mitigation_map, dropped_columns, sample_weights

