# 🛡️ Adaptive Drift Intelligence Challenge

**Team Name:** No Cap Analytics

**Guarding Model Integrity in a Shifting Data World**

### Official challenge source (read this spec)

This solution follows the **public organiser repository** and evaluation contract:

- **Docs & public data layout:** [NAISC-Singtel-2026](https://github.com/lowhonzheng-singtel/NAISC-Singtel-2026) (`public_data/`, `challenge_images/`, README).
- **How they run your code:** `pip install -r requirements.txt` then  
  `python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>` — end-to-end, **no manual steps**.
- **Environment:** **Python 3.12**, **CPU only**, **LightGBM 4.6.0** with fixed kwargs (`verbosity=-1`, `objective="binary"`, `is_unbalance=True`, `random_state=42`, `importance_type="gain"`). Do not change other model hyperparameters or switch model family.
- **Console:** **Data drift detection & mitigation summary** (which columns drifted, drift type/description, mitigation applied), **runtime** (in particular time for drift detection and mitigation), **AU-PRC** on train and on test **after mitigation** (when test labels exist).
- **Root deliverables:** `prediction.csv` (**exactly** `CustomerID`, `probability_score`), `model.joblib`, `report.pdf`, plus `src/`, `requirements.txt`, `.gitignore`, this `README.md` (team name at top). Submission: form [link in organiser README](https://forms.office.com/r/gwDZZQkTDG), then add collaborators **cecilia.lin@singtel.com**, **yiting.jin@singtel.com**, **honzheng.low@singtel.com**.

Local copy of train/test may live in `dataset/` for convenience; public drop is `public_data/` in the organiser repo.

---

## 📖 What This Project Does

This is a comprehensive solution for the **NAISC Singtel 2026 Challenge** that automatically detects, visualizes, and mitigates data drift in machine learning models. When your training data and test data have different distributions (data drift), this system:

1. **Detects Drift**: Automatically identifies which features have changed between training and test data
2. **Quantifies Severity**: Classifies drift as low, medium, or high severity
3. **Mitigates Drift**: Applies appropriate strategies to handle the detected drift
4. **Trains Model**: Trains a LightGBM model with fixed hyperparameters (as per challenge requirements)
5. **Generates Outputs**: Creates prediction.csv and model.joblib files

### Drift pipeline: detect, quantify, mitigate (implementation map)

This section matches what the **code** does and what a **written report** should explain.

#### 1. Detect drift — which features changed?

- **Where:** `src/utils.py` — class `DriftDetector`, method `detect`.
- **How:** For each feature (raw train vs raw test, excluding `CustomerID`, `ChurnStatus`, `Month` from the feature list used for modeling):
  - **Numeric:** two-sample **Kolmogorov–Smirnov** test (significance level `alpha`, default `0.05`) and **PSI** on binned quantiles. **Structural** checks add **sample skewness** shift (left/right) and **IQR / span expansion** for floats.
  - **Categorical:** **Chi-square** test on the train vs test category counts and **PSI** on category proportions. **Structural:** **unseen categories** in test (values never appearing in train).
- **Decision:** `drift_detected` is true if the statistical tests indicate a shift **or** any structural flag is set for that feature.
- **Global signal:** A **drift classifier** (logistic regression discriminating train rows vs test rows over all features) reports **ROC-AUC**; a high score means the two samples are easy to separate (strong overall covariate shift).
- **Outputs:** Per-feature table `drift_table.csv`; console summary (counts, drift %, classifier AUC).

#### 2. Quantify severity — low / medium / high

- **Where:** `src/utils.py` — `psi_severity`, `_combined_severity`, columns in `DriftDetector.detect` output.
- **How:** **PSI → severity bands:** `psi < 0.1` → low; `0.1 ≤ psi < 0.25` → medium; `psi ≥ 0.25` → high. If a **strong structural** issue applies (e.g. unseen levels, range expansion, marked skew shift), severity is **elevated to at least medium** so risky features are not under-labeled.
- **Outputs:** Column `severity` in `drift_table.csv`; challenge-facing descriptions in `drift_mitigation_table.csv` / console table (via `build_challenge_drift_table` in `main.py`).

#### 3. Mitigate drift — what we do about it

- **Where:** `src/utils.py` — class `DriftMitigator`, method `apply`; orchestration and ablation in `src/main.py` (`evaluate_variant`, variant loop).
- **How (in order of application in `apply`):**
  - **Drop feature:** Remove categorical columns with **unseen test categories** (stable encoding, avoids bogus levels).
  - **Feature scaling:** For drifted numerics when enabled: **log1p** (if non-negative) + **`RobustScaler`** fit on train, applied to train and test.
  - **Delta features:** Optional numeric columns encoding deviation from **train median** (stability under shift).
  - **Seasonality matching:** Optional binary feature from **`Month`**: whether the train row’s month appears in the test period (requires `train_month` / `test_month` passed from `main.py`).
  - **Pruning:** Optional drop of **high-severity** features that are also **low importance** from a baseline LightGBM (reduces harmful drifted inputs).
- **Selection:** `main.py` tries **ablation variants** (scaling / delta / seasonality / pruning on or off), scores each on **validation AU-PRC**, picks the best, then retrains on **full** mitigated training data for final test predictions.

#### For your report (short copy-paste summary)

*Detect:* We flag per-feature drift using KS + PSI for numeric data and Chi-square + PSI for categoricals, supplemented by skew, range, and unseen-level checks; a domain classifier summarizes global separability of train vs test.

*Quantify:* Severity is derived primarily from PSI bands, with a floor raise when structural drift is severe.

*Mitigate:* We combine dropping unstable categorical features, robust scaling (and optional deltas and seasonality features), optional pruning, and validation-driven selection of the mitigation bundle before final LightGBM training.

### Key Features

- ✅ **Numeric vs categorical tests**: KS + PSI (numeric); Chi-square + PSI (categorical), plus structural flags
- ✅ **Drift classifier**: Logistic regression domain score (train vs test) reported as ROC-AUC
- ✅ **Mitigation stack**: Drop unseen levels, log1p + robust scaling, delta features, seasonality bit from `Month`, importance-based pruning — ablated in `main.py`
- ✅ **Challenge Compliant**: LightGBM v4.6.0 with fixed hyperparameters (`main.py`)
- ✅ **End-to-end**: Single CLI run; artifacts written to project root
- ℹ️ **`src/drift_detector/`** (alternate statistical tests) and **`app.py`** Streamlit dashboard use additional / legacy paths; the **submission pipeline** is `main.py` + `utils.py`

---

## 🚀 Quick Start Guide

### For Your Team - Getting Started

#### Step 1: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

#### Step 2: Run the solution

**Organisers run your code from the repository root** (same folder as `requirements.txt` and `src/`) using exactly:

```bash
pip install -r requirements.txt

python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>
```

Replace `<train_data_filepath>` and `<test_data_filepath>` with the paths they supply (absolute or relative to the root). This matches the **NAISC-Singtel-2026** README interface; the script uses `argparse` with those exact flag names.

**Windows (same interface, different path separators):**

```powershell
cd path\to\naisc
pip install -r requirements.txt
python .\src\main.py --train_data_filepath .\dataset\train.csv --test_data_filepath .\dataset\test.csv
```

**Local shorthand:** If you omit the two flags, `main.py` falls back to `dataset/train.csv` and `dataset/test.csv` for quick runs only — **do not rely on that for the official grading command**, which will always pass both paths explicitly.

#### Step 3: Check outputs

**On the console** you will see the drift summary, an ASCII **Columns with Drift** table (challenge format), ablation summary, runtime, and AU-PRC.

**Files written to the current working directory** (usually the project root):

| File | What it is |
|------|------------|
| `drift_detection_summary.csv` | One-row headline: drift counts, domain AUC, chosen variant, validation AU-PRC (mirrors early console summary) |
| `drift_mitigation_table.csv` | Four-column drift table (same as printed challenge table) |
| `drift_mitigation_table.txt` | Same table as plain-text ASCII |
| `drift_table.csv` | Full drift metrics per feature (p-values, PSI, flags; **CSV only**, not printed) |
| `ablation_results.csv` | Validation AU-PRC for each mitigation variant (same as console grid) |
| `runtime_summary.csv` | Drift+mitigation timing and total runtime (same as console RUNTIME table) |
| `model_performance.csv` | First column unlabeled in console (`""` in CSV): `Train Set` / `Test Set`; `AU-PRC` to 3 decimals when labels exist (same as console MODEL PERFORMANCE table) |
| `prediction.csv` | Test predictions: `CustomerID`, `probability_score` (comma-separated) |
| `prediction.txt` | Same predictions as UTF-8 **tab-separated** text (one row per line, header row) |
| `model.joblib` | Trained LightGBM model |

#### `drift_detection_summary.csv` (one row per run)

Headline metrics written alongside the console drift summary. Columns:

| Column | Meaning |
|--------|---------|
| `total_features` | Number of modelling features compared (train vs test) |
| `features_with_drift` | Count where drift was flagged |
| `drift_percentage` | Share of features with drift (%) |
| `drift_classifier_auc` | ROC-AUC of a logistic model separating train vs test rows (global shift) |
| `selected_mitigation_variant` | Ablation branch chosen by validation AU-PRC (e.g. `baseline`, `full_policy`) |
| `validation_auprc_baseline` | Validation AU-PRC for the baseline mitigation settings |
| `validation_auprc_best` | Validation AU-PRC for the selected variant |
| `dropped_or_pruned_features` | Semicolon-separated feature names dropped (unseen categories and/or pruning); empty if none |

Example row shape:

```csv
total_features,features_with_drift,drift_percentage,drift_classifier_auc,selected_mitigation_variant,validation_auprc_baseline,validation_auprc_best,dropped_or_pruned_features
42,13,30.9524,0.917105,baseline,0.846923,0.846923,Contract
```

*(Numbers depend on your data run.)*

### Runtime budget (≈10 minutes, CPU)

Evaluation expects **drift detection + mitigation** to finish in about **10 minutes** on large hidden data (millions of rows). The pipeline therefore **caps the heaviest steps** while keeping behaviour principled:

| Mechanism | Where | What it does |
|-----------|--------|----------------|
| Subsampled **KS / PSI / Cohen *d*** | `utils.DriftDetector` | Up to **100k** points per side per feature for numeric two-sample stats (unbiased random subsample, `random_state` chain from 42). |
| Subsampled **skew / range** checks | same | Same cap for shape-only statistics. |
| Subsampled **domain classifier** | `DriftDetector._drift_classifier_auc` | At most **150k** train and **150k** test rows for the logistic “train vs test” AUC (separate RNG seed 999). |
| Full **categorical** tables | same | **Unseen categories** and **Chi² / PSI** still use **all** rows so level coverage is correct. |
| Subsampled **ablation fits** | `main.py` | If the train split exceeds **350k** rows, LightGBM fits inside the ablation loop use a **stratified** subset of that size (`random_state=45`). Validation scoring similarly capped at **120k** rows (`random_state=46`). If **all training rows** exceed **800k**, caps drop to **200k** / **80k** (`SCALABILITY_TIGHT_ROW_THRESHOLD` and `*_TIGHT` constants). |
| Subsampled **importance** for pruning | `main.py` | If the train split exceeds **300k** rows, the baseline importance model is fit on a **stratified** subset (`random_state=44`). Same **800k** tier tightens the importance cap to **150k**. |
| **Fewer ablation variants** | `main.py` | If **all training rows** ≥ **400k**, only **6** mitigation bundles are evaluated (`_ablation_variants` fast path) instead of **9**, because each variant runs full mitigation over the whole train/test frame. |
| **Tighter drift subsamples** | `main.py` + `DriftDetector` | If training rows ≥ **1.2M**, numeric stat / skew caps and the domain-classifier row cap are lowered (**80k** / **80k** / **100k** per side) before `detect`. |

The **final** model is still trained on the **full** mitigated training matrix once a variant is chosen. Tune the module-level constants in `main.py` (`IMPORTANCE_FIT_MAX_ROWS`, `ABLATION_FULL_VARIANT_ROW_THRESHOLD`, …) and `DriftDetector` fields if you need stricter caps.

---

## 📋 Challenge Requirements

### ✅ What We Must Do

1. **Fixed Hyperparameters**: Use LightGBM v4.6.0 with these exact settings:
   - `verbosity: -1`
   - `objective: "binary"`
   - `is_unbalance: True`
   - `random_state: 42`
   - `importance_type: 'gain'`

2. **Command-Line Interface**: Solution runs via:
```bash
   python ./src/main.py --train_data_filepath <train> --test_data_filepath <test>
   ```

3. **Console Output**: Must print:
   - Data Drift Detection & Mitigation Summary
   - Runtime (in seconds)
   - AU-PRC on training set
   - AU-PRC on test set after mitigation

4. **Output Files**: Must generate:
   - `prediction.csv` (CustomerID, probability_score)
   - `model.joblib` (trained model)

### ✅ What We Can Do

- Feature engineering
- Data selection (e.g., sliding windows)
- Sample weighting
- Data transformations
- Drift detection logic
- Automated retraining strategies

### ❌ What We Cannot Do

- Change model hyperparameters
- Switch to a different model family
- Use GPU acceleration

---

## 🏗️ How It Works (Architecture)

### System Overview

```
┌─────────────────────────────────────────┐
│         Main Pipeline (main.py)         │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌─────▼─────┐  ┌───▼──────┐
│ Drift  │  │Mitigation │  │ Training │
│Detection│  │ Strategies│  │ (LightGBM)│
└────────┘  └───────────┘  └──────────┘
```

### Step-by-Step Process

1. **Data Loading**: Reads training and test CSV files
2. **Feature Preparation**: 
   - Identifies CustomerID, ChurnStatus (target), and feature columns
   - Encodes categorical features automatically
   - Handles missing values
3. **Drift Detection**:
   - Detects feature types (numeric/categorical/binary)
   - Runs appropriate statistical tests for each feature
   - Calculates drift severity (low/medium/high)
4. **Mitigation**:
   - Applies robust scaling to numeric features
   - Uses domain adaptation for high-severity drift
   - Applies feature reweighting based on drift severity
5. **Model Training**:
   - Trains LightGBM with fixed hyperparameters
   - Uses sample weighting based on drift
6. **Evaluation & Output**:
   - Calculates AU-PRC metrics
   - Generates prediction.csv and model.joblib
   - Prints summary to console

### Statistical Tests Used

| Feature Type | Tests Applied |
|-------------|---------------|
| **Numeric** | Kolmogorov-Smirnov, Mann-Whitney U, PSI, Wasserstein Distance |
| **Categorical** | Chi-square, PSI |
| **Binary** | Chi-square, PSI |

### Mitigation Strategies

1. **Robust Scaling**: Uses median and IQR instead of mean/std (less sensitive to outliers)
2. **Domain Adaptation**: Maps training distribution to match test distribution
3. **Feature Reweighting**: Reduces weight of drifted features during training

---

## 📊 Expected Output Format

**⚠️ Important**: The challenge provides example images in `NAISC-Singtel-2026/challenge_images/` showing the exact expected format. You should refer to these images for visual reference:

- **`drift_output.png`**: Shows the exact format for Data Drift Detection & Mitigation Summary
- **`runtime_output.png`**: Shows the exact format for Runtime output
- **`performance_output.png`**: Shows the exact format for Model Performance Metrics
- **`prediction_example.png`**: Shows the exact format for prediction.csv file
- **`evaluation_breakdown.png`**: Shows the evaluation criteria breakdown
- **`challenge_overview.png`**: Shows the challenge overview diagram

**Always check these images** to ensure your output format matches exactly what the judges expect!

### Console Output Requirements

Your solution must print **exactly three sections** in this order:

#### 1. Data Drift Detection & Mitigation Summary

Must include:
- Columns with detected drift (list all drifted features)
- Type of drift detected (numeric, categorical, etc.)
- Mitigation methods applied (list all methods used)

**Example Format**:
```
============================================================
DATA DRIFT DETECTION & MITIGATION
============================================================

[1/3] Detecting data drift...
[2/3] Drift Detection Summary:
  - Total features analyzed: 35
  - Features with detected drift: 8
  - Drift percentage: 22.86%
  
  Columns with detected drift:
    - MonthlyCharge (numeric, severity: high)
    - TotalCharges (numeric, severity: medium)
    - InternetType (categorical, severity: low)
    ...
    
[3/3] Applying mitigation strategies...
  Mitigation methods applied: Domain Adaptation, Robust Scaling, Feature Reweighting
```

#### 2. Runtime (in seconds)

Must show:
- Time taken for drift detection and mitigation
- Total runtime (optional but recommended)

**Example Format**:
```
============================================================
RUNTIME
============================================================

Time taken for drift detection and mitigation: 12.34 seconds
Total runtime: 45.67 seconds
```

#### 3. Model Performance Metrics

Must include:
- AU-PRC on training set (6 decimal places)
- AU-PRC on test set after mitigation (6 decimal places)

**Example Format**:
```
============================================================
MODEL PERFORMANCE METRICS
============================================================

AU-PRC on training set: 0.823456
AU-PRC on test set after mitigation: 0.789123
```

**Important**: The exact format may vary, but all three sections must be clearly printed to the console.

### Output Files

**prediction.csv** (Required format):
```csv
CustomerID,probability_score
1610a102a7854c5d,0.234567
2b3c4d5e6f7g8h9i,0.789012
...
```

**Important**: 
- Must be in root directory
- Must have exactly 2 columns: `CustomerID` and `probability_score`
- CustomerID must match test set CustomerIDs
- probability_score should be between 0 and 1

**How to interpret churn risk**:
- `probability_score` close to `0.0` = **low churn risk** (customer is unlikely to churn)
- `probability_score` close to `1.0` = **high churn risk** (customer is likely to churn)
- Practical rule of thumb:
  - `< 0.30`: low risk
  - `0.30 - 0.70`: medium risk
  - `> 0.70`: high risk

**model.joblib**: Binary file containing the trained LightGBM model (saved using joblib)

**Note**: Refer to `NAISC-Singtel-2026/challenge_images/prediction_example.png` for visual reference of the exact format expected.

---

## 🛠️ Project Structure

```
.
├── src/
│   ├── main.py                    # Main entry point (CHALLENGE REQUIRED)
│   ├── utils.py                   # DriftDetector + DriftMitigator (active pipeline)
│   ├── drift_detector/            # Legacy/experimental module
│   ├── mitigation/                # Legacy/experimental module
│   └── visualization/             # Optional plotting helpers
├── app.py                         # Streamlit dashboard (OPTIONAL)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🔧 Technical Details

### Dependencies

- **Python**: 3.8+ (Challenge specifies 3.12, but 3.13 works)
- **Core**: NumPy, Pandas, Scikit-learn, SciPy
- **ML**: LightGBM 4.6.0 (exact version required)
- **Stats**: Statsmodels
- **Visualization**: Plotly, Matplotlib, Seaborn (optional)
- **Dashboard**: Streamlit (optional)

### Key Components

#### Drift + Mitigation (`src/utils.py`)

- **DriftDetector**:
  - Numerical: **K-S + PSI**
  - Categorical: **Chi-square + PSI**
  - Advanced: **Drift Classifier AUC** (domain classifier)
- **DriftMitigator**:
  - **Log/Robust scaling** on drifted numeric features
  - **Delta-based features** (`<feature>__delta_median`)
  - **Seasonality match feature** when `Month` exists
  - **Drift-based feature pruning** (high-drift + low-importance)

#### Main Pipeline (`src/main.py`)

- Handles command-line arguments
- Orchestrates entire pipeline
- Generates required outputs

---

## 🎯 For Your Team - Understanding the Code

### Main Entry Point: `src/main.py`

This is the **only file** executed for challenge runs. It:
1. Parses CLI arguments (with defaults to public data paths)
2. Loads and prepares train/test data
3. Runs drift detection and prints drift table
4. Applies mitigation strategies by feature
5. Trains standardized LightGBM
6. Writes `prediction.csv` and `model.joblib`

### Key Functions to Understand

- `load_data()`: Reads training and test CSVs
- `prepare_features()`: Handles encoding, missing values, and train/test feature alignment
- `train_lightgbm()`: Trains LightGBM with fixed challenge hyperparameters
- `print_drift_table()`: Prints "Data Drift Detection & Mitigation" output table
- `save_outputs()`: Creates `prediction.csv` and `model.joblib`

### How to Modify

If you want to improve the active solution:

1. **Drift Rules / Thresholds**: Modify `DriftDetector` in `src/utils.py`
2. **Mitigation Policy**: Modify `DriftMitigator` in `src/utils.py`
3. **Feature Engineering**: Extend `prepare_features()` in `src/main.py`
4. **Output Formatting**: Adjust `print_drift_table()` in `src/main.py`

---

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the virtual environment
pip install -r requirements.txt
```

### Issue: File not found
```bash
# Check file paths are correct
python ./src/main.py --train_data_filepath <correct_path> --test_data_filepath <correct_path>
```

### Issue: LightGBM version mismatch
```bash
# Uninstall and reinstall exact version
pip uninstall lightgbm
pip install lightgbm==4.6.0
```

### Issue: Memory errors
- The solution handles large datasets, but if you encounter issues, consider sampling

---

## 📝 Testing Your Solution

### Test with bundled data

```bash
# Uses dataset/train.csv and dataset/test.csv by default
python ./src/main.py
```

### Verify Outputs

1. Check console output has all required sections
2. Verify `prediction.csv` (and optional `prediction.txt`) exist and have correct format
3. Verify `model.joblib` exists
4. Check AU-PRC values are reasonable

---

## 🎓 Approach Justification

### Why Multiple Statistical Tests?

Different tests capture different aspects of drift:
- **KS Test**: Overall distribution shape
- **Mann-Whitney**: Central tendency shifts
- **PSI**: Magnitude of distribution change
- **Wasserstein**: Optimal transport distance

Using multiple tests provides comprehensive coverage and reduces false positives/negatives.

### Why These Mitigation Strategies?

- **Robust Scaling**: Less sensitive to outliers than standard scaling
- **Domain Adaptation**: Directly addresses distribution mismatch
- **Feature Reweighting**: Preserves all data while adjusting importance

### Why This Architecture?

- **Modular**: Easy to modify individual components
- **Extensible**: Can add new tests or strategies easily
- **Maintainable**: Clear separation of concerns

---

## ⚠️ Known Limitations

1. **Large Datasets**: May need optimization for very large datasets (>1M rows)
2. **High Dimensionality**: Many features may slow analysis
3. **Complex Patterns**: May miss subtle, non-linear drift
4. **Time Series**: Not optimized for temporal drift

---

## 📊 Dataset Information

### Public Dataset Structure

The challenge provides public train and test datasets. Key information:

**Required Columns**:
- `CustomerID` (object): Unique identifier - **MUST be preserved for prediction.csv**
- `ChurnStatus` (object): Target variable (Yes/No) - **converted to 1/0 automatically**
- `Month` (object): Time indicator - **excluded from features**

**Feature Columns** (35+ features):
- **Demographics**: UserGender, UserAge, Country, State, LocationCity, etc.
- **Service Features**: VoiceService, InternetType, Contract, Offer, etc.
- **Usage**: DataUsageAvg, TenureinMonths, NumberofReferrals, etc.
- **Financial**: MonthlyCharge, TotalCharges, TotalRevenue, CustomerLifetimeValue, etc.
- **Services**: CyberSecuritySvc, CloudStorageSvc, VideoSvc_A, VideoSvc_B, etc.

**Data Types**:
- Numeric: int64, float64 (automatically detected)
- Categorical: object (Yes/No, categories) - automatically encoded

**Important Notes**:
- Final evaluation uses a **hidden dataset** with same structure but different distributions
- The solution must handle any valid CSV with this structure
- Missing values are handled automatically

---

## 📚 Additional Resources

### Optional Dashboard

The Streamlit dashboard (`app.py`) is available for interactive exploration:

```bash
streamlit run app.py
```

This provides:
- Interactive data upload
- Real-time drift visualization
- Mitigation strategy testing
- Model performance analysis

**Note**: This is optional and not required for challenge submission, but can be helpful for understanding drift patterns.

---

## ✅ Pre-Submission Checklist

Before submitting, make sure:

- [ ] Team name is added at the top of README.md
- [ ] Solution runs end-to-end without errors
- [ ] `prediction.csv` is generated with correct format (CustomerID, probability_score)
- [ ] `model.joblib` is generated
- [ ] Console output includes all required sections
- [ ] Repository structure matches requirements
- [ ] `requirements.txt` includes all dependencies
- [ ] `.gitignore` excludes `__pycache__` and stray CSVs; `prediction.csv` and `model.joblib` can be committed for grading
- [ ] Report.pdf is created and documents your approach
- [ ] All files are pushed to main branch
- [ ] Microsoft form is submitted before deadline (12PM, 17 April 2026 SGT)
- [ ] Collaborators are added to private repo

---

## 🎯 Next Steps for Your Team

1. **Test with Public Data**: Run the solution with provided datasets
2. **Add Team Name**: Update `[YOUR_TEAM_NAME_HERE]` at the top of this README
3. **Optimize**: Fine-tune mitigation strategies based on results
4. **Generate Report**: Create `report.pdf` documenting your approach
5. **Submit**: Follow challenge submission guidelines above

---

## 📤 Submission Requirements

### Important Dates
- **Challenge Period**: 6 March 2026 – 12PM, 17 April 2026
- **Submission Deadline**: 12PM, 17 April 2026 (SGT)

### Submission Steps

1. **Push to Repository**
   - Push your final solution to your team's private repo main branch
   - Ensure all required files are included (see structure below)

2. **Submit Form**
   - **Before the deadline**, submit to: https://forms.office.com/r/gwDZZQkTDG
   - Form collects: team name, GitHub repo link, final GitHub commit hash
   - **Note**: Only the latest submission before the deadline will be considered

3. **Add Collaborators**
   - Add these emails as collaborators to your private repo:
     - cecilia.lin@singtel.com
     - yiting.jin@singtel.com
     - honzheng.low@singtel.com

### Required Repository Structure

Your repo must have this exact structure:

```
.
├── src/                    # Solution source code
│   ├── main.py             # Main function (REQUIRED)
│   ├── utils.py            # Active drift+mitigation implementation
│   ├── drift_detector/     # Optional legacy/experimental module
│   ├── mitigation/         # Optional legacy/experimental module
│   └── visualization/      # Visualization (optional)
├── dataset/                # Optional: bundled train.csv / test.csv (or use organiser public_data/)
├── notebooks/              # (Optional) Jupyter per organiser layout; see .gitkeep
├── prediction.csv          # REQUIRED after run: CustomerID, probability_score (tracked)
├── model.joblib            # REQUIRED after run: trained model (tracked)
├── report.pdf             # REQUIRED: PDF report of solution
├── .gitignore             # REQUIRED: Exclude __pycache__ and CSVs
├── requirements.txt       # REQUIRED: List of dependencies
└── README.md              # REQUIRED: Main landing page with Team Name
```

### Required Files Checklist

- ✅ `src/main.py` - Main entry point
- ✅ `prediction.csv` - Test predictions (CustomerID, probability_score)
- ✅ `model.joblib` - Trained model
- ✅ `report.pdf` - Solution documentation
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - With team name at top
- ✅ `.gitignore` - Excludes `__pycache__`, extra CSVs; **allows** `dataset/*.csv`, `prediction.csv`, and `model.joblib` per organiser layout

### Evaluation Criteria

Your submission will be evaluated on:

1. **Drift Detection Accuracy** (Hidden Data)
2. **Drift Mitigation Performance** (Hidden Test Set)
3. **Robustness & Scalability**
4. **Report Quality**
5. **Optional Dashboard** (Bonus points)
6. **Presentation** (Top 10 finalists only)

---

## 🔍 Understanding the Dataset

### Target Variable
- **ChurnStatus**: Binary classification (Yes = 1, No = 0)
- This is what we're predicting

### Important Features
- **CustomerID**: Must be preserved for prediction.csv output
- **Month**: Excluded from features (time indicator)
- **All other columns**: Used as features after encoding

### Data Preprocessing
Our solution automatically:
1. Encodes categorical features using LabelEncoder
2. Handles missing values
3. Aligns train/test to common features
4. Runs drift detection on raw features before encoded mitigation/training

---

## 💡 Tips for Your Team

### Improving Performance

1. **Feature Engineering**
   - Create interaction features
   - Derive new features from existing ones
   - Handle high-cardinality categoricals

2. **Drift Detection Tuning**
   - Adjust `alpha` (significance level) in `DriftDetector`
   - Tune `psi_threshold` for PSI test
   - Experiment with different test combinations

3. **Mitigation Strategies**
   - Try different combinations of mitigation methods
   - Adjust feature reweighting based on drift severity
   - Consider data augmentation for high-severity drift

4. **Model Training**
   - Experiment with sample weighting
   - Try different feature selection strategies
   - Consider ensemble approaches (if allowed)

### Common Pitfalls to Avoid

1. ❌ **Don't modify LightGBM hyperparameters** - Challenge requirement
2. ❌ **Don't forget to exclude Month column** - It's a time indicator
3. ❌ **Don't forget CustomerID in prediction.csv** - Required format
4. ❌ **Don't use GPU** - CPU only per challenge rules
5. ❌ **Don't miss the deadline** - 12PM, 17 April 2026 (SGT)

---

## 🆘 Getting Help

### Challenge Support
- **GitHub Discussions**: Official forum for questions
- **Help Desk**: Technical questions and troubleshooting
- **Announcements**: Official updates during challenge

### Team Resources
- Review the code comments in `src/main.py`
- Check active drift and mitigation logic in `src/utils.py`
- Use `src/drift_detector/` and `src/mitigation/` as optional reference modules
- Use the optional Streamlit dashboard for experimentation

---

## 📄 License

This project is developed for the NAISC Singtel 2026 Adaptive Drift Intelligence Challenge.

---

**Built with ❤️ for the Adaptive Drift Intelligence Challenge**

**Good luck to your team! 🚀**
