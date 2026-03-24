# 🛡️ Adaptive Drift Intelligence Challenge

**Team Name:** [YOUR_TEAM_NAME_HERE]

**Guarding Model Integrity in a Shifting Data World**

---

## 📖 What This Project Does

This is a comprehensive solution for the **NAISC Singtel 2026 Challenge** that automatically detects, visualizes, and mitigates data drift in machine learning models. When your training data and test data have different distributions (data drift), this system:

1. **Detects Drift**: Automatically identifies which features have changed between training and test data
2. **Quantifies Severity**: Classifies drift as low, medium, or high severity
3. **Mitigates Drift**: Applies appropriate strategies to handle the detected drift
4. **Trains Model**: Trains a LightGBM model with fixed hyperparameters (as per challenge requirements)
5. **Generates Outputs**: Creates prediction.csv and model.joblib files

### Key Features

- ✅ **Automatic Feature Type Detection**: Automatically identifies numeric, categorical, and binary features
- ✅ **Multiple Statistical Tests**: Uses KS test, Mann-Whitney U, Chi-square, PSI, and Wasserstein distance
- ✅ **Adaptive Mitigation**: Applies robust scaling, domain adaptation, and feature reweighting
- ✅ **Challenge Compliant**: Uses LightGBM v4.6.0 with fixed hyperparameters
- ✅ **End-to-End Pipeline**: Runs completely automated without manual intervention

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

#### Step 2: Run the Solution

```bash
# Basic usage with public datasets
python ./src/main.py \
  --train_data_filepath NAISC-Singtel-2026/public_data/train.csv \
  --test_data_filepath NAISC-Singtel-2026/public_data/test.csv
```

#### Step 3: Check Outputs

After running, you'll get:
- **Console output**: Drift detection summary, runtime, and AU-PRC metrics
- **prediction.csv**: Test predictions with CustomerID and probability_score
- **model.joblib**: Trained LightGBM model

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

### Test with Public Data

```bash
# Make sure you have the public datasets
python ./src/main.py \
  --train_data_filepath NAISC-Singtel-2026/public_data/train.csv \
  --test_data_filepath NAISC-Singtel-2026/public_data/test.csv
```

### Verify Outputs

1. Check console output has all required sections
2. Verify `prediction.csv` exists and has correct format
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
- [ ] `.gitignore` excludes __pycache__ and CSVs (except prediction.csv)
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
├── notebooks/              # (Optional) Jupyter notebooks
│   └── xxx.ipynb
├── prediction.csv         # REQUIRED: Predictions on public test set
├── model.joblib           # REQUIRED: Trained model on public train set
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
- ✅ `.gitignore` - Excludes __pycache__ and CSVs (except prediction.csv)

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
