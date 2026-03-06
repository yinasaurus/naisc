# Challenge-Specific Improvements

## Overview

This document outlines the improvements made to align the project with NAISC Singtel 2026 challenge requirements.

## Key Changes Made

### 1. Main Entry Point (`src/main.py`)

**Created**: New main entry point that meets challenge requirements

**Features**:
- Command-line interface with `--train_data_filepath` and `--test_data_filepath` arguments
- End-to-end pipeline execution
- Proper console output formatting
- Generates `prediction.csv` and `model.joblib`
- Calculates AU-PRC metrics
- Tracks and reports runtime

**Console Outputs**:
1. Data Drift Detection & Mitigation Summary
   - Columns with detected drift
   - Type of drift detected
   - Mitigation methods applied
2. Runtime (in seconds)
3. Model Performance Metrics
   - AU-PRC on training set
   - AU-PRC on test set after mitigation

### 2. LightGBM Integration

**Updated**: Model training to use LightGBM v4.6.0 with fixed hyperparameters

**Fixed Hyperparameters**:
- `verbosity: -1`
- `objective: "binary"`
- `is_unbalance: True`
- `random_state: 42`
- `importance_type: 'gain'`

**Note**: No other hyperparameters are modified (as per challenge requirements)

### 3. Requirements Update

**Updated**: `requirements.txt`
- Pinned LightGBM to version 4.6.0 (challenge requirement)
- Updated other dependencies for Python 3.12 compatibility
- Removed optional packages that may cause issues

### 4. Data Handling

**Implemented**: Proper handling of challenge dataset structure
- Handles `CustomerID` column (preserved for prediction.csv)
- Handles `ChurnStatus` target (Yes/No converted to 1/0)
- Excludes `Month` column from features
- Proper feature/target separation

### 5. Output Files

**Generated**:
- `prediction.csv`: Contains `CustomerID` and `probability_score` columns
- `model.joblib`: Trained LightGBM model saved using joblib

### 6. README Update

**Updated**: README.md to match challenge requirements
- Added team name placeholder
- Updated quick start instructions
- Added challenge-specific information
- Updated project structure
- Added evaluation criteria section

### 7. .gitignore Update

**Updated**: To exclude data CSVs but allow challenge output files
- Excludes `public_data/` directory
- Allows `prediction.csv` and `model.joblib` (challenge requirements)

## Challenge Compliance Checklist

✅ **Entry Point**: `src/main.py` with command-line arguments  
✅ **Model**: LightGBM v4.6.0 with fixed hyperparameters  
✅ **Console Outputs**: Drift summary, runtime, performance metrics  
✅ **Output Files**: `prediction.csv` and `model.joblib`  
✅ **Metrics**: AU-PRC calculation for training and test sets  
✅ **Runtime**: Tracked and reported  
✅ **Python Version**: Compatible with Python 3.12  
✅ **Structure**: Matches challenge repository structure  

## Usage

### Basic Usage

```bash
python ./src/main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

### Expected Output Structure

```
Console Output:
==============
[1] Data Drift Detection & Mitigation Summary:
    Columns with detected drift: feature1, feature2
    Type of drift detected: ...
    Mitigation methods applied: ...

[2] Runtime: X.XXXX seconds

[3] Model Performance Metrics:
    AU-PRC on training set: X.XXXXXX
    AU-PRC on test set after mitigation: X.XXXXXX

Files Generated:
- prediction.csv
- model.joblib
```

## Testing

To test the solution:

1. Place training and test CSV files in appropriate location
2. Run the main script with file paths
3. Verify console outputs match expected format
4. Check that `prediction.csv` and `model.joblib` are generated
5. Verify `prediction.csv` has correct format (CustomerID, probability_score)

## Notes

- The solution runs end-to-end without manual intervention
- All drift detection and mitigation happens automatically
- Model training uses only the fixed hyperparameters
- Feature engineering and data transformations are allowed (as per challenge rules)
- Sample weighting can be applied (though not currently implemented in main.py)

## Future Enhancements

Potential improvements for better performance:

1. **Feature Engineering**: Add domain-specific feature engineering
2. **Sample Weighting**: Implement sample weighting based on drift severity
3. **Advanced Mitigation**: More sophisticated domain adaptation techniques
4. **Ensemble Methods**: Use multiple mitigation strategies and combine results
5. **Hyperparameter Tuning**: While not allowed for model, can optimize drift detection parameters

---

**Last Updated**: Based on NAISC Singtel 2026 challenge requirements
