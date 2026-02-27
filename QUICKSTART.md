# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Option 1: Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t drift-intelligence .

# Run the container
docker run -p 8501:8501 drift-intelligence
```

Open your browser at `http://localhost:8501`

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 3: Using Python Script

```bash
# Run the example script
python example_usage.py
```

## 📝 Basic Workflow

1. **Upload Data**
   - Go to "📥 Data Upload" page
   - Upload training and test CSV files
   - Or click "Generate Sample Data with Drift"

2. **Detect Drift**
   - Go to "🔍 Drift Detection" page
   - Configure parameters (optional)
   - Click "🔍 Detect Drift"

3. **View Visualizations**
   - Go to "📊 Visualizations" page
   - Explore drift summary and feature-level analysis

4. **Apply Mitigation**
   - Go to "🔧 Mitigation" page
   - Select a strategy
   - Click "Apply Mitigation"

5. **Train Model**
   - Go to "🤖 Adaptive Training" page
   - Configure model settings
   - Click "🚀 Train Model"

6. **Export Results**
   - Go to "📈 Results" page
   - Download drift report

## 🎯 Example Use Cases

### Use Case 1: Quick Drift Check

```python
from src.drift_detector import DriftDetector
import pandas as pd

# Load your data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Detect drift
detector = DriftDetector()
results = detector.detect_drift(train_df, test_df)

# Check summary
print(f"Drift detected in {results['features_with_drift']} features")
```

### Use Case 2: Full Pipeline

```python
from src.drift_detector import DriftDetector
from src.mitigation import DriftMitigator, AdaptiveTrainer
import pandas as pd

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Detect drift
detector = DriftDetector()
drift_results = detector.detect_drift(train_df, test_df)

# Apply mitigation
mitigator = DriftMitigator()
train_mitigated = mitigator.apply_robust_scaling(train_df, test_df, detector.feature_types)

# Train model
trainer = AdaptiveTrainer()
results = trainer.train_with_drift_awareness(
    train_mitigated[0], train_df['target'],
    train_mitigated[1], test_df['target'],
    drift_results
)
```

## 📊 Data Format Requirements

### CSV Format
- First row should contain column names
- Missing values can be empty cells or NaN
- Numeric columns should contain numbers
- Categorical columns can contain any strings

### Example Structure
```
feature_1,feature_2,feature_3,target
1.2,5.3,A,0
2.1,6.1,B,1
...
```

## ⚙️ Configuration Options

### Drift Detection
- `alpha`: Significance level (default: 0.05)
- `psi_threshold`: PSI threshold (default: 0.2)
- `use_multiple_tests`: Use multiple tests per feature (default: True)

### Mitigation
- `reweighting_method`: 'importance', 'inverse_drift', or 'uniform'
- `augmentation_ratio`: Ratio of augmented samples (default: 0.2)

### Training
- `model_type`: 'auto', 'rf', 'xgb', or 'lgb'
- `task_type`: 'classification' or 'regression'
- `use_weights`: Use feature weights (default: True)

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Memory errors with large datasets
**Solution**: Use sampling or increase system memory

### Issue: Docker build fails
**Solution**: Check Docker is running and you have sufficient disk space

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Run `python example_usage.py` to see a complete example
- Explore the dashboard for interactive analysis

## 💡 Tips

1. Start with sample data to understand the workflow
2. Use multiple tests for comprehensive drift detection
3. Try different mitigation strategies to find what works best
4. Export results for further analysis
5. Use feature weights for better model performance

---

**Need help?** Check the documentation or review the example scripts.
