# 🛡️ Adaptive Drift Intelligence Challenge

**Guarding Model Integrity in a Shifting Data World**

A comprehensive solution for detecting, visualizing, and mitigating data drift in machine learning models. This system automatically identifies distribution shifts between training and test datasets, provides intuitive visualizations, and implements adaptive mitigation strategies to maintain model performance.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Framework & Tools](#framework--tools)
- [Solution Workflow](#solution-workflow)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Approach Justification](#approach-justification)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

Data drift is a critical challenge in production ML systems where the distribution of incoming data differs from the training data. This solution provides:

- **Automatic Drift Detection**: Identifies feature types and applies appropriate statistical tests
- **Comprehensive Visualizations**: Clear, interactive charts showing drift extent and nature
- **Adaptive Mitigation**: Multiple strategies to handle detected drift
- **Interactive Dashboard**: User-friendly interface for end-to-end workflow

## ✨ Features

### 1. Automatic Feature Type Detection
- Automatically classifies features as numeric, categorical, or binary
- Applies appropriate statistical tests based on feature type

### 2. Multi-Test Drift Detection
- **Numeric Features**: Kolmogorov-Smirnov test, Mann-Whitney U test, PSI, Wasserstein distance
- **Categorical Features**: Chi-square test, PSI
- Configurable significance levels and thresholds

### 3. Comprehensive Visualizations
- Distribution comparisons (histograms, bar charts)
- Statistical test results visualization
- Feature-level statistics comparison
- Overall drift summary dashboard

### 4. Mitigation Strategies
- **Feature Reweighting**: Adjust feature importance based on drift severity
- **Data Augmentation**: Augment training data with test-like samples
- **Robust Scaling**: Apply robust scaling to reduce drift impact
- **Domain Adaptation**: Map training distribution to test distribution
- **Adaptive Training**: Train models with drift-aware techniques

### 5. Interactive Dashboard
- Streamlit-based web interface
- Step-by-step workflow guidance
- Real-time visualization updates
- Export capabilities

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│                      (app.py)                            │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   Drift      │ │ Visualization│ │ Mitigation  │
│  Detector    │ │   Module     │ │   Module    │
└───────┬──────┘ └──────────────┘ └──────┬──────┘
        │                                 │
┌───────▼─────────────────────────────────▼──────┐
│         Statistical Tests Module               │
│  (KS, Mann-Whitney, Chi-square, PSI, etc.)    │
└────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Input**: Training and test datasets
2. **Feature Analysis**: Automatic type detection
3. **Drift Detection**: Statistical tests execution
4. **Visualization**: Interactive charts generation
5. **Mitigation**: Strategy application
6. **Adaptive Training**: Model training with drift awareness

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd naisc
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Docker Installation

```bash
# Build the Docker image
docker build -t drift-intelligence .

# Run the container
docker run -p 8501:8501 drift-intelligence
```

## 📖 Usage

### 1. Data Upload

- Upload training and test CSV files
- Or generate sample data with intentional drift for testing

### 2. Drift Detection

- Configure detection parameters (α, PSI threshold)
- Select features to analyze
- Run drift detection

### 3. Visualizations

- View overall drift summary
- Explore feature-level distributions
- Analyze statistical test results

### 4. Mitigation

- Select mitigation strategy
- Apply to training data
- View recommendations

### 5. Adaptive Training

- Configure model parameters
- Train with drift awareness
- Evaluate performance

### 6. Results

- Review detection summary
- Export results as CSV

## 🛠️ Framework & Tools

### Core Libraries

- **NumPy** (1.24.3): Numerical computations
- **Pandas** (2.0.3): Data manipulation and analysis
- **Scikit-learn** (1.3.0): Machine learning utilities
- **SciPy** (1.11.1): Statistical tests and functions

### Statistical Testing

- **SciPy.stats**: Kolmogorov-Smirnov, Mann-Whitney U, Chi-square tests
- **Custom PSI Implementation**: Population Stability Index
- **Wasserstein Distance**: Earth Mover's Distance

### Visualization

- **Plotly** (5.15.0): Interactive visualizations
- **Matplotlib** (3.7.2): Static plotting
- **Seaborn** (0.12.2): Statistical visualizations

### Dashboard

- **Streamlit** (1.25.0): Web application framework

### Machine Learning

- **XGBoost** (1.7.6): Gradient boosting
- **LightGBM** (4.0.0): Lightweight gradient boosting
- **Scikit-learn**: Random Forest, preprocessing

### Development Tools

- **Python 3.8+**: Programming language
- **Docker**: Containerization for reproducibility
- **Git**: Version control

## 🔄 Solution Workflow

### End-to-End Process

1. **Data Ingestion**
   - Load training and test datasets
   - Validate data formats
   - Handle missing values

2. **Feature Analysis**
   - Automatic feature type detection
   - Statistical summary generation
   - Data quality assessment

3. **Drift Detection**
   - Apply appropriate statistical tests
   - Calculate drift metrics
   - Determine drift severity

4. **Visualization**
   - Generate distribution comparisons
   - Create statistical test charts
   - Build summary dashboards

5. **Mitigation Strategy Selection**
   - Analyze drift patterns
   - Recommend mitigation approaches
   - Apply selected strategies

6. **Adaptive Training**
   - Prepare data with mitigation
   - Train models with drift awareness
   - Evaluate performance

7. **Results & Reporting**
   - Generate comprehensive reports
   - Export results
   - Document findings

## 🧠 Model Architecture

### Drift Detection Models

#### Statistical Tests

1. **Kolmogorov-Smirnov Test**
   - Tests equality of continuous distributions
   - Non-parametric, distribution-free
   - Suitable for numeric features

2. **Mann-Whitney U Test**
   - Non-parametric alternative to t-test
   - Robust to outliers
   - Tests distribution shifts

3. **Chi-Square Test**
   - Tests independence of categorical variables
   - Compares frequency distributions
   - Suitable for categorical features

4. **Population Stability Index (PSI)**
   - Measures distribution shift magnitude
   - Threshold-based detection
   - Works for both numeric and categorical

5. **Wasserstein Distance**
   - Earth Mover's Distance
   - Measures distribution difference
   - Normalized for interpretability

### Mitigation Models

1. **Feature Reweighting**
   - Adjusts feature importance
   - Based on drift severity
   - Multiple weighting strategies

2. **Data Augmentation**
   - Samples from test distribution
   - Augments training data
   - Balances distributions

3. **Robust Scaling**
   - Uses median and IQR
   - Less sensitive to outliers
   - Reduces drift impact

4. **Domain Adaptation**
   - Maps training to test distribution
   - Preserves relationships
   - Maintains model validity

### Adaptive Training Models

1. **Random Forest**
   - Ensemble method
   - Feature importance weighting
   - Robust to drift

2. **XGBoost**
   - Gradient boosting
   - Sample weighting support
   - High performance

3. **LightGBM**
   - Fast gradient boosting
   - Memory efficient
   - Handles large datasets

## 📊 Performance Metrics

### Drift Detection Metrics

- **Detection Accuracy**: Percentage of correctly identified drifted features
- **False Positive Rate**: Features incorrectly flagged as drifted
- **False Negative Rate**: Drifted features missed
- **Severity Classification**: Accuracy of severity assessment

### Model Performance Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score
- **Regression**: RMSE, MAE, R²
- **Feature Importance**: Contribution to predictions

### System Performance

- **Processing Speed**: Time per feature analysis
- **Scalability**: Performance with large datasets
- **Memory Usage**: Resource efficiency

## 💡 Approach Justification

### Why Multiple Statistical Tests?

Different tests capture different aspects of drift:
- **KS Test**: Overall distribution shape
- **Mann-Whitney**: Central tendency shifts
- **PSI**: Magnitude of distribution change
- **Wasserstein**: Optimal transport distance

Using multiple tests provides comprehensive coverage and reduces false positives/negatives.

### Why Adaptive Mitigation?

- **Feature Reweighting**: Preserves all data while adjusting importance
- **Data Augmentation**: Maintains training data integrity while adapting
- **Robust Scaling**: Reduces impact of outliers and shifts
- **Domain Adaptation**: Directly addresses distribution mismatch

### Why Ensemble Training?

- Combines strengths of different algorithms
- More robust to drift
- Better generalization
- Weighted by drift severity

### Comparison to Alternatives

| Approach | Advantages | Disadvantages |
|----------|-----------|---------------|
| **Single Test** | Fast, simple | May miss drift types |
| **Multiple Tests** | Comprehensive | More computation |
| **Retraining** | Simple | Loses original model |
| **Adaptive Training** | Preserves knowledge | More complex |
| **Manual Mitigation** | Full control | Time-consuming |
| **Automated Mitigation** | Fast, consistent | Less flexibility |

Our approach balances comprehensiveness, automation, and flexibility.

## ⚠️ Limitations

### Known Limitations

1. **Large Datasets**
   - Performance may degrade with very large datasets (>1M rows)
   - Solution: Implement sampling or batch processing

2. **High-Dimensional Data**
   - Many features may slow down analysis
   - Solution: Feature selection or dimensionality reduction

3. **Complex Drift Patterns**
   - May not detect subtle, non-linear drift
   - Solution: Advanced ML-based drift detection

4. **Categorical Features**
   - Limited handling of high-cardinality categories
   - Solution: Category grouping or embedding

5. **Time Series Data**
   - Not optimized for temporal drift
   - Solution: Time-aware drift detection

6. **Missing Data**
   - Assumes missing data patterns are consistent
   - Solution: Missing data pattern analysis

### Future Improvements

- Real-time drift monitoring
- Automated model retraining triggers
- Advanced ML-based drift detection
- Time series drift detection
- Multi-variate drift detection
- Automated hyperparameter tuning for mitigation

## 📁 Project Structure

```
naisc/
├── src/
│   ├── __init__.py
│   ├── drift_detector/
│   │   ├── __init__.py
│   │   ├── detector.py          # Main drift detection class
│   │   └── statistical_tests.py # Statistical test implementations
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py            # Visualization components
│   └── mitigation/
│       ├── __init__.py
│       ├── strategies.py         # Mitigation strategies
│       └── adaptive_training.py  # Adaptive training techniques
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── README.md                     # This file
└── .gitignore                   # Git ignore rules
```

## 🤝 Contributing

This is a challenge submission. For questions or improvements:

1. Review the code structure
2. Test with your datasets
3. Provide feedback on improvements

## 📄 License

This project is developed for the Adaptive Drift Intelligence Challenge.

## 🙏 Acknowledgments

- Challenge organizers for the problem statement
- Open-source community for excellent libraries
- Statistical methods from research literature

---

**Built with ❤️ for the Adaptive Drift Intelligence Challenge**
