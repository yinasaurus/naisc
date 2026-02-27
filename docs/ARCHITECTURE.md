# Architecture Documentation

## System Architecture

### Overview

The Adaptive Drift Intelligence system is designed with a modular architecture that separates concerns into distinct components:

1. **Drift Detection Module**: Core statistical analysis
2. **Visualization Module**: Interactive chart generation
3. **Mitigation Module**: Strategy implementation
4. **Dashboard Interface**: User interaction layer

### Component Details

#### 1. Drift Detection Module (`src/drift_detector/`)

**Purpose**: Automatically detect and quantify data drift

**Key Classes**:
- `DriftDetector`: Main orchestrator for drift detection
- `StatisticalTests`: Collection of statistical test implementations

**Features**:
- Automatic feature type detection
- Multi-test approach for comprehensive analysis
- Severity classification (low/medium/high)
- Configurable thresholds

**Statistical Tests**:
- Kolmogorov-Smirnov (numeric)
- Mann-Whitney U (numeric)
- Chi-Square (categorical)
- PSI (all types)
- Wasserstein Distance (all types)

#### 2. Visualization Module (`src/visualization/`)

**Purpose**: Generate clear, insightful visualizations

**Key Classes**:
- `DriftVisualizer`: Creates all visualization types

**Visualization Types**:
- Distribution comparisons
- Statistical test results
- Feature statistics comparison
- Overall drift summary
- Comprehensive dashboards

**Technologies**:
- Plotly for interactivity
- Matplotlib/Seaborn for static charts

#### 3. Mitigation Module (`src/mitigation/`)

**Purpose**: Implement strategies to handle detected drift

**Key Classes**:
- `DriftMitigator`: Strategy implementations
- `AdaptiveTrainer`: Drift-aware model training

**Mitigation Strategies**:
1. **Feature Reweighting**: Adjust importance based on drift
2. **Data Augmentation**: Add test-like samples
3. **Robust Scaling**: Reduce drift impact
4. **Domain Adaptation**: Map distributions
5. **Adaptive Training**: Train with drift awareness

#### 4. Dashboard Interface (`app.py`)

**Purpose**: User-friendly web interface

**Features**:
- Step-by-step workflow
- Real-time visualization
- Interactive configuration
- Results export

**Pages**:
1. Data Upload
2. Drift Detection
3. Visualizations
4. Mitigation
5. Adaptive Training
6. Results

### Data Flow

```
User Input (CSV files)
    ↓
Data Validation & Loading
    ↓
Feature Type Detection
    ↓
Statistical Test Selection
    ↓
Drift Detection Execution
    ↓
Results Aggregation
    ↓
Visualization Generation
    ↓
Mitigation Strategy Application
    ↓
Adaptive Model Training
    ↓
Results & Export
```

### Design Patterns

1. **Strategy Pattern**: Different mitigation strategies
2. **Factory Pattern**: Model creation
3. **Observer Pattern**: Results updates
4. **Template Method**: Test execution flow

### Scalability Considerations

- Modular design allows component replacement
- Stateless operations enable parallelization
- Efficient data structures (pandas, numpy)
- Configurable batch processing

### Extensibility

- Easy to add new statistical tests
- Pluggable mitigation strategies
- Customizable visualization components
- Flexible model integration
