# Fault Detection and Predictive Maintenance of Aquaculture Systems Using Supervised Algorithm

A comprehensive machine learning solution for detecting faults and predicting maintenance needs in aquaculture systems using supervised learning algorithms.

## Overview

This project implements an advanced fault detection and predictive maintenance system specifically designed for aquaculture operations. It combines sequential machine learning pipeline execution with real-time monitoring capabilities.

## Key Components

The project is structured into two main components:

1. **Machine Learning Pipeline** - Sequential Jupyter Notebook execution (6 steps)
2. **Live Monitoring Dashboard** - Real-time fault detection and predictive maintenance interface

## Technology Stack

- **Jupyter Notebook** (94%) - Interactive machine learning pipeline and analysis
- **Python** (4.1%) - Core algorithms and data processing
- **HTML** (1.9%) - Dashboard interface and visualization

## Project Structure

```
AquacultureProject/
├── notebooks/                                  # Machine Learning Pipeline
│   ├── Sensor_Fault_Labeling_Step1.ipynb     # Step 1: Sensor Fault Labeling
│   ├── feature_engineering_Step2.ipynb       # Step 2: Feature Engineering
│   ├── isolation_forest_Step3.ipynb          # Step 3: Isolation Forest Detection
│   ├── dbscan_step4.ipynb                    # Step 4: DBSCAN Clustering Analysis
│   ├── XGBoost_step5.ipynb                   # Step 5: XGBoost Classification
│   └── step_6_lstmi.ipynb                    # Step 6: LSTM Time Series Analysis
│
├── dashboard/                          # Live Monitoring Dashboard
│   ├── app.py                         # Dashboard application
│   ├── templates/                     # HTML templates
│   ├── static/                        # CSS, JavaScript, assets
│   └── config.py                      # Configuration settings
│
├── models/                             # Trained Models
│   ├── fault_detection_model.pkl      # Serialized model
│   └── scaler.pkl                     # Feature scaler
│
├── data/                               # Data Directory
│   ├── raw/                           # Raw sensor data
│   └── processed/                     # Processed data
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Machine Learning Pipeline (6 Steps)

The pipeline must be executed **sequentially** to ensure proper data processing and model development:

### Step 1: Sensor Fault Labeling
**File**: `notebooks/Sensor_Fault_Labeling_Step1.ipynb`

- Load sensor data from aquaculture systems
- Label faulty and normal operating conditions
- Data cleaning and preprocessing
- Handle missing values and outliers
- Data validation and quality checks

### Step 2: Feature Engineering
**File**: `notebooks/feature_engineering_Step2.ipynb`

- Create new features from raw sensor data
- Feature scaling and normalization
- Statistical feature extraction
- Feature importance analysis
- Select optimal features for model training

### Step 3: Isolation Forest Detection
**File**: `notebooks/isolation_forest_Step3.ipynb`

- Anomaly detection using Isolation Forest algorithm
- Identify outliers and unusual patterns
- Generate anomaly scores
- Validation of detected anomalies

### Step 4: DBSCAN Clustering Analysis
**File**: `notebooks/dbscan_step4.ipynb`

- Density-Based Spatial Clustering (DBSCAN) analysis
- Cluster fault patterns and system states
- Identify fault signatures and clusters
- Cluster validation and interpretation

### Step 5: XGBoost Classification
**File**: `notebooks/XGBoost_step5.ipynb`

- Train XGBoost classification model
- Implement cross-validation strategies
- Hyperparameter tuning
- Model comparison and performance evaluation
- Feature importance from XGBoost

### Step 6: LSTM Time Series Analysis
**File**: `notebooks/step_6_lstmi.ipynb`

- Long Short-Term Memory (LSTM) model for time series
- Sequential pattern detection
- Predictive maintenance timing using temporal data
- Model serialization and export

## Live Monitoring Dashboard

**Location**: `dashboard/`

The dashboard provides real-time monitoring and predictive maintenance capabilities:

### Features

- **Real-time Fault Detection**: Continuous monitoring of aquaculture system health
- **Predictive Alerts**: Proactive maintenance warnings before system failures
- **System Metrics**: Live sensor data visualization
- **Fault History**: Historical fault logs and patterns
- **Maintenance Schedule**: Recommended maintenance timing and actions
- **System Status**: Overall health indicators and risk assessment

### Dashboard Access

```bash
python dashboard/app.py
```

The dashboard will be available at `http://localhost:5000`

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### 1. Clone Repository

```bash
git clone https://github.com/ANGB022210151/AquacultureProject.git
cd AquacultureProject
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Running the ML Pipeline

Execute the notebooks **in order**:

```bash
# Open Jupyter
jupyter notebook

# Then run in sequence:
# 1. Sensor_Fault_Labeling_Step1.ipynb
# 2. feature_engineering_Step2.ipynb
# 3. isolation_forest_Step3.ipynb
# 4. dbscan_step4.ipynb
# 5. XGBoost_step5.ipynb
# 6. step_6_lstmi.ipynb
```

**Important**: Each step depends on outputs from the previous step. Do not skip or reorder steps.

### 4. Running the Live Dashboard

After completing the ML pipeline:

```bash
cd dashboard
python app.py
```

Navigate to `http://localhost:5000` in your web browser to access the live monitoring dashboard.

## Supervised Learning Algorithms

The project implements the following supervised and unsupervised learning approaches:

- **Isolation Forest**: Unsupervised anomaly detection for fault identification
- **DBSCAN**: Density-based clustering for fault pattern recognition
- **XGBoost**: Gradient boosting classification for fault prediction
- **LSTM**: Recurrent neural networks for temporal sequence analysis and predictive maintenance

## Data Requirements

The system requires sensor data from aquaculture operations including:

- Water quality parameters (pH, dissolved oxygen, temperature)
- System performance metrics (pump status, flow rates)
- Equipment operational data (power consumption, cycles)
- Historical fault and maintenance records

## Model Performance

Expected performance metrics:

- **Fault Detection Accuracy**: Target >95%
- **Maintenance Prediction Recall**: Target >90%
- **False Positive Rate**: <5%
- **Response Time**: Real-time alerts (<1 second)

## Configuration

Key configuration options in `dashboard/config.py`:

```python
# Model settings
MODEL_PATH = 'models/fault_detection_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Dashboard settings
REFRESH_INTERVAL = 5  # seconds
ALERT_THRESHOLD = 0.7
```

## Usage Examples

### Pipeline Execution Example

```python
# In notebook XGBoost_step5.ipynb
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)
```

### Dashboard Integration Example

```python
# In dashboard/app.py
@app.route('/predict', methods=['POST'])
def predict():
    sensor_data = request.json
    prediction = model.predict([sensor_data])
    return jsonify({'fault_detected': prediction[0]})
```

## Troubleshooting

### Pipeline Issues

- **Missing Data**: Check `Sensor_Fault_Labeling_Step1.ipynb` for data source configuration
- **Feature Errors**: Ensure `feature_engineering_Step2.ipynb` completed successfully
- **Model Training Failures**: Verify data quality in step 1 and feature selection in step 2

### Dashboard Issues

- **Connection Errors**: Ensure the trained model exists in `models/` directory
- **Prediction Failures**: Re-run the full pipeline to regenerate model files
- **Performance Issues**: Check system resources and reduce data refresh interval

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch for new models or features
2. Document changes in relevant notebooks
3. Ensure pipeline executes sequentially without errors
4. Update dashboard if adding new visualizations
5. Submit a pull request with detailed description

## Best Practices

- **Always run notebooks sequentially** - Do not skip steps
- **Version your data** - Keep track of raw data versions
- **Document parameters** - Note all hyperparameter changes
- **Validate predictions** - Compare dashboard predictions with actual outcomes
- **Regular retraining** - Periodically retrain with new data

## Performance Considerations

- Pipeline execution time: ~30-60 minutes (depending on data size)
- Dashboard memory usage: ~500MB-2GB
- Real-time prediction latency: <1 second per sample
- Dashboard refresh rate: 5 seconds (configurable)

## Security

- Protect trained model files with appropriate access controls
- Use environment variables for sensitive configuration
- Implement authentication for dashboard access in production
- Regularly validate model predictions against ground truth

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Support

For issues, questions, or suggestions:

1. Check existing documentation in notebooks
2. Review troubleshooting section
3. Open an issue in the repository
4. Contact the development team

## References

- Aquaculture System Monitoring Best Practices
- Supervised Learning for Predictive Maintenance
- Feature Engineering for Time Series Data
- Real-time Monitoring System Design
- XGBoost Documentation
- LSTM and Recurrent Neural Networks for Time Series

## Author

ANGB022210151

---

**Project Status**: Active Development

**Last Updated**: June 2026

**Version**: 1.0.0
