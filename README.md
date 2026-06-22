# Fault Detection and Predictive Maintenance of Aquaculture Systems Using Supervised Algorithm

A comprehensive machine learning solution for detecting faults and predicting maintenance needs in aquaculture systems using supervised learning algorithms.

## Overview

This project implements an advanced fault detection and predictive maintenance system specifically designed for aquaculture operations. It combines sequential machine learning pipeline execution with a real-time web-based monitoring dashboard providing live fault detection and predictive maintenance capabilities.

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
├── dashboard_design/                   # Live Monitoring Dashboard Application
│   ├── main.py                        # Dashboard automation & data pipeline orchestration
│   ├── main.html                      # Interactive web dashboard UI
│   ├── data_cleaning_and_inference_model.py  # Data cleaning and ML inference engine
│   ├── final_merge.py                 # Data merging and sorting utility
│   ├── merge.py                       # CSV file merging
│   ├── models/                        # Directory for trained models
│   ├── Combined/                      # Cumulative sensor data storage
│   ├── output/                        # Processed sensor data output
│   └── sensor_data_*.json             # Real-time sensor data files
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

**Location**: `dashboard_design/`

The dashboard provides real-time monitoring and predictive maintenance capabilities through an integrated web application and data pipeline orchestration system.

### Architecture Overview

The dashboard system consists of three integrated layers:

1. **Data Pipeline Layer** (`main.py`)
   - Automated data acquisition from Datacake cloud dashboard
   - CSV export and automated download using Selenium WebDriver
   - Headless browser support for cloud/container execution
   - Duplicate file cleanup and management
   - Azure Blob Storage integration for data backup

2. **Processing Layer** (`data_cleaning_and_inference_model.py`, `final_merge.py`)
   - Data cleaning and feature engineering
   - XGBoost-based fault classification (9 classes)
   - LSTM-based time series forecasting (30-minute predictions)
   - Cascaded inference combining multiple ML models
   - Real-time prediction output in JSON format

3. **Visualization Layer** (`main.html`)
   - Interactive web-based dashboard UI
   - Real-time sensor data visualization
   - Multi-site monitoring support
   - Dynamic chart generation using Chart.js
   - Automatic data refresh every 60 seconds

### Dashboard Features

#### Real-time Monitoring
- **Live Sensor Gauges**: Current readings for Temperature, pH, Turbidity, and TDS (Total Dissolved Solids)
- **Multi-Site Support**: Switch between different aquaculture sites
- **System Health Status**: XGBoost classification results with color-coded alerts
- **Auto-Refresh**: Dashboard automatically updates sensor data every 60 seconds

#### Predictive Analytics
- **XGBoost Fault Classification**: Identifies 9 different fault types:
  - Aeration Inefficiency
  - Filter Clogging
  - Pump Degradation
  - Temperature anomalies
  - Turbidity anomalies
  - TDS anomalies
  - pH anomalies
  - Unknown/Noise
  - Normal operation

- **LSTM Forecasting**: 30-minute ahead predictions for:
  - Temperature trends
  - pH level changes
  - Turbidity projections
  - TDS concentration forecasts

#### Data Visualization
- **Historical Trends**: 8-point historical sensor data with trend analysis
- **Forecasted Values**: Dashed line projections showing predicted 30-minute future state
- **Multi-Sensor Charts**: Separate visualizations for each sensor parameter
- **Maintenance Log**: Real-time event log showing system status and alerts

### Data Flow

```
Datacake Cloud Dashboard
        ↓
   main.py (Selenium Download)
        ↓
   Merge CSV Files (merge.py)
        ↓
   Final Merge & Sort (final_merge.py)
        ↓
   Combined/merged_sensor_data.csv
        ↓
   Data Cleaning (data_cleaning_and_inference_model.py)
        ↓
   ├─→ XGBoost Inference (Current State Classification)
   │
   └─→ LSTM Inference (30-Min Forecast + Future Classification)
        ↓
   prediction.json & sensor_data_current_timestamp.json
        ↓
   main.html (Web Dashboard Display)
```

### Dashboard Components

#### Sensor Data Cards
- Display current sensor values with real-time updates
- Color-coded status indicators
- Unit labels (°C, NTU, ppm, pH scale)

#### Trend Charts
- **pH Level Trend**: Historical and forecasted pH values
- **Temperature Trend**: Thermal pattern analysis with predictions
- **Turbidity Trend**: Water clarity monitoring and forecasting
- **TDS Trend**: Dissolved solids concentration tracking

#### System Status Banner
- Large, prominent display of XGBoost classification result
- Color-coded indicators:
  - **Green (#2ecc71)**: Normal operation
  - **Yellow (#f1c40f)**: Warning state
  - **Red (#e74c3c)**: Critical alert

#### Maintenance Log
- Scrollable real-time log of system events
- LSTM prediction confidence levels
- Fault type notifications
- Dashboard refresh timestamps

### Running the Dashboard

#### Prerequisites
- Python 3.8 or higher
- Chrome browser (for Selenium WebDriver)
- Selenium and WebDriver dependencies
- TensorFlow/Keras for LSTM models
- XGBoost for classification
- Pandas for data processing

#### Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Automated Data Pipeline**
```bash
cd dashboard_design
python main.py
```

Available command-line options:
```bash
python main.py --headless          # Run without showing browser
python main.py --upload-to-blob    # Upload results to Azure Blob Storage
```

3. **Access the Dashboard**
Open your web browser and navigate to the generated HTML file:
```bash
# After main.py completes successfully
open dashboard_design/main.html
# Or use your preferred browser to open: file:///path/to/dashboard_design/main.html
```

#### Automated Scheduling

For continuous monitoring, schedule the automation with:

**Windows (Task Scheduler):**
```batch
# Run main.py every hour
schtasks /create /tn "AquacultureDashboard" /tr "python dashboard_design/main.py" /sc hourly
```

**Linux/Mac (Crontab):**
```bash
# Run every hour
0 * * * * cd /path/to/AquacultureProject/dashboard_design && python main.py
```

### Data Files Reference

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates entire data pipeline, downloads data, runs inference |
| `main.html` | Web dashboard - serves as the UI interface |
| `data_cleaning_and_inference_model.py` | Handles data cleaning, feature engineering, XGBoost & LSTM inference |
| `final_merge.py` | Merges new sensor data into cumulative dataset with time sorting |
| `merge.py` | Initial CSV merger for sensor exports |
| `models/` | Directory containing trained ML models (.joblib, .keras) |
| `Combined/` | Cumulative historical sensor data merged over time |
| `output/` | Current batch of processed sensor data |
| `sensor_data_current_timestamp.json` | Latest 8 sensor readings (feeds dashboard) |
| `sensor_data_previous_timestamp.json` | Previous 8 readings (detects new data) |
| `prediction.json` | Current XGBoost + LSTM predictions |

### Configuration

Key settings in the dashboard application:

```python
# main.py configuration
DASHBOARD_URL = "https://app.datacake.de/..."  # Datacake dashboard URL
DOWNLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
WAIT_TIME = 10  # seconds to wait for page elements

# data_cleaning_and_inference_model.py thresholds
TEMP_RANGE = (20.0, 40.0)
TDS_RANGE = (0.0, 1000)
TURBIDITY_RANGE = (0.0, 900)
PH_RANGE = (6.5, 9.5)

# main.html refresh interval
REFRESH_INTERVAL = 60  # seconds
```

### Troubleshooting Dashboard Issues

#### Dashboard Not Updating
- Check that `sensor_data_current_timestamp.json` exists and contains recent data
- Verify `main.py` is running and completing successfully
- Check browser console for JavaScript errors (F12)

#### Prediction Failures
- Ensure all trained model files exist in `models/` directory:
  - `multi_label_xgboost_model.joblib`
  - `predictive_maintenance_lstm.keras`
  - `lstm_input_scaler.joblib`
  - `lstm_target_scaler.joblib`
- Verify data cleaning steps completed without errors
- Check that input CSV contains all required sensor columns

#### Download Automation Issues
- Verify Datacake dashboard URL is accessible
- Check that Chrome/Chromium is installed and accessible
- For headless mode, ensure you're using `--headless=new` compatible Chrome version
- Check browser driver logs for Selenium errors

#### Azure Integration Issues
- Verify `AZURE_STORAGE_CONNECTION_STRING` environment variable is set
- Check Azure Blob Storage container exists
- Ensure service principal has necessary permissions

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager
- Chrome browser (for dashboard automation)

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
cd dashboard_design
python main.py
```

Open `main.html` in your web browser to access the live monitoring dashboard.

## Supervised Learning Algorithms

The project implements the following supervised and unsupervised learning approaches:

- **Isolation Forest**: Unsupervised anomaly detection for fault identification
- **DBSCAN**: Density-based clustering for fault pattern recognition
- **XGBoost**: Gradient boosting classification for fault prediction (9 classes)
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
- **Dashboard Update Latency**: <2 seconds

## Configuration

Key configuration options:

```python
# Dashboard refresh settings (main.html)
REFRESH_INTERVAL = 60  # seconds

# Alert thresholds (data_cleaning_and_inference_model.py)
TEMPERATURE_MIN = 20.0
TEMPERATURE_MAX = 40.0
TDS_MIN = 0.0
TDS_MAX = 1000
TURBIDITY_MIN = 0.0
TURBIDITY_MAX = 900
PH_MIN = 6.5
PH_MAX = 9.5

# Inference thresholds
XGBOOST_PROBABILITY_THRESHOLD = 0.5
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
# In dashboard_design/data_cleaning_and_inference_model.py
def xgboost_inference():
    """Perform XGBoost inference on the last row"""
    loaded_models = joblib.load('models/multi_label_xgboost_model.joblib')
    inference_preds = np.zeros((1, len(classes)), dtype=int)
    
    for i, model in enumerate(loaded_models):
        preds = (model.predict_proba(X_inference)[:, 1] > 0.5).astype(int)
        inference_preds[0, i] = preds[-1]
    
    return labels
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
- **Data Not Loading**: Verify sensor data files are in the correct format and location

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
- **Monitor system performance** - Track dashboard load times and prediction accuracy

## Performance Considerations

- Pipeline execution time: ~30-60 minutes (depending on data size)
- Dashboard memory usage: ~500MB-2GB
- Real-time prediction latency: <1 second per sample
- Dashboard refresh rate: 60 seconds (configurable)
- Selenium download automation: ~5-10 minutes per cycle

## Security

- Protect trained model files with appropriate access controls
- Use environment variables for sensitive configuration
- Implement authentication for dashboard access in production
- Regularly validate model predictions against ground truth
- Secure Azure Blob Storage credentials with environment variables

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
- Selenium WebDriver for Automated Testing
- Chart.js for Data Visualization

## Author

ANGB022210151

---

**Project Status**: Active Development

**Last Updated**: June 2026

**Version**: 1.0.0
