# data cleaning and inference models
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def data_cleaning(input_filename):
    """
    Load and clean the sensor data, perform feature engineering, and save the last 8 rows.
    """
    df = pd.read_csv(input_filename)
    print(f"Loaded {len(df)} rows for inference.")

    print("Applying fault detection logic...")
    temp_min, temp_max = 20.0, 40.0
    tds_min, tds_max = 0.0, 1000
    turbidity_min, turbidity_max = 0.0, 900
    ph_min, ph_max = 6.5, 9.5

    # 1-4. Fault Flags (Cast to integer to match training data)
    df['temp_fault'] = ((df['Temperature (°C)'] < temp_min) | (df['Temperature (°C)'] > temp_max) | (df['Temperature (°C)'].isna())).astype(int)
    df['tds_fault'] = ((df['Time Dissolve Solid (ppm)'] < tds_min) | (df['Time Dissolve Solid (ppm)'] > tds_max) | (df['Time Dissolve Solid (ppm)'] == 0) | (df['Time Dissolve Solid (ppm)'].isna())).astype(int)
    df['turbidity_fault'] = ((df['Turbidity (NTU)'] < turbidity_min) | (df['Turbidity (NTU)'] > turbidity_max) | (df['Turbidity (NTU)'].isna())).astype(int)
    df['ph_fault'] = ((df['pH'] < ph_min) | (df['pH'] > ph_max) | (df['pH'].isna())).astype(int)

    # 5. Time preprocessing (FIXED: Handling the long string and converting to Malaysia Time)
    # Remove the literal " (Coordinated Universal Time)" or similar text in parentheses at the end
    df['time'] = df['time'].str.replace(r' \(.*\)$', '', regex=True)

    # Parse the datetime. The %z handles the +0000 part, making it timezone-aware (UTC)
    df['time'] = pd.to_datetime(df['time'], format="%a %b %d %Y %H:%M:%S GMT%z", errors='coerce')

    # Convert the timezone to Malaysia Time (Asia/Kuala_Lumpur)
    df['time'] = df['time'].dt.tz_convert('Asia/Kuala_Lumpur')

    # Sort and get time difference
    df = df.sort_values(by='time').reset_index(drop=True)
    df['time_diff_seconds'] = df['time'].diff().dt.total_seconds()

    # 6. Cleaning sensor values (Forward Fill)
    df['temperature_cleaned'] = df['Temperature (°C)'].replace(-999, np.nan).ffill()
    df['tds_cleaned'] = df['Time Dissolve Solid (ppm)'].replace(0, np.nan).ffill()

    df['pH_cleaned'] = df['pH'].copy()
    df.loc[(df['pH_cleaned'] <= 0) | (df['pH_cleaned'] > 14), 'pH_cleaned'] = np.nan
    df['pH_cleaned'] = df['pH_cleaned'].ffill()

    df['turbidity_cleaned'] = df['Turbidity (NTU)'].replace(999, np.nan).ffill()

    # 7-10. Feature Engineering
    df['Rate of Change (ΔT/Δt)'] = df['temperature_cleaned'].diff() / df['time_diff_seconds']

    window_size = 5
    df['Rolling Variance (σ²ₚₕ)'] = df['pH_cleaned'].rolling(window=window_size).var()
    df['Short-Term Gradient (ΔNTU)'] = df['turbidity_cleaned'].diff()

    df['tds_rolling_mean'] = df['tds_cleaned'].rolling(window=window_size).mean()
    df['Moving Average Deviation'] = df['tds_cleaned'] - df['tds_rolling_mean']

    # Save the last 8 rows for LSTM inference (overwrites if exists, creates if not)
    last_8_rows = df.tail(8)
    last_8_rows.to_csv('models/last_8_rows.csv', index=False)

    return df

def xgboost_inference():
    """
    Perform XGBoost inference on the last row of the last 8 rows CSV and return predicted faults.
    """
    # Load the last 8 rows
    df_last_8 = pd.read_csv('models/last_8_rows.csv')

    # Take the last row
    last_row = df_last_8.tail(1)

    # Select the exact 12 features expected by the model
    feature_columns = [
        'temperature_cleaned', 'tds_cleaned', 'turbidity_cleaned', 'pH_cleaned',
        'Rate of Change (ΔT/Δt)', 'Moving Average Deviation', 'Short-Term Gradient (ΔNTU)',
        'Rolling Variance (σ²ₚₕ)', 'temp_fault', 'tds_fault', 'turbidity_fault', 'ph_fault'
    ]

    X_inference = last_row[feature_columns].copy()

    # Handle NaNs resulting from rolling windows and diffs (fill with column mean, fallback to 0)
    X_inference = X_inference.fillna(X_inference.mean()).fillna(0)

    print("Loading the trained XGBoost model...")
    # This will load the list of 9 separate XGBoost classifiers
    loaded_models = joblib.load('models/multi_label_xgboost_model.joblib')

    # The classes corresponding to your models
    classes = ['Aeration Inefficiency', 'Filter Clogging', 'Normal', 'Pump Degradation',
               'TDS', 'Temperature', 'Turbidity', 'Unknown/Noise', 'pH']

    print("Running XGBoost inference on the last row...")
    # Array to hold the binary predictions (0 or 1) for each class
    inference_preds = np.zeros((1, len(classes)), dtype=int)

    for i, model in enumerate(loaded_models):
        # Get prediction for the last row only
        preds = (model.predict_proba(X_inference)[:, 1] > 0.5).astype(int)
        inference_preds[0, i] = preds[-1]

    # Decode binary predictions back to text labels
    row = inference_preds[0]
    labels = [classes[idx] for idx, val in enumerate(row) if val == 1]
    if not labels:
        labels = ['Normal']  # Failsafe if no faults are triggered

    print(f"Predicted faults for the last row: {labels}")
    return labels

def lstm_inference():
    """
    Perform LSTM inference using the last 8 rows.
    """
    # Load the last 8 rows
    df_sample = pd.read_csv('models/last_8_rows.csv')

    # Load models and scalers
    lstm_model = load_model('models/predictive_maintenance_lstm.keras', compile=False)
    xgb_model = joblib.load('models/multi_label_xgboost_model.joblib')
    X_scaler = joblib.load('models/lstm_input_scaler.joblib')
    y_scaler = joblib.load('models/lstm_target_scaler.joblib')

    # Define feature groups
    lstm_features = [
        'temperature_cleaned', 'tds_cleaned', 'turbidity_cleaned', 'pH_cleaned',
        'Rate of Change (ΔT/Δt)', 'Moving Average Deviation',
        'Short-Term Gradient (ΔNTU)', 'Rolling Variance (σ²ₚₕ)'
    ]

    # Classes for decoding
    classes = ['Aeration Inefficiency', 'Filter Clogging', 'Normal', 'Pump Degradation',
               'TDS', 'Temperature', 'Turbidity', 'Unknown/Noise', 'pH']

    def run_cascaded_inference(recent_df):
        """
        recent_df: A DataFrame with the last 8 rows of data.
        """
        # --- STEP A: LSTM PREDICTION (The 30-Min Forecast) ---
        lstm_data = recent_df[lstm_features].tail(8)
        scaled_lstm_input = X_scaler.transform(lstm_data)
        lstm_input_reshaped = scaled_lstm_input.reshape(1, 8, 8)

        # Predict future sensor values
        forecasted_scaled = lstm_model.predict(lstm_input_reshaped)
        forecasted_raw = y_scaler.inverse_transform(forecasted_scaled)[0]

        # --- STEP B: XGBOOST PREDICTION (The Future Classification) ---
        # 1. Prepare the 12-feature vector for the XGBoost models
        f_temp, f_tds, f_turb, f_ph = forecasted_raw
        latest_trends = recent_df[lstm_features[4:8]].iloc[-1].values
        current_fault_status = recent_df[['temp_fault', 'tds_fault', 'turbidity_fault', 'ph_fault']].iloc[-1].values

        xgb_input_vector = np.concatenate([[f_temp, f_tds, f_turb, f_ph], latest_trends, current_fault_status])
        xgb_input_final = xgb_input_vector.reshape(1, -1)

        # 2. Run XGBoost prediction
        future_binary_preds = np.zeros(len(classes), dtype=int)

        for i, model in enumerate(xgb_model):
            # Predict probability and apply threshold
            prob = model.predict_proba(xgb_input_final)[0, 1]
            future_binary_preds[i] = (prob > 0.5).astype(int)

        # 3. Decode the labels
        future_labels = [classes[idx] for idx, val in enumerate(future_binary_preds) if val == 1]
        if not future_labels:
            future_labels = ['Normal']

        # --- Print Results ---
        print(f"\n--- 30-MINUTE PREDICTIVE MAINTENANCE FORECAST ---")
        print(f"Forecasted State: Temp={f_temp:.2f}, TDS={f_tds:.2f}, Turbidity={f_turb:.2f}, pH={f_ph:.2f}")
        print(f"Predicted Faults in 30 mins: {future_labels}")

        return forecasted_raw, future_labels

    # Run the inference
    forecast, predicted_fault_list = run_cascaded_inference(df_sample.tail(8))
    return forecast, predicted_fault_list

if __name__ == "__main__":
    # Task 2: Use output/merged_sensor_data.csv as input
    df = data_cleaning('output/merged_sensor_data.csv')
    predicted_faults = xgboost_inference()
    lstm_inference()
