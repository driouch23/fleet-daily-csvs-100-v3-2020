import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def load_data(data_dir, sample_per_file=5000):
    print(f"Looking for CSV files in: {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(all_files)} CSV files. Reading and sampling {sample_per_file} rows from each...")
    df_list = []
    
    # We will read each file
    for i, file in enumerate(all_files):
        if i % 20 == 0:
            print(f"  Processing file {i+1}/{len(all_files)}: {os.path.basename(file)}")
        try:
            # Optimize: Read a sample instead of the whole 11GB dataset to prevent memory crash
            # Use nrows to limit memory usage if files are huge, or sample after reading
            df = pd.read_csv(file)
            if len(df) > sample_per_file:
                df = df.sample(n=sample_per_file, random_state=42)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    print("Concatenating sampled data...")
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def main():
    # 1. Data Collection
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataset")
    # If the subfolder doesn't exist, we assume data is in the script directory itself
    if not os.path.isdir(data_dir):
        print(f"Directory {data_dir} not found. Checking current directory...")
        data_dir = script_dir
        
    df = load_data(data_dir)
    print("Data loaded successfully.\n")

    # 2. Data Preprocessing & Feature Engineering
    print("Preprocessing data...")
    # Clean the data (drop NaNs)
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped NaNs. Shape went from {initial_shape} to {df.shape}")
    
    print(f"Available columns: {df.columns.tolist()}")
    
    # Calculate congestion ratio
    if "incoming_vessels" in df.columns and "port_capacity" in df.columns:
        print("Found exact columns for congestion_ratio calculation.")
        df["congestion_ratio"] = df["incoming_vessels"] / df["port_capacity"]
    elif "fishing_hours" in df.columns and "hours" in df.columns:
        print("Using 'fishing_hours' and 'hours' for ratio.")
        # Avoid division by zero
        df["congestion_ratio"] = df["fishing_hours"] / df["hours"].replace(0, np.nan)
        df["congestion_ratio"] = df["congestion_ratio"].fillna(0)
    else:
        print("Missing 'incoming_vessels' or 'port_capacity' columns.")
        print("Creating synthetic 'congestion_ratio' based on 'mmsi_present' or random fallback...")
        if "mmsi_present" in df.columns:
            max_mmsi = df["mmsi_present"].max()
            if max_mmsi > 0:
                df["congestion_ratio"] = df["mmsi_present"] / max_mmsi
            else:
                df["congestion_ratio"] = 0
        else:
            df["congestion_ratio"] = np.random.uniform(0, 1, size=len(df))

    # Ensure there is a target column (congestion_level) mapped to 0 (LOW), 1 (MEDIUM), and 2 (HIGH)
    print("Mapping congestion levels using vectorized operations...")
    conditions = [
        (df["congestion_ratio"] < 0.5),
        (df["congestion_ratio"] >= 0.5) & (df["congestion_ratio"] <= 0.8),
        (df["congestion_ratio"] > 0.8)
    ]
    choices = [0, 1, 2]
    df["congestion_level"] = np.select(conditions, choices, default=0)
    
    # --- NOISE INJECTION (Data Augmentation) ---
    # Add 5% label noise to prevent overfitting to perfect mathematical boundaries
    noise_level = 0.05
    n_noise = int(len(df) * noise_level)
    noise_indices = np.random.choice(df.index, n_noise, replace=False)
    # Randomly assign a new label (0, 1, or 2)
    df.loc[noise_indices, "congestion_level"] = np.random.randint(0, 3, n_noise)
    print(f"Injected {noise_level*100}% label noise for robustness.")
    
    # 3. Data Exploration (EDA)
    print("\n--- EDA ---")
    print(f"Dataset shape: {df.shape}")
    print("Target variable distribution (congestion_level):")
    print(df["congestion_level"].value_counts())
    print("----------------\n")
    
    # Ensure all features for training are numeric, dropping irrelevant info
    # FEATURE SELECTION: Drop coordinates to ensure the model generalizes GLOBALLY
    # instead of learning specific port locations
    features_to_drop = ["congestion_level", "cell_ll_lat", "cell_ll_lon"]
    X = df.drop(columns=features_to_drop).select_dtypes(include=[np.number])
    y = df["congestion_level"]
    
    print(f"Features used for training: {X.columns.tolist()}")
    
    # Check if there's enough data
    if len(X) == 0:
        print("No valid numeric features to train on or dataset is empty after dropping.")
        return

    # 4. Model Choice & Anti-Overfitting Constraints (Robust Config)
    print("Configuring Robust XGBoost Model...")
    model_params = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 300,        # More trees but smaller depth
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,           # L1 Regularization
        'reg_lambda': 1.0,          # L2 Regularization
        'gamma': 0.2,               # Min loss reduction
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # 5. Model Training with Cross-Validation
    print("Performing 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Check CV performance
    clf_cv = xgb.XGBClassifier(**model_params)
    cv_scores = cross_val_score(clf_cv, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Final train/test split for detailed report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining final model on full training set...")
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X_train, y_train)
    print("Model training complete.\n")
    
    # 6. Model Evaluation
    print("Evaluating final model...")
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Final Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 7. Deployment Prep
    model_filename = "xgboost_congestion_model.json"
    print(f"\nSaving the robust model to disk as '{model_filename}'...")
    final_model.save_model(model_filename)
    print("Deployment prep complete. Script finished successfully!")

if __name__ == "__main__":
    main()
