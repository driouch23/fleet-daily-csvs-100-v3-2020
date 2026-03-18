import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model_path, data_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found!")
        return

    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # --- DATA CLEANING (Robustness) ---
    initial_len = len(data)
    data.dropna(subset=['hours', 'fishing_hours', 'mmsi_present'], inplace=True)
    if len(data) < initial_len:
        print(f"Dropped {initial_len - len(data)} rows with NaNs.")

    # Recalculate ratio if missing or to ensure consistency
    if 'congestion_ratio' not in data.columns:
        print("Calculating missing 'congestion_ratio'...")
        data['congestion_ratio'] = data['fishing_hours'] / data['hours'].replace(0, np.nan)
        data['congestion_ratio'].fillna(0, inplace=True)
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Prepare features
    features = ['hours', 'fishing_hours', 'mmsi_present', 'congestion_ratio']
    # Ensure all required features are present
    missing = [f for f in features if f not in data.columns]
    if missing:
        print(f"Error: Missing columns {missing} in the input data!")
        return

    X = data[features].select_dtypes(include=[np.number])
    y_true = data['true_label'] if 'true_label' in data.columns else None
    
    if len(X) == 0:
        print("Error: No valid numeric data to evaluate.")
        return

    print(f"Making predictions on {len(X)} rows...")
    y_pred = model.predict(X)
    
    # 1. Prediction Output
    data['predicted_label'] = y_pred
    output_csv = "evaluation_results_with_predictions.csv"
    data.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # 2. Performance Metrics (Only if true_label is available)
    if y_true is not None:
        acc = accuracy_score(y_true, y_pred)
        print(f"\nOverall Model Accuracy: {acc:.4f}")
        
        print("\nDetailed Classification Report:")
        report = classification_report(y_true, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH'], zero_division=0)
        print(report)
        
        # Export results to a text file for record keeping
        with open("evaluation_report.txt", "w") as f:
            f.write("XGBoost Model Evaluation Report\n")
            f.write("==============================\n")
            f.write(f"Source Data: {data_path}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))

        # Generate Confusion Matrix Visualization
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['LOW', 'MEDIUM', 'HIGH'],
                    yticklabels=['LOW', 'MEDIUM', 'HIGH'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Congestion Level Prediction Results')
        plt.savefig('confusion_matrix.png')
        print("\nSummary: Generated evaluation_report.txt and confusion_matrix.png")
    else:
        print("\nNo 'true_label' found in dataset. Only predictions were generated.")

if __name__ == "__main__":
    evaluate_model("xgboost_congestion_model.json", "test_dataset_large.csv")
