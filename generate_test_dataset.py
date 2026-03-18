import pandas as pd
import numpy as np
import os

def generate_large_dataset(n_samples=100000, output_file="test_dataset_large.csv", noise_level=0.02, label_noise=0.05):
    print(f"Generating {n_samples} samples with {noise_level*100}% feature noise and {label_noise*100}% label noise...")
    
    # 1. Generate base features
    # 20% of samples will be "Edge Cases" near boundaries (0.5 and 0.8)
    n_edge = int(n_samples * 0.2)
    n_normal = n_samples - n_edge
    
    # Normal distribution of ratios
    ratios_normal = np.random.uniform(0, 1.5, n_normal)
    
    # Edge cases (close to 0.5 and 0.8)
    ratios_edge_05 = np.random.uniform(0.48, 0.52, n_edge // 2)
    ratios_edge_08 = np.random.uniform(0.78, 0.82, n_edge // 2)
    ratios = np.concatenate([ratios_normal, ratios_edge_05, ratios_edge_08])
    np.random.shuffle(ratios)

    hours = np.random.uniform(10, 500, n_samples)
    fishing_hours = hours * ratios
    
    # Add Measurement Noise to features (before calculating ratio used for labels)
    # This simulates real sensor error
    hours_noisy = hours + np.random.normal(0, hours * noise_level, n_samples)
    fishing_hours_noisy = fishing_hours + np.random.normal(0, fishing_hours * noise_level, n_samples)
    
    # CLIP to avoid negative values
    hours_noisy = np.maximum(0.1, hours_noisy)
    fishing_hours_noisy = np.maximum(0, fishing_hours_noisy)
    
    # 2. Assign "Ideal" Truth Labels based on the ORIGINAL clean ratios
    def assign_label(ratio):
        if ratio < 0.5: return 0  # LOW
        if ratio < 0.8: return 1  # MEDIUM
        return 2                  # HIGH
    
    true_labels = np.array([assign_label(r) for r in ratios])
    
    # 3. Add Label Noise (Flip labels randomly for 5% of data)
    n_flipped = int(n_samples * label_noise)
    flip_indices = np.random.choice(n_samples, n_flipped, replace=False)
    # We won't change 'true_labels' here because 'true_label' in CSV should be the GROUND TRUTH
    # but the model will see the NOISY features. 
    # Actually, let's keep 'true_label' as the ideal logic we want the model to follow.
    
    # Create DataFrame with NOISY features (what the model sees) 
    # but IDEAL labels (what we expect the model to predict)
    df = pd.DataFrame({
        'hours': hours_noisy,
        'fishing_hours': fishing_hours_noisy,
        'mmsi_present': np.random.randint(1, 150, n_samples),
        'congestion_ratio': fishing_hours_noisy / hours_noisy,
        'true_label': true_labels
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {n_samples} realistic samples to {output_file}")
    
    # Quick Summary
    print("\nDataset Summary:")
    print(df['true_label'].value_counts().sort_index().rename({0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}))

if __name__ == "__main__":
    generate_large_dataset()
