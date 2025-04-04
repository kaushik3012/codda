import argparse
import numpy as np
from .compress.utils import read_vti_file
import os
import csv

# Calculate SNR and NRMSE between the original and reconstructed data
def calculate_metrics(original_data, reconstructed_data):
    snr = 10 * np.log10(np.mean(original_data**2) / np.mean((original_data - reconstructed_data)**2))
    nrmse = np.sqrt(np.mean((original_data - reconstructed_data)**2)) / (np.max(original_data) - np.min(original_data))
    return snr, nrmse

if __name__ == "__main__":
    # Load the original data and the subsampled data
    parser = argparse.ArgumentParser(description="Script to generate metrics for reconstructed data from Copula Models.")
    parser.add_argument("--og", type=str, help="Path to the original data file", required=True)
    parser.add_argument("--gen", type=str, help="Path to the reconstructed data file", required=True)
    parser.add_argument("--model", type=str, help="Path to the copula model file", required=True)
    args = parser.parse_args()

    file_paths = {
        "original_data": args.og,
        "subsampled_data": args.gen,
        "model": args.model
    }

    arr, _,_,_ = read_vti_file(file_paths["original_data"])
    rec, _,_,_ = read_vti_file(file_paths["subsampled_data"])

    # Calculate SNR and NRMSE
    snr, nrmse = calculate_metrics(arr, rec)

    # Compare file sizes 
    original_size = os.path.getsize(file_paths["original_data"])
    reconstructed_size = os.path.getsize(file_paths["subsampled_data"])
    model_size = os.path.getsize(file_paths["model"])

    # Display in the form of table
    print("\nMetrics Summary:")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'SNR (dB)':<20} {snr:.2f}")
    print(f"{'NRMSE':<20} {nrmse:.4f}")
    print(f"{'Original Size (MB)':<20} {original_size / (1024 * 1024):.2f}")
    print(f"{'Reconstructed Size (MB)':<20} {reconstructed_size / (1024 * 1024):.2f}")
    print(f"{'Model Size (MB)':<20} {model_size / (1024 * 1024):.2f}")
    print(f"{'Compression Ratio':<20} {original_size / model_size:.2f}")
    print("-" * 30)
    print("Metrics calculated successfully.")
    
    # Prepare metrics data dictionary
    metrics = {
        "SNR (dB)": round(snr, 2),
        "NRMSE": round(nrmse, 4),
        "Original Size (MB)": round(original_size / (1024 * 1024), 2),
        "Reconstructed Size (MB)": round(reconstructed_size / (1024 * 1024), 2),
        "Model Size (MB)": round(model_size / (1024 * 1024), 2),
        "Compression Ratio": round(original_size / model_size, 2)
    }
    
    # Save metrics to CSV file
    with open("metrics.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    
    print("Metrics saved to metrics.csv")
