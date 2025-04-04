import os
import re
import subprocess
import csv

#!/usr/bin/env python
"""
Run the metrics.py script on all files in the reconstructed_fields directory by matching
the original, generated, and model file naming conventions, and stores the output of each
run on disk in the metrics_output directory.
"""

# Directories
MODEL = "GaussianMultivariate"
RECONS_DIR = os.path.join("reconstructed_fields", MODEL)
SUBSAMPLED_DIR = "subsampled_field"
MODELS_DIR = os.path.join("copula_models", MODEL)
OUTPUT_DIR = "metrics_output"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Regex to match the reconstructed file naming:
# e.g., reconstructed_field_P_(100, 100, 20).vti
re_filename = re.compile(
    r"reconstructed_field_(?P<field>[A-Za-z]+)_\((?P<dims>[\d,\s]+)\)\.vti"
)

def main():

    FINAL_CSV = f"final_metrics_{MODEL}.csv"
    # Check if the final CSV file exists
    if os.path.exists(FINAL_CSV):
        # Remove the existing final CSV file
        os.remove(FINAL_CSV)
        print(f"Removed existing final CSV file {FINAL_CSV}.")

    # Loop over distribution directories in reconstructed_fields
    for distribution in os.listdir(RECONS_DIR):
        dist_path = os.path.join(RECONS_DIR, distribution)
        if not os.path.isdir(dist_path):
            continue

        # Each distribution directory has resolution subdirectories (e.g., "10")
        for resolution in os.listdir(dist_path):
            res_path = os.path.join(dist_path, resolution)
            if not os.path.isdir(res_path):
                continue

            # Process every .vti file in the resolution directory
            for filename in os.listdir(res_path):
                if not filename.endswith('.vti'):
                    continue

                match = re_filename.match(filename)
                if not match:
                    print(f"Skipping file with unexpected name: {filename}")
                    continue

                field = match.group("field")  # e.g., "P"
                dims = match.group("dims")    # e.g., "100, 100, 20"
                
                # Build file paths based on naming conventions
                gen_file = os.path.join(res_path, filename)
                # Original file uses pattern: {field}f25_(dims).vti in subsampled_field
                og_file = os.path.join(SUBSAMPLED_DIR, f"{field}f25_({dims}).vti")
                # Model file follows: model_P_TC_Velocity_CLOUD_PRECIP_QCLOUD_QGRAUP_QICE_QRAIN_QSNOW_QVAPOR_U_V_W_{distribution}_{resolution}.bin
                model_filename = (
                    f"model_P_TC_Velocity_CLOUD_PRECIP_QCLOUD_QGRAUP_QICE_QRAIN_QSNOW_QVAPOR_U_V_W_"
                    f"{distribution}_{resolution}.bin"
                )
                model_file = os.path.join(MODELS_DIR, model_filename)

                # Construct and run the command
                cmd = [
                    "python", "-m", "scripts.metrics",
                    "--og", og_file,
                    "--gen", gen_file,
                    "--model", model_file
                ]
                print("Running command:", " ".join(cmd))

                # Run the command and capture output
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Build a unique output filename (replace spaces and commas for safety)
                safe_dims = dims.replace(" ", "").replace(",", "-")
                output_filename = f"{distribution}_{resolution}_{field}_{safe_dims}.txt"
                output_filepath = os.path.join(OUTPUT_DIR, output_filename)

                with open(output_filepath, "w") as f:
                    if result.stdout:
                        f.write("STDOUT:\n")
                        f.write(result.stdout)
                    if result.stderr:
                        f.write("\nSTDERR:\n")
                        f.write(result.stderr)
                print(f"Metrics output saved to {output_filepath}")

                METRICS_CSV = "metrics.csv"

                # Check if the metrics CSV file exists
                if not os.path.exists(METRICS_CSV):
                    print(f"Metrics CSV file {METRICS_CSV} does not exist. Skipping appending.")
                    continue

                # Read metrics from the file. The CSV is expected to have two columns:
                # the first is the metric name and the second is its value.
                metrics = {}
                with open(METRICS_CSV, "r") as src:
                    reader = csv.reader(src)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        # Strip spaces from both the metric name and its value.
                        metrics[row[0].strip()] = row[1].strip()

                # Define the desired metric names in the desired order.
                desired_metrics = [
                    "SNR (dB)",
                    "NRMSE",
                    "Original Size (MB)",
                    "Reconstructed Size (MB)",
                    "Model Size (MB)",
                    "Compression Ratio"
                ]

                # Build a row for the final CSV with context columns first then metric values.
                final_row = [distribution, resolution, field, dims]
                for metric in desired_metrics:
                    final_row.append(metrics.get(metric, ""))

                # Append the values as a single row (columns instead of rows) to the final CSV.
                header_exists = os.path.exists(FINAL_CSV) and os.path.getsize(FINAL_CSV) > 0
                with open(FINAL_CSV, "a", newline="") as dst:
                    writer = csv.writer(dst)
                    if not header_exists:
                        header = ["Distribution", "Block Size", "Field", "Dims"] + desired_metrics
                        writer.writerow(header)
                    writer.writerow(final_row)
                print(f"Appended {METRICS_CSV} to {FINAL_CSV}")

if __name__ == '__main__':
    main()