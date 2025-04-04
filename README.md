# Copula Project README

This document outlines how to run the various scripts in the project and details all available parameters for each script.

## Prerequisites

- Python 3.9+
- Required Python packages (install via `pip install -r requirements.txt`)

## Running Scripts

Each script is designed to perform a specific task. The following sections provide usage instructions and explain the parameters.

---

### 1. reconstruct

This script reconstructs data using a pre-trained model.

**Usage:**

    python -m scripts.reconstruct --n_samples <int> --vars <str> --model_path <path_to_model> --target_shape <dim1> <dim2> <dim3>

**Parameters:**

- `--n_samples`: (int) Number of samples to reconstruct.
- `--vars`: (string) Variables to process (e.g., P, TC, Velocity).
- `--model_path`: (path) File path to the pre-saved model.
- `--target_shape`: (int, int, int) Dimensions for the reconstructed output (e.g., width, height, depth).

**Example:**

    python -m scripts.reconstruct --n_samples 250 --vars P --model_path copula_models/GaussianMixtureCopula/model_P_TC_Velocity_CLOUD_PRECIP_QCLOUD_QGRAUP_QICE_QRAIN_QSNOW_QVAPOR_U_V_W_Histogram_5.bin --target_shape 100 100 20

### 2. compress

This script compresses data using predefined parameters.

**Usage:**

    python -m scripts.compress

**Parameters:**

All parameters for this script can be manually adjusted in the `compress/params.py` file. Ensure the file is correctly configured before running the script.

### 3. metrics

This script evaluates the quality of reconstructed data by comparing it with the original data.

**Usage:**

    python -m scripts.metrics --og <path_to_original_field> --gen <path_to_generated_field> --model <path_to_model>

**Parameters:**

- `--og`: (path) File path to the original data field.
- `--gen`: (path) File path to the generated/reconstructed data field.
- `--model`: (path) File path to the model used for reconstruction.

**Example:**

    python -m scripts.metrics --og subsampled_field/QVAPORf25_(250, 250, 50).vti --gen reconstructed_fields/GaussianMultivariate/Histogram/5/reconstructed_field_QVAPOR_(250, 250, 50).vti --model copula_models/GaussianMultivariate/model_P_TC_Velocity_CLOUD_PRECIP_QCLOUD_QGRAUP_QICE_QRAIN_QSNOW_QVAPOR_U_V_W_Histogram_5.bin

### 4. metrics_all

This script evaluates the quality of reconstructed data across multiple fields and models.

**Usage:**

    python -m scripts.metrics_all

**Parameters:**

All parameters for this script can be manually adjusted in the `metrics_all.py` file. Ensure the file is correctly configured before running the script.

**Example:**

    python -m scripts.metrics_all

### 5. subsample

This script downsamples a given dataset to a specified target shape.

**Usage:**

    python -m scripts.subsample --input <file_path> --target_shape <dim1> <dim2> <dim3>

**Parameters:**

- `--input`: (path) File path to the input dataset.
- `--target_shape`: (int, int, int) Dimensions for the downsampled output (e.g., width, height, depth).

**Example:**

    python -m scripts.subsample --input data/original_field.vti --target_shape 100 100 20

### 6. subsample_all

This script downsamples multiple datasets to specified target shapes.

**Usage:**

    python -m subsample_all

**Parameters:**

All parameters for this script can be manually adjusted in the `subsample_all.py` file. Ensure the file is correctly configured before running the script.

**Example:**

    python -m subsample_all


## Additional Notes

- Always verify the paths provided for data and models.
- Use the `--help` flag with any script to get a detailed explanation of its parameters. For example:  
    python -m scripts.reconstruct --help

- Ensure all dependencies are installed and environment variables are properly set if required.

Happy Coding!