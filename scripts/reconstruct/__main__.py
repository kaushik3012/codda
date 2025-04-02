from math import sqrt
import numpy as np
import struct
from copulas.multivariate import Multivariate
import numpy as np
from scipy.stats import rv_histogram
import argparse
from ..marginals_map import marginal_types_map
from .utils import numpy_to_vti, write_vti
import os

###############################################################################
# 4. SAMPLING UTILITIES
###############################################################################
def distance_3d(x1, y1, z1, x2, y2, z2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def sample_block_copula(copula, N):
    """
    Generate N samples from the block's copula.
    Returns a DataFrame with columns corresponding to each scalar variable and x, y, z.
    """
    samples_df = copula.sample(N)
    return samples_df

###############################################################################
# 5. RECONSTRUCT AT ORIGINAL RESOLUTION (FOR MULTIPLE VARIABLES)
###############################################################################
def reconstruct_field_original_resolution(block_copulas, global_shape, scalar_vars, samples_per_block=1000):
    """
    block_copulas: list of tuples (copula, x0, y0, z0, block_size).
    global_shape: (Nx, Ny, Nz) of the original field.
    scalar_vars: list of scalar variable names.
    
    Returns a dictionary mapping each scalar variable to its reconstructed 3D array.
    """
    Nx, Ny, Nz = global_shape
    S = {var: np.zeros((Nx, Ny, Nz), dtype=float) for var in scalar_vars}
    W = {var: np.zeros((Nx, Ny, Nz), dtype=float) for var in scalar_vars}
    
    for (copula, x0, y0, z0, block_size) in block_copulas:
        df_samples = sample_block_copula(copula, samples_per_block)
        for idx, row in df_samples.iterrows():
            x_float = row['x']
            y_float = row['y']
            z_float = row['z']
            
            base_x = int(np.floor(x_float))
            base_y = int(np.floor(y_float))
            base_z = int(np.floor(z_float))
            base_x = max(0, min(base_x, Nx - 2))
            base_y = max(0, min(base_y, Ny - 2))
            base_z = max(0, min(base_z, Nz - 2))
            
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        gx = base_x + i
                        gy = base_y + j
                        gz = base_z + k
                        dist = distance_3d(x_float, y_float, z_float, gx, gy, gz)
                        if dist < 1e-9:
                            dist = 1e-9
                        weight = 1.0 / dist
                        for var in scalar_vars:
                            S[var][gx, gy, gz] += row[var] * weight
                            W[var][gx, gy, gz] += weight

    for var in scalar_vars:
        mask = (W[var] > 1e-12)
        S[var][mask] /= W[var][mask]
        
    return S

###############################################################################
# 6. RECONSTRUCT AT ARBITRARY RESOLUTION (FOR MULTIPLE VARIABLES)
###############################################################################
def reconstruct_field_arbitrary_resolution(block_copulas, original_shape, target_shape, scalar_vars, samples_per_block=10000):
    """
    block_copulas: list of copulas.
    original_shape: (Nx, Ny, Nz) of the original data.
    target_shape: (Tx, Ty, Tz) for the new grid.
    scalar_vars: list of scalar variable names.
    
    Returns a dictionary mapping each scalar variable to its reconstructed 3D array at the target resolution.
    """
    Nx, Ny, Nz = original_shape
    Tx, Ty, Tz = target_shape
    
    S = {var: np.zeros((Tx, Ty, Tz), dtype=float) for var in scalar_vars}
    W = {var: np.zeros((Tx, Ty, Tz), dtype=float) for var in scalar_vars}
    
    for copula in block_copulas:
        df_samples = sample_block_copula(copula, samples_per_block)
        for idx, row in df_samples.iterrows():
            x_old = row['x']
            y_old = row['y']
            z_old = row['z']
            
            x_new = (x_old / (Nx - 1)) * (Tx - 1) if Nx > 1 and Tx > 1 else 0.0
            y_new = (y_old / (Ny - 1)) * (Ty - 1) if Ny > 1 and Ty > 1 else 0.0
            z_new = (z_old / (Nz - 1)) * (Tz - 1) if Nz > 1 and Tz > 1 else 0.0
            
            base_x = int(np.floor(x_new))
            base_y = int(np.floor(y_new))
            base_z = int(np.floor(z_new))
            base_x = max(0, min(base_x, Tx - 2))
            base_y = max(0, min(base_y, Ty - 2))
            base_z = max(0, min(base_z, Tz - 2))
            
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        gx = base_x + i
                        gy = base_y + j
                        gz = base_z + k
                        dist = sqrt((x_new - gx)**2 + (y_new - gy)**2 + (z_new - gz)**2)
                        if dist < 1e-9:
                            dist = 1e-9
                        weight = 1.0 / dist
                        for var in scalar_vars:
                            S[var][gx, gy, gz] += row[var] * weight
                            W[var][gx, gy, gz] += weight
    
    for var in scalar_vars:
        mask = (W[var] > 1e-12)
        S[var][mask] /= W[var][mask]
        
    return S


def load_copula_binary(file_name, marginal_types_reverse_map):
    """
    Load copula parameters from a binary file and reconstruct the copula objects.
    
    Parameters:
        file_name (str): Input binary file.
        marginal_types_reverse_map (dict): Mapping of integer marginal type codes back to their names.

    Returns:
        list: Reconstructed list of copula structures.
    """
    copula_structures = []

    with open(file_name, "rb") as f:
        # Read original dimensions
        original_dims = struct.unpack("3H", f.read(6))  # Read 3 unsigned integers

        # Read number of variables in the copula
        num_vars = struct.unpack("B", f.read(1))[0]

        while f.peek(1):
            copula_params = {}

            # Read half of the correlation matrix (symmetric)
            corr_size = (num_vars * (num_vars - 1)) // 2
            half_corr = np.frombuffer(f.read(corr_size * 4), dtype=np.float32)  # Read as float32

            # Reconstruct full correlation matrix
            corr_matrix = np.eye(num_vars, dtype=np.float32)
            indices = np.triu_indices(num_vars, k=1)
            corr_matrix[indices] = half_corr
            corr_matrix[(indices[1], indices[0])] = half_corr  # Mirror the upper triangle
            copula_params["correlation"] = corr_matrix

            # Read marginal distributions
            univariates = []
            for _ in range(num_vars):
                # Read marginal type
                marginal_type_code = struct.unpack("B", f.read(1))[0]
                marginal_type = marginal_types_reverse_map[marginal_type_code]

                # Read marginal parameters
                params = {}
                # Handle Histogram separately
                if marginal_type == "Histogram":
                    # Read histogram data
                    num_bins = struct.unpack("B", f.read(1))[0]
                    
                    bin_densities = np.frombuffer(f.read(num_bins * 4), dtype=np.float32)
                    bin_edges = np.frombuffer(f.read((num_bins + 1) * 4), dtype=np.float32)
                    bin_densities = bin_densities / np.sum(bin_densities)  # Normalize
                    bin_edges = bin_edges.tolist()
                    params['histogram'] = rv_histogram((bin_densities, bin_edges), density=True)
                    params["type"] = "scripts.custom_univariates.Histogram"
                else:
                    if marginal_type == "TruncatedGaussian":
                        keys = ["a", "b", "loc", "scale"]
                    elif marginal_type == "UniformUnivariate":
                        keys = ["loc", "scale"]
                    elif marginal_type == "GaussianUnivariate":
                        keys = ["loc", "scale"]
                    elif marginal_type == "BetaUnivariate":
                        keys = ["a", "b", "loc", "scale"]
                    elif marginal_type == "GammaUnivariate":
                        keys = ["a", "loc", "scale"]
                    elif marginal_type == "LogLaplace":
                        keys = ["c", "loc", "scale"]
                    elif marginal_type == "StudentTUnivariate":
                        keys = ["df", "loc", "scale"]
                    else:
                        raise ValueError(f"Unsupported marginal type: {marginal_type}")

                    for key in keys:
                        # Read numpy float32
                        params[key] = struct.unpack("f", f.read(4))[0]
                        # Convert to float
                        params[key] = float(params[key])

                    params["type"] = f"copulas.univariate.{marginal_type}"
                univariates.append(params)

            copula_params["univariates"] = univariates
            # copula_params['columns'] = [f"var_{i}" for i in range(num_vars)[:-3]] + ["x", "y", "z"]
            model_file_name = os.path.basename(file_name)
            model_file_split = model_file_name.split(".")[0].split("_")
            var_names = model_file_split[1:-1]
            copula_params['columns'] = var_names + ["x", "y", "z"]
            copula_params['type'] = 'copulas.multivariate.GaussianMultivariate'
            copula_structures.append(copula_params)

    print(f"Loaded {len(copula_structures)} copula models from binary file.")
    return copula_structures, original_dims, num_vars

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Script to reconstruct data from Copula Models.")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to generate.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=(250, 250, 50), help="Target shape for reconstruction (Tx, Ty, Tz).")
    # parser.add_argument("--var_names", type=str, nargs='+', default=["Pressure", "Temperature", "Velocity"], help="Variable names to reconstruct.")
    parser.add_argument("--vars", type=str, nargs='+', default=None, help="Variable names to reconstruct.")
    parser.add_argument("--model_path", type=str, help="Path to the copula model file.", required=True)
    args = parser.parse_args()

    n_samples = args.n_samples
    target_shape = tuple(args.target_shape)
    
    # Reverse mapping from integer code to marginal type string
    marginal_types_reverse_map = {v: k for k, v in marginal_types_map.items()}

    # Get the variable names from the model file name
    model_file_name = os.path.basename(args.model_path)
    model_file_split = model_file_name.split(".")[0].split("_")
    var_names = model_file_split[1:-1]
    scalar_vars = args.vars
    if scalar_vars is None:
        scalar_vars = var_names
    elif not set(scalar_vars).issubset(set(var_names)):
        raise ValueError(f"Variable names provided do not exist in the model file: {args.vars}. Expected Choices: {var_names}.")
        
    # Load copulas from file
    copula_data, dims, num_vars = load_copula_binary(args.model_path, marginal_types_reverse_map)

    copula_list = []
    # Print reconstructed copula structures
    for i, copula in enumerate(copula_data):
        # Convert to GaussianMultivariate object
        copula = Multivariate.from_dict(copula)
        copula_list.append(copula)

    # (F) Reconstruct the field at the original resolution.
    # The reconstruction returns a dictionary mapping variable names to 3D arrays.
    
    S_arbitrary = reconstruct_field_arbitrary_resolution(copula_list, dims, target_shape, scalar_vars, samples_per_block=n_samples)

    # Save the reconstructed fields to VTI files
    # Create a directory to save the files
    model_name =model_file_split[-1]
    if not os.path.isdir("reconstructed_fields"):
        os.mkdir("reconstructed_fields")    
    save_path = "reconstructed_fields/"+model_name
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Define the spacing for the VTI files
    spacing = (1.0, 1.0, 1.0)

    for var in scalar_vars:

        # Convert your NumPy array to vtkImageData
        imageData = numpy_to_vti(S_arbitrary[var],var, spacing=spacing)
        
        # Write out to file
        write_vti(imageData, save_path+"/reconstructed_field_"+var+"_"+str(target_shape)+".vti")
        
        print("Finished writing "+save_path+"/reconstructed_field_"+var+"_"+str(target_shape)+".  You can now open it in ParaView!")

    