from math import sqrt
import numpy as np
import struct
from copulas.multivariate import Multivariate
import numpy as np
from scipy.stats import rv_histogram
import argparse
from ..dist_maps import marginal_types_map, copula_types_map
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

def load_univariates(f, var_names):
    """
    Load univariate distributions from a binary file.
    
    Parameters:
        f (file object): Opened binary file.
        var_names (list): List of variable names.

    Returns:
        list: List of dictionaries containing univariate distribution parameters.
    """
    # Reverse mapping from integer code to marginal type string
    marginal_types_reverse_map = {v: k for k, v in marginal_types_map.items()}

    num_vars = len(var_names)+3  # x, y, z are added to the number of variables
    
    univariates = []
    for _ in range(num_vars):
        marginal_type_code = struct.unpack("B", f.read(1))[0]
        marginal_type = marginal_types_reverse_map[marginal_type_code]
        params = {}
        if marginal_type == "Histogram":
            num_bins = struct.unpack("B", f.read(1))[0]
            bin_densities = np.frombuffer(f.read(num_bins * 4), dtype=np.float32)
            bin_edges = np.frombuffer(f.read((num_bins + 1) * 4), dtype=np.float32)
            bin_densities = bin_densities / np.sum(bin_densities)
            bin_edges = bin_edges.tolist()
            params['histogram'] = rv_histogram((bin_densities, bin_edges), density=True)
            params["type"] = "scripts.custom_models.histogram.Histogram"
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
                params[key] = float(struct.unpack("f", f.read(4))[0])
            
            params["type"] = f"copulas.univariate.{marginal_type}"
        
        univariates.append(params)
    
    return univariates

def load_GaussianCopula_binary(f, var_names):

    num_vars = len(var_names)+3  # x, y, z are added to the number of variables

    copula_structures = []
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
        copula_params["univariates"] = load_univariates(f, var_names)
        copula_params['columns'] = var_names + ["x", "y", "z"]
        
        # Add copula type
        copula_params['type'] = 'copulas.multivariate.GaussianMultivariate'
        copula_structures.append(copula_params)
    return copula_structures

def load_GMC_binary(f, var_names):
    num_vars = len(var_names)+3  # x, y, z are added to the number of variables
    copula_structures = []
    
    while f.peek(1):

        copula_params = {
            'gmm_params': {}
        }

        # Read the number of mixture components
        num_components = struct.unpack("B", f.read(1))[0]
        copula_params['gmm_params']["n_components"] = num_components
        
        copula_params['gmm_params']["weights"] = np.array(np.frombuffer(f.read(num_components * 4), dtype=np.float32))
        copula_params['gmm_params']["weights"] /= np.sum(copula_params['gmm_params']["weights"])

        # Read means
        means = np.frombuffer(f.read(num_components * num_vars * 4), dtype=np.float32)
        means = means.reshape(num_components, num_vars)
        copula_params['gmm_params']["means"] = means

        covs = np.zeros((num_components, num_vars, num_vars), dtype=np.float32)
        for i in range(num_components):
            # Read half of the correlation matrix (symmetric)
            cov_size = num_vars + (num_vars * (num_vars - 1)) // 2
            half_cov = np.frombuffer(f.read(cov_size * 4), dtype=np.float32)
            cov_matrix = np.eye(num_vars, dtype=np.float32)
            indices = np.triu_indices(num_vars, k=0)
            cov_matrix[indices] = half_cov
            cov_matrix[(indices[1], indices[0])] = half_cov
            covs[i] = cov_matrix
        copula_params['gmm_params']["covariances"] = covs

        # Read marginal distributions
        copula_params["univariates"] = load_univariates(f, var_names)
        copula_params["columns"] = var_names + ["x", "y", "z"]
        copula_params["type"] = "scripts.custom_models.gmc.GaussianMixtureCopula"
        copula_structures.append(copula_params)
    return copula_structures

def load_copula_binary(file_name, var_names):
    """
    Load copula parameters from a binary file and reconstruct the copula objects.
    
    Parameters:
        file_name (str): Input binary file.
        marginal_types_reverse_map (dict): Mapping of integer marginal type codes back to their names.

    Returns:
        list: Reconstructed list of copula structures.
    """

    with open(file_name, "rb") as f:
        # Read original dimensions
        original_dims = struct.unpack("3H", f.read(6))  # Read 3 unsigned integers

        # Read type of copula
        copula_type_code = struct.unpack("B", f.read(1))[0]

        # Reverse mapping from integer code to copula type string
        copula_types_reverse_map = {v: k for k, v in copula_types_map.items()}
        copula_type = copula_types_reverse_map[copula_type_code]

        if copula_type == "GaussianCopula" or copula_type == "IndependentMultivariate":
            copula_structures = load_GaussianCopula_binary(f, var_names)
        elif copula_type == "GaussianMixtureCopula":
            copula_structures = load_GMC_binary(f, var_names)
        else:
            raise ValueError(f"Unsupported copula type: {copula_type}")

    print(f"Loaded {len(copula_structures)} copula models from binary file.")
    return copula_structures, original_dims, copula_type

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Script to reconstruct data from Copula Models.")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to generate.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=(250, 250, 50), help="Target shape for reconstruction (Tx, Ty, Tz).")
    parser.add_argument("--vars", type=str, nargs='+', default=None, help="Variable names to reconstruct.")
    parser.add_argument("--model_path", type=str, help="Path to the copula model file.", required=True)
    args = parser.parse_args()

    n_samples = args.n_samples
    target_shape = tuple(args.target_shape)

    # Get the variable names from the model file name
    model_file_name = os.path.basename(args.model_path)
    model_file_split = model_file_name.split(".")[0].split("_")
    var_names = model_file_split[1:-2]
    scalar_vars = args.vars
    if scalar_vars is None:
        scalar_vars = var_names
    elif not set(scalar_vars).issubset(set(var_names)):
        raise ValueError(f"Variable names provided do not exist in the model file: {args.vars}. Expected Choices: {var_names}.")

    # Load copulas from file
    copula_data, dims, copula_type = load_copula_binary(args.model_path, var_names)

    copula_list = []
    # Print reconstructed copula structures
    for i, copula in enumerate(copula_data):
        # Convert to Multivariate object
        copula = Multivariate.from_dict(copula)
        copula_list.append(copula)

    # (F) Reconstruct the field at the original resolution.
    # The reconstruction returns a dictionary mapping variable names to 3D arrays.
    
    S_arbitrary = reconstruct_field_arbitrary_resolution(copula_list, dims, target_shape, scalar_vars, samples_per_block=n_samples)

    # Save the reconstructed fields to VTI files
    # Create a directory to save the files
    model_name =model_file_split[-2]

    save_path = "reconstructed_fields"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)    

    save_path = os.path.join(save_path, copula_type)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, model_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    block_size = model_file_split[-1]
    if not os.path.isdir(save_path+"/"+block_size):
        os.mkdir(save_path+"/"+block_size)
    save_path = save_path+"/"+block_size

    # Define the spacing for the VTI files
    spacing = (1.0, 1.0, 1.0)

    for var in scalar_vars:

        # Convert your NumPy array to vtkImageData
        imageData = numpy_to_vti(S_arbitrary[var],var, spacing=spacing)
        
        # Write out to file
        write_vti(imageData, save_path+"/reconstructed_field_"+var+"_"+str(target_shape)+".vti")
        
        print("Finished writing "+save_path+"/reconstructed_field_"+var+"_"+str(target_shape)+".  You can now open it in ParaView!")

    