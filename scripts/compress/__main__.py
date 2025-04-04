from genericpath import isdir
from math import dist
import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from ..custom_univariates.gmc import GaussianMixtureCopula
from copulas.univariate import TruncatedGaussian, UniformUnivariate
from sklearn.mixture import GaussianMixture
import struct
import os
from .utils import read_vti_files
from .params import file_paths, marginal_distributions, block_size, copula_type
from ..dist_maps import marginal_types_map, copula_types_map


###############################################################################
# 2. DIVIDE INTO BLOCKS (USING THE GRID FROM ONE OF THE VARIABLES)
###############################################################################
def divide_into_blocks(array, block_size):
    """
    Returns a list of (block, i0, j0, k0) for each full-size sub-block.
    'array' is one 3D numpy array used solely for determining the block indices.
    """
    blocks = []
    nx, ny, nz = array.shape
    for i in range(0, nx, block_size):
        for j in range(0, ny, block_size):
            for k in range(0, nz, block_size):
                block = array[i:i+block_size, j:j+block_size, k:k+block_size]
                # Only keep if it is a 'full-size' block:
                if block.shape == (block_size, block_size, block_size):
                    blocks.append((block, i, j, k))
    return blocks

###############################################################################
# 3. FIT A COPULA MODEL FOR EACH BLOCK (MULTIVARIABLE VERSION WITH MARGINALS)
###############################################################################
def create_copula_model_multivariable(block_vars, x0, y0, z0, marginal_distributions, copula_type="GaussianMultivariate"):
    """
    block_vars: dictionary mapping each scalar variable name to its 3D block 
                (shape: (block_size, block_size, block_size)).
    (x0, y0, z0): integer coordinates of the block's "top-left-front" corner.
    marginal_distributions: dictionary mapping variable names to univariate distribution
                            classes to be used in the copula model.
                            For example:
                            {
                              "Pressure": TruncatedGaussian,
                              "Temperature": TruncatedGaussian,
                              "Velocity": TruncatedGaussian,
                              "x": UniformUnivariate,
                              "y": UniformUnivariate,
                              "z": UniformUnivariate
                            }
    Returns a GaussianMultivariate copula fitted to the flattened scalar data plus coordinates.
    """
    # Assume all blocks have the same shape:
    sample_block = next(iter(block_vars.values()))
    block_nx, block_ny, block_nz = sample_block.shape
    
    # Create coordinate array for each voxel in the block:
    coords = []
    for i in range(block_nx):
        for j in range(block_ny):
            for k in range(block_nz):
                coords.append((x0 + i, y0 + j, z0 + k))
    coords = np.array(coords)  # shape: (block_nx*block_ny*block_nz, 3)
    
    # Build a data dictionary: one column per variable
    data_dict = {}
    for var, block in block_vars.items():
        data_dict[var] = block.flatten()
    
    # Add coordinate columns:
    data_dict['x'] = coords[:, 0]
    data_dict['y'] = coords[:, 1]
    data_dict['z'] = coords[:, 2]
    
    df = pd.DataFrame(data_dict)
    
    # Build the distribution mapping:
    distribution = {}
    for var in block_vars.keys():
        # Use the provided marginal distribution, defaulting to TruncatedGaussian if not specified.
        distribution[var] = marginal_distributions.get(var, TruncatedGaussian)
    # For coordinates, use the provided marginal distribution or default to UniformUnivariate.
    distribution['x'] = marginal_distributions.get('x', UniformUnivariate)
    distribution['y'] = marginal_distributions.get('y', UniformUnivariate)
    distribution['z'] = marginal_distributions.get('z', UniformUnivariate)

    num_vars = len(block_vars)

    if copula_type == "IndependentMultivariate":
        copula = GaussianMultivariate(distribution=distribution)
        copula.fit(df)
        copula.correlation = pd.DataFrame(np.eye(num_vars))
    elif copula_type == "GaussianMultivariate":
        copula = GaussianMultivariate(distribution=distribution)
        copula.fit(df)
    elif copula_type == "GaussianMixtureCopula":
        # Estimate the number of components using BIC
        bic_scores = []
        for n_components in range(1, 5):
            gmm = GaussianMixtureCopula(distribution=distribution, n_components=n_components, random_state=0)
            gmm.fit(df)
            bic_scores.append(gmm.bic(df))
        estimated_components = int(np.argmin(bic_scores) + 1)  # +1 because n_components starts from 1
        copula = GaussianMixtureCopula(distribution=distribution, n_components=estimated_components)
        copula.fit(df)
    else:
        raise ValueError(f"Unknown copula type: {copula_type}")
    return copula

def save_GaussianCopula_binary(file_name, block_copulas, dims):
    """
    Save copula parameters in a binary format compatible with C++.
    
    Parameters:
        file_name (str): Output binary file.
        block_copulas (list): List of tuples (copula, i0, j0, k0, block_size).
        block_size (int): Block size (single global value).
        marginal_types_map (dict): Mapping of variable names to integer marginal types.
    """
    with open(file_name, "wb") as f:

        # Write original dimensions
        f.write(struct.pack('H', dims[0]))  # uint16 (2 bytes)
        f.write(struct.pack('H', dims[1]))  # uint16 (2 bytes)
        f.write(struct.pack('H', dims[2]))  # uint16 (2 bytes)

        # Type of copula
        copula_type_code = copula_types_map.get(copula_type, 0)
        f.write(struct.pack("B", copula_type_code))  # uint8 (1 byte)

        # # Number of scalar variables modelled
        # num_vars = len(block_copulas[0][0].to_dict()["univariates"])
        # f.write(struct.pack("B", num_vars))  # uint8 (1 byte)
    
        for copula, _,_,_, _ in block_copulas:

            # Get copula parameters
            copula_params = copula.to_dict()  # Dictionary of params

            # Store half of the correlation matrix (since it's symmetric)
            corr_matrix = np.array(copula_params["correlation"], dtype=np.float32)
            size = corr_matrix.shape[0]
            half_corr = corr_matrix[np.triu_indices(size, k=1)]  # Extract upper triangle (excluding diagonal)
            f.write(half_corr.tobytes())  # Store as raw float32 values

            # Store marginal types
            for params_dict in copula_params["univariates"]:
                marginal_type = params_dict["type"].split(".")[-1]
                if marginal_type in marginal_types_map.keys():
                    marginal_type_code = marginal_types_map[marginal_type]
                    f.write(struct.pack("B", marginal_type_code))
                else:
                    raise ValueError(f"Unknown marginal type: {marginal_type}")
                
                if marginal_type == "Histogram":
                    # Store histogram parameters
                    hist_dist = params_dict["histogram"]
                    bin_densities = hist_dist._histogram[0]
                    bin_edges = hist_dist._histogram[1]

                    num_bins = len(bin_densities)
                    f.write(struct.pack("B", num_bins))

                    # Store bin densities
                    bin_densities = np.array(bin_densities, dtype=np.float32)
                    f.write(bin_densities.tobytes())
                    # Store bin edges
                    bin_edges = np.array(bin_edges, dtype=np.float32)
                    f.write(bin_edges.tobytes())
                    continue
                
                # Store marginal parameters
                for key, value in params_dict.items():
                    if key != "type":
                        if isinstance(value, list):
                            # Convert list to float32 array
                            value = np.array(value, dtype=np.float32)
                            f.write(value.tobytes())
                        else:
                            # Convert single value to float32
                            f.write(struct.pack("f", float(value)))

    print(f"Copula models saved in binary format: {file_name} of size {os.path.getsize(file_name)} bytes.")

def save_GMC_binary(file_name, block_copulas, dims):
    """
    Save Gaussian Mixture Copula parameters in a binary format compatible with C++.

    Parameters:
        file_name (str): Output binary file.
        block_copulas (list): List of tuples (copula, i0, j0, k0, block_size).
        dims (tuple): Original dimensions of the data.
    
    The binary file structure is:
        - 3 x uint16: original dims (each 2 bytes).
        - 1 x uint8: copula type code.
    For each block:
        - 1 x uint8: number of mixture components.
        - For each component:
            - 4 bytes: weight (float32).
        - For each component:
            - (n*(n-1)/2) x float32: half of the correlation matrix (upper triangular, excluding diagonal),
              where n is the number of scalar variables.
        - For each univariate distribution:
            - 1 x uint8: marginal type code.
            - Followed by marginal parameters:
                * If the marginal type is "Histogram", write:
                    - 1 x uint8: number of bins.
                    - bin densities as float32 array.
                    - bin edges as float32 array.
                * Otherwise, write each parameter as float32 (if a list, convert to float32 array).
    """
    # Assume copula_types_map and marginal_types_map are imported from the proper module
    # In this context, "copula_type" is the string identifier for GaussianMixtureCopula.
    cur_copula_type = "GaussianMixtureCopula"
    
    with open(file_name, "wb") as f:
        # Write original dimensions (3 x uint16)
        f.write(struct.pack('H', dims[0]))
        f.write(struct.pack('H', dims[1]))
        f.write(struct.pack('H', dims[2]))

        # Write copula type code (1 x uint8)
        copula_type_code = copula_types_map.get(cur_copula_type, 0)
        f.write(struct.pack("B", copula_type_code))
    
        # Process each block's copula model
        for copula, _, _, _, _ in block_copulas:
            copula_params = copula.to_dict()  # Get the dictionary with all parameters

            # Write mixture model parameters:
            # Write number of mixture components (uint8)
            weights = copula_params['gmm_params']['weights']
            num_components = len(weights)
            f.write(struct.pack("B", num_components))
            
            # Write each component's weight (float32)
            for w in weights:
                f.write(struct.pack("f", float(w)))

            # Write Means
            # For each component, write its mean vector.
            means = copula_params['gmm_params']['means']
            for mean in means:
                mean_vector = np.array(mean, dtype=np.float32)
                f.write(mean_vector.tobytes())
            
            # For each component, write half of its covariance matrix.
            # It is assumed that copula_params["components"] is a list of component parameters.
            covariances = copula_params['gmm_params']['covariances']
            for cov in covariances:
                # Each component is assumed to have a "covariance" matrix.
                cov_matrix = np.array(cov, dtype=np.float32)
                size = cov_matrix.shape[0]
                half_cov = cov_matrix[np.triu_indices(size, k=0)]
                f.write(half_cov.tobytes())
            
            # Write univariate parameters (assumed same as for GaussianCopula)
            for params_dict in copula_params["univariates"]:
                marginal_type_str = params_dict["type"].split(".")[-1]
                if marginal_type_str in marginal_types_map.keys():
                    marginal_type_code = marginal_types_map[marginal_type_str]
                    f.write(struct.pack("B", marginal_type_code))
                else:
                    raise ValueError(f"Unknown marginal type: {marginal_type_str}")
                
                if marginal_type_str == "Histogram":
                    hist_dist = params_dict["histogram"]
                    bin_densities = hist_dist._histogram[0]
                    bin_edges = hist_dist._histogram[1]
                    
                    num_bins = len(bin_densities)
                    f.write(struct.pack("B", num_bins))
                    
                    # Write bin densities as float32 array
                    bin_densities = np.array(bin_densities, dtype=np.float32)
                    f.write(bin_densities.tobytes())
                    # Write bin edges as float32 array
                    bin_edges = np.array(bin_edges, dtype=np.float32)
                    f.write(bin_edges.tobytes())
                    continue
                    
                # For other types, write remaining parameters as float32 values.
                for key, value in params_dict.items():
                    if key != "type":
                        if isinstance(value, list):
                            value = np.array(value, dtype=np.float32)
                            f.write(value.tobytes())
                        else:
                            f.write(struct.pack("f", float(value)))
                            
    print(f"Gaussian Mixture Copula models saved in binary format: {file_name} of size {os.path.getsize(file_name)} bytes.")


###############################################################################
# 7. FULL PIPELINE EXAMPLE
###############################################################################
if __name__ == "__main__":
    
    # Read all VTI files. All files are assumed to share the same dims, spacing, and origin.
    arrays, dims, spacing, origin = read_vti_files(file_paths)
    
    # Use one variable’s array to get block indices (they are assumed to be on the same grid)
    key0 = list(arrays.keys())[0]
    blocks_info = divide_into_blocks(arrays[key0], block_size)
    
    # Fit a copula for each block using the multivariable data.
    print("Fitting copula models for each block...")
    block_copulas = []
    for (block, i0, j0, k0) in blocks_info:
        # For each variable, extract the corresponding block.
        block_vars = {var: arr[i0:i0+block_size, j0:j0+block_size, k0:k0+block_size] 
                      for var, arr in arrays.items()}
        copula = create_copula_model_multivariable(block_vars, i0, j0, k0, marginal_distributions, copula_type)
        block_copulas.append((copula, i0, j0, k0, block_size))
    print("Fitted copula models for", len(block_copulas), "blocks.")

    
    save_path = "copula_models"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, copula_type)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    var_names = [key for key in file_paths.keys()]
    dist_names = [marginal_distributions[key].__name__ for key in file_paths.keys()]
        
    if copula_type == "GaussianMultivariate" or copula_type == "IndependentMultivariate":
        save_GaussianCopula_binary(save_path+"/model_" + "_".join(var_names) + "_" + dist_names[-1] + "_"+str(block_size)+".bin", block_copulas, dims)
    elif copula_type == "GaussianMixtureCopula":
        save_GMC_binary(save_path+"/model_" + "_".join(var_names) + "_" + dist_names[-1] + "_"+str(block_size)+".bin", block_copulas, dims)

   
