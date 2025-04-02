from genericpath import isdir
from math import dist
import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import TruncatedGaussian, UniformUnivariate
import struct
import os
from .utils import read_vti_files


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
def create_copula_model_multivariable(block_vars, x0, y0, z0, marginal_distributions):
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

    copula = GaussianMultivariate(distribution=distribution)
    copula.fit(df)
    return copula

def save_copula_binary(file_name, block_copulas, dims, marginal_types_map):
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

        # Number of scalar variables modelled
        num_vars = len(block_copulas[0][0].to_dict()["univariates"])
        f.write(struct.pack("B", num_vars))  # uint8 (1 byte)
    
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
                if marginal_type in marginal_types_map:
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


###############################################################################
# 7. FULL PIPELINE EXAMPLE
###############################################################################
if __name__ == "__main__":
    
    from .params import file_paths, marginal_distributions, block_size
    from ..marginals_map import marginal_types_map
    
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
        copula = create_copula_model_multivariable(block_vars, i0, j0, k0, marginal_distributions)
        block_copulas.append((copula, i0, j0, k0, block_size))
    print("Fitted copula models for", len(block_copulas), "blocks.")

    if not os.path.isdir("copula_models"):
        os.makedirs("copula_models")
    
    var_names = [key for key in file_paths.keys()]
    dist_names = [marginal_distributions[key].__name__ for key in file_paths.keys()]
            
    # save_copula_binary("copula_models/model_.bin", block_copulas, dims, marginal_types_map)
    save_copula_binary("copula_models/model_" + "_".join(var_names) + "_" + dist_names[-1] + ".bin", block_copulas, dims, marginal_types_map)
    
   
