import argparse
import os
import numpy as np
from .compress.utils import read_vti_file
from .reconstruct.utils import numpy_to_vti, write_vti

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a vti file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original data file")
    parser.add_argument("--target_shape", type=int, nargs=3, default=(250, 250, 50), help="Target shape for reconstruction (Tx, Ty, Tz).")
    args = parser.parse_args()

    input_file = args.input
    target_shape = tuple(args.target_shape)

    # Subsample the original data to match the new resolution
    original_data,original_shape,_,_  = read_vti_file(args.input)
    original_data = np.array(original_data)
    subsampled_data = np.zeros(target_shape)

    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            for k in range(target_shape[2]):
                x = i * (original_shape[0] - 1) / (target_shape[0] - 1)
                y = j * (original_shape[1] - 1) / (target_shape[1] - 1)
                z = k * (original_shape[2] - 1) / (target_shape[2] - 1)
                i0 = int(np.floor(x))
                j0 = int(np.floor(y))
                k0 = int(np.floor(z))
                i1 = min(i0 + 1, original_shape[0] - 1)
                j1 = min(j0 + 1, original_shape[1] - 1)
                k1 = min(k0 + 1, original_shape[2] - 1)
                dx = x - i0
                dy = y - j0
                dz = z - k0
                subsampled_data[i, j, k] = (
                    (1 - dx) * (1 - dy) * (1 - dz) * original_data[i0, j0, k0] +
                    (1 - dx) * (1 - dy) * dz * original_data[i0, j0, k1] +
                    (1 - dx) * dy * (1 - dz) * original_data[i0, j1, k0] +
                    (1 - dx) * dy * dz * original_data[i0, j1, k1] +
                    dx * (1 - dy) * (1 - dz) * original_data[i1, j0, k0] +
                    dx * (1 - dy) * dz * original_data[i1, j0, k1] +
                    dx * dy * (1 - dz) * original_data[i1, j1, k0] +
                    dx * dy * dz * original_data[i1, j1, k1]
                )

    # Suppose S_reconstructed.shape = (250,250,50)
    # and you know the voxel spacing, e.g. (sx, sy, sz)
    spacing = (1.0, 1.0, 1.0)

    # Get the variable name from the file path (eg: "Pf25.binLE.raw_corrected_2_subsampled.vti")
    var = os.path.basename(input_file).split(".")[0]

    if not os.path.isdir("subsampled_field"):
        os.mkdir("subsampled_field")    
    save_path = "subsampled_field"

    # Convert your NumPy array to vtkImageData
    imageData = numpy_to_vti(subsampled_data,var, spacing=spacing)

    # Write out to file
    write_vti(imageData, f"{save_path}/{var}_{target_shape}.vti")

    print(f"Finished writing '{save_path}/{var}_{target_shape}.vti'.  You can now open it in ParaView!")

