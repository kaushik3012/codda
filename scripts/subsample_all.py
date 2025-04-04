import os
import subprocess

# Define the directory containing the files
data_dir = "Isabel_data_all_variables_vti"

# Define the target shapes
target_shapes = [(250,250,50)]

# Iterate over all files in the directory
for file_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file_name)
    
    # Skip if it's not a file
    if not os.path.isfile(file_path):
        continue
    print(f"Processing file: {file_path}")
    # Run the subsample.py script for each target shape
    for target_shape in target_shapes:
        target_shape_str = f"{target_shape[0]},{target_shape[1]},{target_shape[2]}"
        subprocess.run(["python", "-m", "scripts.subsample", "--input", file_path, "--target_shape", str(target_shape[0]), str(target_shape[1]), str(target_shape[2])])