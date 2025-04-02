from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk


def read_vti_file(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()
    dims = data.GetDimensions()    # (nx, ny, nz)
    spacing = data.GetSpacing()      # (sx, sy, sz)
    origin = data.GetOrigin()        # (ox, oy, oz)

    # Flatten the VTK data into a 1D numpy array
    array = vtk_to_numpy(data.GetPointData().GetScalars())
    # Reshape into 3D; use order='F' if your data are in Fortran order
    array_3d = array.reshape(dims, order='F')
    return array_3d, dims, spacing, origin

def read_vti_files(file_dict):
    """
    file_dict: dictionary mapping variable name (str) to its VTI file path.
    
    Returns:
       arrays: dict mapping variable name to its 3D numpy array.
       dims, spacing, origin: from the first file (all files are assumed to share these).
    """
    arrays = {}
    dims = spacing = origin = None
    for var, path in file_dict.items():
        arr, d, sp, orig = read_vti_file(path)
        arrays[var] = arr
        if dims is None:
            dims, spacing, origin = d, sp, orig
        else:
            # Optionally, add consistency checks for dims, spacing, origin here.
            pass
    return arrays, dims, spacing, origin