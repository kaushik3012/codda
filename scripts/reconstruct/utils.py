import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np

def numpy_to_vti(array_3d, var_name, spacing=(1.0,1.0,1.0), origin=(0.0,0.0,0.0)):
    """
    Convert a 3D NumPy array into a vtkImageData object.
    array_3d is assumed to have shape (Nx, Ny, Nz).
    """
    Nx, Ny, Nz = array_3d.shape
    
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(Nx, Ny, Nz)
    imageData.SetSpacing(spacing)
    imageData.SetOrigin(origin)
    
    # Flatten the array in Fortran order so it matches VTK's x-fastest indexing
    flat_array = array_3d.flatten(order='F').astype(np.float32)
    
    # Convert to vtk data array
    vtk_data = numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data.SetName(var_name)  # or any appropriate scalar name
    
    # Attach to vtkImageData
    imageData.GetPointData().SetScalars(vtk_data)
    return imageData

def write_vti(imageData, filename):
    """
    Writes vtkImageData to a .vti XML file.
    """
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()