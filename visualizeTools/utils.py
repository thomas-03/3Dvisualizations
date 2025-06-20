import numpy as np
import h5py
import sys
import os
import toml
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import k3d
import time

#Finds the index in an array whose value is closest to the value you input
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = np.argmin((np.abs(array - value)))
    return idx

#Finds the value in an array closest to the value you input
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return array[idx]

def find_idx_above_below(dataInfo, x,y,z):
    nz,ny,nx = dataInfo.conf['grid_shape']

    xmin=dataInfo.conf['lower'][0]
    ymin=dataInfo.conf['lower'][1]
    zmin=dataInfo.conf['lower'][2]

    sizex=dataInfo.conf['size'][0]
    sizey=dataInfo.conf['size'][1]
    sizez=dataInfo.conf['size'][2]

    xmax=xmin+sizex
    ymax=ymin+sizey
    zmax=zmin+sizez

    dx=sizex/nx
    dy=sizey/ny
    dz=sizez/nz

    x0=np.arange(xmin,xmax,dx)
    y0=np.arange(ymin,ymax,dy)
    z0=np.arange(zmin,zmax,dz)

    nearx0 = find_nearest_index(x0,x)
    neary0 = find_nearest_index(y0,y)
    nearz0 = find_nearest_index(z0,z)

    if(nearx0<= x0.size-1):
        rx = nearx0
        lx = nearx0 - 1
    elif(nearx0 >= 0):
        lx = 0
        rx = 1
    elif(x0[nearx0]>x):
        rx = nearx0
        lx = nearx0-1
    elif(x0[nearx0]<=x):
        lx = nearx0
        rx = nearx0+1
        
    if(neary0<= y0.size-1):
        ry = neary0
        ly = neary0 - 1
    elif(neary0 >= 0):
        ly = 0
        ry = 1    
    elif(y0[neary0]>y):
        ry = neary0
        ly = neary0-1
    elif(y0[neary0]<=y):
        ly = neary0
        ry = neary0+1
        
    if(nearz0<= z0.size-1):
        rz = nearz0
        lz = nearz0 - 1
    elif(nearz0 >= 0):
        lz = 0
        rz = 1
    elif(z0[nearz0]>z):
        rz = nearz0
        lz = nearz0-1
    elif(z0[nearz0]<=z):
        lz = nearz0
        rz = nearz0+1
        
    return lx, rx, ly, ry, lz, rz



def readTomlFile(filename):
    """
    Reads a TOML file and returns its content as a dictionary.
    
    Parameters:
    - filename: Path to the TOML file.
    
    Returns:
    - Dictionary containing the TOML file content.
    """
    try:
        with open(filename, 'r') as f:
            return toml.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML file {filename}: {e}")
        sys.exit(1)

def formatDataArray(dataArr, dataInfo):
    """
    Formats the data array by converting HDF5 data to NumPy arrays and flipping them if necessary.
    
    Parameters:
    - dataArr: List of dictionaries containing HDF5 data.
    - dataInfo: Dictionary containing metadata about the data.
    
    Returns:
    - List of dictionaries with formatted NumPy arrays.
    """
    formatted_data = []
    
    for data in dataArr:
        numpy_data = h5ToNumpy(data)
        formatted_data.append(numpy_data)
    
    return flipArrays(formatted_data, dataInfo)

def flipArrays(dataArr,dataInfo):
    """
    Flips the arrays in the data dictionary such that the x and z axes are swapped if necessary.
    
    Parameters:
    - data: Dictionary containing arrays to be flipped.
    
    Returns:
    - Dictionary with flipped arrays.
    """
    #use the very first data file to determine the grid shape
    data = dataArr[0]
    nx, ny, nz = dataInfo['N']

    if 'grid_shape' not in dataInfo:
        gx, gy, gz = data[list(data.keys())[0]].shape
        dataInfo['grid_shape'] = (nx, ny, nz)
    else:
        gx, gy, gz = dataInfo['grid_shape']
    

    if ((nx<nz) and not(gx<gz)) or ((nx>=nz) and not(gx>=gz)):
        for i in range(len(dataArr)):
            for key in list(dataArr[i].keys()):
                dataArr[i][key] = dataArr[i][key].T
            
    return dataArr

def h5ToNumpy(data):
    """
    Converts HDF5 data to NumPy arrays.
    
    Parameters:
    - data: HDF5 data object.
    
    Returns:
    - Dictionary with NumPy arrays.
    """
    numpy_data = {}
    for key in data.keys():
        numpy_data[key] = np.array(data[key])
    return numpy_data
