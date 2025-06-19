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

def find_idx_above_below(data1, x,y,z):
    nz,ny,nx = data1.Bx.shape

    xmin=data1.conf['lower'][0]
    ymin=data1.conf['lower'][1]
    zmin=data1.conf['lower'][2]

    sizex=data1.conf['size'][0]
    sizey=data1.conf['size'][1]
    sizez=data1.conf['size'][2]

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

#3D Linear Interpolation of the array value of a point
def interpolate(data1, h, val_array, x, y, z):
    nz,ny,nx = data1.Bx.shape

    xmin=data1.conf['lower'][0]
    ymin=data1.conf['lower'][1]
    zmin=data1.conf['lower'][2]

    sizex=data1.conf['size'][0]
    sizey=data1.conf['size'][1]
    sizez=data1.conf['size'][2]

    xmax=xmin+sizex
    ymax=ymin+sizey
    zmax=zmin+sizez

    dx=sizex/nx
    dy=sizey/ny
    dz=sizez/nz

    #has the different x, y, and z values that are possible
    x0=np.arange(xmin,xmax,dx)
    y0=np.arange(ymin,ymax,dy)
    z0=np.arange(zmin,zmax,dz)

    val_array = np.asarray(val_array)
    lx, rx, ly, ry, lz, rz = find_idx_above_below(data1,x,y,z)
    
    c000 = val_array[lz,ly,lx]
    c100 = val_array[lz,ly,rx]
    c110 = val_array[lz,ry,rx]
    c010 = val_array[lz,ry,lx]
    
    c001 = val_array[rz,ly,lx]
    c011 = val_array[rz,ry,lx]
    c111 = val_array[rz,ry,rx]
    c101 = val_array[rz,ly,rx]
    
    return c000 *((1-h)**3) + (c100+c010+c001)*((1-h)**2)*h + (c101+c011+c110)*(1-h)*(h**2) + c111*(h**3)
    

def eulerIntegrate(data1, x, y, z, ns, B, Bx, By, Bz):
    nz,ny,nx = data1.Bx.shape

    xmin=data1.conf['lower'][0]
    ymin=data1.conf['lower'][1]
    zmin=data1.conf['lower'][2]

    sizex=data1.conf['size'][0]
    sizey=data1.conf['size'][1]
    sizez=data1.conf['size'][2]

    xmax=xmin+sizex
    ymax=ymin+sizey
    zmax=zmin+sizez

    dx=sizex/nx
    dy=sizey/ny
    dz=sizez/nz

    #has the different x, y, and z values that are possible
    x0=np.arange(xmin,xmax,dx)
    y0=np.arange(ymin,ymax,dy)
    z0=np.arange(zmin,zmax,dz)

    for i in range(0,ns):
        B0= interpolate(data1,dx,B,x[i],y[i],z[i])
        
        x.append(x[i] + (interpolate(data1,dx,Bx,x[i],y[i],z[i])*dx)/B0)
        
        y.append(y[i] + (interpolate(data1,dy,By,x[i],y[i],z[i])*dy)/B0)
        
        z.append(z[i] + (interpolate(data1,dz,Bz,x[i],y[i],z[i])*dz)/B0)
            
        if(x[i+1]<=xmin or x[i+1]>=xmax or y[i+1]<=ymin or y[i+1]>=ymax or z[i+1]<=zmin or z[i+1]>=zmax):
            x.pop()
            y.pop()
            z.pop()
            break
    vertices=np.transpose(np.array([x,y,z]))
    return vertices

def dataObjectToDict(data1):
    """
    Convert a data object to a dictionary format.
    
    Parameters:
    - data1: Data object containing magnetic field information.
    
    Returns:
    - data_dict: Dictionary representation of the data object.
    """
    data_dict = {
        'Bx': data1.Bx,
        'By': data1.By,
        'Bz': data1.Bz,
        'Ex': data1.Ex,
        'Ey': data1.Ey,
        'Ez': data1.Ez,
        'conf': data1.conf
    }
    return data_dict