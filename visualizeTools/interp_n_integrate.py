import numpy as np
import h5py
import sys
import os
import toml
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import k3d
import time
import visualizeTools.utils as utils

#3D Linear Interpolation of the array value at a point (x,y,z)
def linInterp(dataInfo, h, val_array, x, y, z):
    val_array = np.asarray(val_array)
    lx, rx, ly, ry, lz, rz = utils.find_idx_above_below(dataInfo,x,y,z)
    
    c000 = val_array[lx,ly,lz]
    c100 = val_array[lx,ly,rz]
    c110 = val_array[lx,ry,rz]
    c010 = val_array[lx,ry,lz]
    
    c001 = val_array[rx,ly,lz]
    c011 = val_array[rx,ry,lz]
    c111 = val_array[rx,ry,rz]
    c101 = val_array[rx,ly,rz]
    
    return c000 *((1-h)**3) + (c100+c010+c001)*((1-h)**2)*h + (c101+c011+c110)*(1-h)*(h**2) + c111*(h**3)
    

def eulerIntegrate(data,dataInfo, x, y, z, ns, B, Bx, By, Bz):
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
    

    for i in range(0,ns):
        B0= linInterp(dataInfo,dx,data['B'],x[i],y[i],z[i])
        
        x.append(x[i] + (linInterp(dataInfo,dx,data['Bx'],x[i],y[i],z[i])*dx)/B0)
        
        y.append(y[i] + (linInterp(dataInfo,dy,data['By'],x[i],y[i],z[i])*dy)/B0)
        
        z.append(z[i] + (linInterp(dataInfo,dz,data['Bz'],x[i],y[i],z[i])*dz)/B0)
            
        if(x[i+1]<=xmin or x[i+1]>=xmax or y[i+1]<=ymin or y[i+1]>=ymax or z[i+1]<=zmin or z[i+1]>=zmax):
            x.pop()
            y.pop()
            z.pop()
            break
    vertices=np.transpose(np.array([x,y,z]))
    return vertices