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
import visualizeTools.interp_n_integrate as integ


def plot_single3D(data1,dataInfo, seed_points, B, Bx, By, Bz, x0, y0, z0, output_html_name, ns=2000):
    
    nx,ny,nz = dataInfo.conf['grid_shape']
    xmin=dataInfo.conf['lower'][0]
    ymin=dataInfo.conf['lower'][1]
    zmin=dataInfo.conf['lower'][2]

    sizex=dataInfo.conf['size'][0]
    sizey=dataInfo.conf['size'][1]
    sizez=dataInfo.conf['size'][2]

    xmax=xmin+sizex
    ymax=ymin+sizey
    zmax=zmin+sizez

    plot=k3d.plot(grid=(xmin,ymin,zmin,xmax,ymax,zmax))
    
    vertices = []

    allLines = np.empty([seed_points.shape[0]],dtype=object)
    

    i=0
    for vert in seed_points:
            x=[]
            y=[]
            z=[]
            
            x.append(vert[0])
            y.append(vert[1])
            z.append(vert[2])
            vertices.append(integ.eulerIntegrate(x,y,z,ns,B,Bx,By,Bz))
            allLines[i] = k3d.line(vertices[i],shader='mesh',width=0.05,opacity=0.5)
            i=i+1
    
    plot+=allLines.sum()
    plot.snapshot_type = 'online'
    plot.display()
    data = plot.get_snapshot()
    with open(str(output_html_name), 'w') as f:
        f.write(data)