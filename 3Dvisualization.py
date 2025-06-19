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
import visualizeTools.seedPoints as seedPoints


def plot_3Dmovie(data1, seed_points, B, Bx, By, Bz, x0, y0, z0, ns, timeStep, init_time,output_html_name,filename_setup):
    nx,ny,nz = data1.Bx.shape
    xmin=data1.conf['lower'][0]
    ymin=data1.conf['lower'][1]
    zmin=data1.conf['lower'][2]

    sizex=data1.conf['size'][0]
    sizey=data1.conf['size'][1]
    sizez=data1.conf['size'][2]

    xmax=xmin+sizex
    ymax=ymin+sizey
    zmax=zmin+sizez
    si=0
    sf=100
    ns = int((sf-si)/0.05)
    plot=k3d.plot(grid=(xmin,ymin,zmin,xmax,ymax,zmax))
    vertices = []
    allLines = np.empty([seed_points.shape[0]],dtype=object)
    allLinePics = np.empty([seed_points.shape[0]],dtype=dict)

    i=0
    for vert in seed_points:
            x=[]
            y=[]
            z=[]
            
            x.append(vert[0])
            y.append(vert[1])
            z.append(vert[2])
            vertices.append(utils.eulerIntegrate(x,y,z,ns,init_time[0],init_time[1],init_time[2],init_time[3]))
            allLines[i] = k3d.line(vertices[i],shader='mesh',color=0x028260,width=0.05,opacity=0.5)
            allLinePics[i]=dict({0:vertices[i]})
            i=i+1

    

    plot+=allLines.sum()
    plot.snapshot_type = 'online'
    plot.display()   

    for t in range(1,timeStep+1):
        print(t)
        snap = h5py.File(str(filename_setup)+"{:02d}".format(t)+".h5",'r')
        snap = np.asarray([np.sqrt(np.square(np.asarray(snap['Bx']))+np.square(np.asarray(snap['By']))+np.square(np.asarray(snap['Bz']))), np.asarray(snap['Bx']), np.asarray(snap['By']), np.asarray(snap['Bz']),np.sqrt(np.square(np.asarray(snap['Ex']))+np.square(np.asarray(snap['Ey']))+np.square(np.asarray(snap['Ez'])))])
        
        i=0
        for vert in seed_points:
            x=[]
            y=[]
            z=[]
            
            x.append(vert[0])
            y.append(vert[1])
            z.append(vert[2])
            vertices.append(utils.eulerIntegrate(x,y,z,ns,snap[0],snap[1],snap[2],snap[3]))
            allLinePics[i][t-1]=vertices[i]
            i=i+1


    for vertIndex in range(0,seed_points.shape[0]):
        allLines[vertIndex].vertices = allLinePics[vertIndex]

    data = plot.get_snapshot()

    with open(str(output_html_name), 'w') as f:
        f.write(data)

if __name__ == '__main__':
    sys.path.append('/faculty/yyuan/codes/CoffeeGPU/python')

    x, y, z = seedPoints.axisymmetricSeedPoints(data1, stellar_radius=1.0, num_points=50)
    
    from datalib import Data
    #check what format the data file is so I can know what to tell people to put in
    #generally though I will have the first argument be the data folder
    input_folder = str(sys.argv[1])
    #the second argument will be whatever they want the output html file to be called
    output_html_name = str(sys.argv[2])

    data = Data(input_folder)

    #ns= number of steps you want to take???

