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

    #has the different x, y, and z values that are possible
    x0=np.arange(xmin,xmax,dx)
    y0=np.arange(ymin,ymax,dy)
    z0=np.arange(zmin,zmax,dz)

    nearx0 = find_nearest_index(x0,x)
    neary0 = find_nearest_index(y0,y)
    nearz0 = find_nearest_index(z0,z)
    #need to include cases where at the bounds 
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

def plot_single3D(data1, seed_points,B, Bx, By, Bz, x0, y0, z0, ns, output_html_name):
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

    plot=k3d.plot(grid=(xmin,ymin,zmin,xmax,ymax,zmax))
    si=0
    sf=100
    ns = int((sf-si)/0.05)
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
            vertices.append(eulerIntegrate(x,y,z,ns,B,Bx,By,Bz))
            allLines[i] = k3d.line(vertices[i],shader='mesh',width=0.05,opacity=0.5)
            i=i+1
    
    plot+=allLines.sum()
    plot.snapshot_type = 'online'
    plot.display()
    data = plot.get_snapshot()
    with open(str(output_html_name), 'w') as f:
        f.write(data)

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
            vertices.append(eulerIntegrate(x,y,z,ns,init_time[0],init_time[1],init_time[2],init_time[3]))
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
            vertices.append(eulerIntegrate(x,y,z,ns,snap[0],snap[1],snap[2],snap[3]))
            allLinePics[i][t-1]=vertices[i]
            i=i+1


    for vertIndex in range(0,seed_points.shape[0]):
        allLines[vertIndex].vertices = allLinePics[vertIndex]

    data = plot.get_snapshot()

    with open(str(output_html_name), 'w') as f:
        f.write(data)

if __name__ == '__main__':
    sys.path.append('/faculty/yyuan/codes/CoffeeGPU/python')
    from datalib import Data
    #check what format the data file is so I can know what to tell people to put in
    #generally though I will have the first argument be the data folder
    input_folder = str(sys.argv[1])
    #the second argument will be whatever they want the output html file to be called
    output_html_name = str(sys.argv[2])

    data = Data(input_folder)

    #ns= number of steps you want to take???

