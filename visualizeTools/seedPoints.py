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

def axisymmetricSeedPoints(data1, stellar_radius, num_points=50):
    """
    Generate axisymmetric seed points for 3D visualization.
    
    Parameters:
    - data1: Data object containing magnetic field information.
    
    Returns:
    - vertices: List of vertices for the seed points after integration.
    """
    
    rr, theta, phi = np.meshgrid(stellar_radius * np.ones(num_points),
                             np.linspace(0,np.pi, num_points),
                             np.linspace(0, 2*np.pi, num_points))
    
    x = rr * np.sin(theta) * np.cos(phi)
    y = rr * np.sin(theta) * np.sin(phi)
    z = rr * np.cos(theta)
    
    
    return [x,y,z]