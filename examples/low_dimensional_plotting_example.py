
import importlib
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture
# Plot options
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math


'''
1. The State Space and Grid Setup
The system has three state variables:

x - Position along the x-axis.
y - Position along the y-axis.
θ - The car’s heading (orientation).

The state space spans from [−4,4]x, y and from −π to π in θ.
dims = 3 means we are working with a 3D system.
The grid resolution is 40×40×40.
pd = [2] means the θ imension is periodic, ensuring the heading wraps around at ±π.

2. The System Dynamics: DubinsCapture
The car follows Dubins Car dynamics as defined in DubinsCapture.py:

x_dot = -speed + speed*cos(theta) + w*y
y_dot = speed*sin(theta) - w*x
theta_dot = d - w

w is the control input (steering rate).
d is the disturbance input (external perturbation).
The car moves forward at a constant speed (speed=1.0).
The player (car) maximizes control, while the disturbance minimizes it, leading to a pursuit-evasion game.

3. The Target Set (Initial Value Function)

center = np.zeros(dims)
radius = 1.0
ignore_dims = [2]
Initial_value_f = CylinderShape(g, ignore_dims, center, radius)

This defines a circular target set in the (x, y) plane.
ignore_dims = [2] means the cylinder is independent of θ.
The radius is 1.0, meaning the target set includes all points within a unit circle centered at (0,0) in the xy-plane.
'''

# STEP 1: Define grid
grid_min = np.array([-4.0, -4.0, -math.pi])
grid_max = np.array([4.0, 4.0, math.pi])
dims = 3
N = np.array([40, 40, 40])
pd=[2]
g = Grid(grid_min, grid_max, dims, N, pd)

# STEP 2: Generate initial values for grid using shape functions
center = np.zeros(dims)
radius = 1.0
ignore_dims = [2]
Initial_value_f = CylinderShape(g, ignore_dims, center, radius)

# STEP 3: Time length for computations
lookback_length = 2.0
t_step = 0.05

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# STEP 4: User-defined System dynamics for computation
sys = DubinsCapture(uMode="max", dMode="min")

# While file needs to be saved locally, set save_fig=True and filename, recommend to set interactive_html=True for better interaction
po2 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2],
                  slicesCut=[], colorscale="Bluered", save_fig=True, filename="3D_0_sublevel_set.png", interactive_html=True)
                  
# STEP 5: Initialize plotting option
po1 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2])

# STEP 6: Call HJSolver function (BRS)
compMethod = { "TargetSetMode": "None"}
result_3 = HJSolver(sys, g, Initial_value_f, tau, compMethod, po1, saveAllTimeSteps=True)

# STEP 7: Visualizing output
plot_isosurface(g, result_3, po2)