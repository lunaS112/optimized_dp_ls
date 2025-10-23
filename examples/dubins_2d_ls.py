"""
2D Dubins-like car (point-mass) reachability example.
- Single controlled vehicle moving in 2D.
- Control: [u_x, u_y] bounded by uMin/uMax
- Disturbance: [d_x, d_y]
- Goal: reach (or avoid) a circular target region.
"""

import numpy as np
import math
import heterocl as hcl

from odp.Grid import Grid
from odp.Shapes import ShapeEllipsoid
from odp.solver import HJSolver
from odp.Plots import PlotOptions, plot_isosurface
from odp.dynamics.DubinsCar2D import DubinsCar2D  # your new 2D class

# ---------------------------------------------------------------------
# 1️⃣ Grid definition
# ---------------------------------------------------------------------
grid_resolution = np.array([81, 81])
g = Grid(
    np.array([-4.0, -4.0]),     # min values [x, y]
    np.array([4.0, 4.0]),       # max values [x, y]
    2,                          # number of dimensions
    grid_resolution,            # grid resolution
    []                          # no periodic dimensions
)

# ---------------------------------------------------------------------
# 2️⃣ Define the target set (circle centered at origin)
# ---------------------------------------------------------------------
goal_radius = 1.0
Initial_value_f = ShapeEllipsoid(g, np.array([0.0, 0.0]), np.array([goal_radius, goal_radius]))

# ---------------------------------------------------------------------
# 3️⃣ Define system dynamics
# ---------------------------------------------------------------------
sys = DubinsCar2D(
    uMin=[-1.0, -1.0],
    uMax=[1.0, 1.0],
    dMin=[-0.1, -0.1],
    dMax=[0.1, 0.1],
    uMode="min",   # minimize value function → move toward target
    dMode="max"    # disturbance tries to push away
)

# ---------------------------------------------------------------------
# 4️⃣ Time parameters
# ---------------------------------------------------------------------
lookback_length = 2.0
t_step = 0.05
tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)

# ---------------------------------------------------------------------
# 5️⃣ Solver options
# ---------------------------------------------------------------------
compMethod = {
    "TargetSetMode": "minVWithV0"  # compute backward reachable tube
}

po = PlotOptions(
    do_plot=True,
    plot_type="set",
    plotDims=[0, 1],
    slicesCut=[],
    min_isosurface=-0.1,
    max_isosurface=0.1,
    colorscale="Viridis",
    opacity=0.9,
    surface_count=1,
    save_fig=True,
    filename="dubins_2d_result.png"
)


# ---------------------------------------------------------------------
# 6️⃣ Run solver
# ---------------------------------------------------------------------
result = HJSolver(sys, g, Initial_value_f, tau, compMethod, saveAllTimeSteps=True)

print("✅ HJ Reachability (2D) complete!")
print("Result shape:", result.shape)

# ---------------------------------------------------------------------
# 7️⃣ Plot the final reachable set
# ---------------------------------------------------------------------
last_time_step = result[..., 0]
# plot_isosurface(g, last_time_step, po)
plot_isosurface(g, result, po)

np.save("dubins_2d_value_function.npy", last_time_step)

