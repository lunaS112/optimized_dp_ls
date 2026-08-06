"""
Animated version of visualize_interactive.py -- includes the time slider
that the plain version doesn't have.

visualize_interactive.py loads the saved results/<config>/V{x,y,z}.npy files,
which are deliberately just the final (fully-propagated) BRT -- a single
static array with no time axis, since that's what reconstruct.py needs for
V=max(Vx,Vy,Vz). Feeding a static (no-time-axis) array to odp.Plots is
exactly why plot_isosurface/plot_valuefunction did NOT add a slider: the
slider/play-button only appears when the array passed in has an extra
trailing time dimension (see plot_isosurface's "with animation" branches in
odp/Plots/plotting_utilities.py).

To get the slider, this script re-solves each subsystem (same dynamics,
grid, target as solve_subsystem_*.py) but keeps the FULL saveAllTimeSteps=True
result (all time steps, not just index 0) and passes that to the plotting
functions -- so you can scrub through how the BRS grows into the BRT.
This does NOT change results/*.npy used by reconstruct.py.

Run with: python3 visualize_interactive_animated.py [--config independent_control|shared_l2_control]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import plotly.graph_objects as go

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import QuadrotorHoverX4D, QuadrotorHoverY4D, QuadrotorHoverZ2D
from odp.solver import HJSolver
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction

# Same kaleido/Chromium hang workaround as visualize_interactive.py -- see
# that file for the full explanation. Only affects this script's process.
go.Figure.write_image = lambda self, *args, **kwargs: None

from quad_config import (
    GRID_MIN_X, GRID_MAX_X, GRID_N_X, TARGET_MIN_X, TARGET_MAX_X,
    GRID_MIN_Y, GRID_MAX_Y, GRID_N_Y, TARGET_MIN_Y, TARGET_MAX_Y,
    GRID_MIN_Z, GRID_MAX_Z, GRID_N_Z, TARGET_MIN_Z, TARGET_MAX_Z,
    LOOKBACK_LENGTH_XY, LOOKBACK_LENGTH_Z, T_STEP, get_control_bounds,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def nearest_index(grid, dim, value):
    return int(round((value - grid.min[dim]) / grid.dx[dim]))


def solve_full(dyn_cls, grid, target_min, target_max, u_max, lookback_length):
    g = grid
    target = ShapeRectangle(g, target_min, target_max)
    sys_obj = dyn_cls(uMin=-u_max, uMax=u_max, uMode="min")
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=T_STEP)
    compMethod = {"TargetSetMode": "minVWithV0"}
    # Keep the FULL time history (all len(tau) slices) -- this is the only
    # difference from solve_subsystem_*.py, which discards all but index 0.
    return HJSolver(sys_obj, g, target, tau, compMethod, saveAllTimeSteps=True)


def main(config_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    bounds = get_control_bounds(config_name)

    gx = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])
    gy = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])
    gz = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

    print("Re-solving subsystem X (full time history)...")
    Vx_full = solve_full(QuadrotorHoverX4D, gx, TARGET_MIN_X, TARGET_MAX_X, bounds["ux_max"], LOOKBACK_LENGTH_XY)
    print("Re-solving subsystem Y (full time history)...")
    Vy_full = solve_full(QuadrotorHoverY4D, gy, TARGET_MIN_Y, TARGET_MAX_Y, bounds["uy_max"], LOOKBACK_LENGTH_XY)
    print("Re-solving subsystem Z (full time history)...")
    Vz_full = solve_full(QuadrotorHoverZ2D, gz, TARGET_MIN_Z, TARGET_MAX_Z, bounds["uz_max"], LOOKBACK_LENGTH_Z)

    q0 = nearest_index(gx, 3, 0.0)
    p0 = nearest_index(gy, 3, 0.0)

    po_x = PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[q0],
        colorscale="Bluered", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_X_isosurface_animated"),
    )
    plot_isosurface(gx, Vx_full, po_x)

    po_y = PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[p0],
        colorscale="Bluered", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_Y_isosurface_animated"),
    )
    plot_isosurface(gy, Vy_full, po_y)

    po_z = PlotOptions(
        do_plot=False, plot_type="value", plotDims=[0, 1],
        colorscale="RdBu", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_Z_value_animated"),
    )
    plot_valuefunction(gz, Vz_full, po_z)

    print("Saved animated interactive HTML plots to:", PLOTS_DIR)
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.startswith(config_name) and "animated" in f and f.endswith(".html"):
            print(" -", os.path.join(PLOTS_DIR, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    main(args.config)
