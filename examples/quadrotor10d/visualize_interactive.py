"""
Interactive visualization using ODP's own built-in Plotly-based plotting
(odp.Plots.plot_isosurface / plot_valuefunction), instead of the static
matplotlib version in visualize.py.

This cluster's head node is headless (no browser to pop up), so instead of
do_plot=True we save each figure as a standalone interactive .html file
(save_fig=True, interactive_html=True) that you open locally -- rotate/zoom/
pan all work offline once the file is open, no server needed.

X and Y are 4D, so plot_type="set" with 3 of the 4 dims (px,vx,theta or
py,vy,phi) is shown as a rotatable 3D isosurface, with the 4th dim (q or p)
sliced at its nearest-to-zero grid index. Z is 2D, shown as an interactive
3D value-function surface.

Run with: python3 visualize_interactive.py [--config independent_control|shared_l2_control]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import plotly.graph_objects as go

from odp.Grid import Grid
from odp.Plots import PlotOptions, plot_isosurface, plot_valuefunction

# odp.Plots.plotting_utilities unconditionally also calls fig.write_image(...)
# after writing the interactive HTML (even when we only asked for HTML), which
# shells out to kaleido -> a headless Chromium to rasterize a static PNG. That
# browser process hangs indefinitely on this cluster (sandbox/display
# restrictions), with no timeout, blocking forever. We only want the
# interactive HTML, so neutralize write_image for this script only -- this
# does not modify odp/Plots/plotting_utilities.py itself.
go.Figure.write_image = lambda self, *args, **kwargs: None

from quad_config import (
    GRID_MIN_X, GRID_MAX_X, GRID_N_X,
    GRID_MIN_Y, GRID_MAX_Y, GRID_N_Y,
    GRID_MIN_Z, GRID_MAX_Z, GRID_N_Z,
    results_dir_for,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def nearest_index(grid, dim, value):
    return int(round((value - grid.min[dim]) / grid.dx[dim]))


def main(config_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results_dir = results_dir_for(config_name)

    gx = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])
    gy = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])
    gz = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

    Vx = np.load(os.path.join(results_dir, "Vx.npy"))
    Vy = np.load(os.path.join(results_dir, "Vy.npy"))
    Vz = np.load(os.path.join(results_dir, "Vz.npy"))

    q0 = nearest_index(gx, 3, 0.0)
    p0 = nearest_index(gy, 3, 0.0)

    # Subsystem X: 3D isosurface in (px, vx, theta), sliced at q=0
    po_x = PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[q0],
        colorscale="Bluered", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_X_isosurface"),
    )
    plot_isosurface(gx, Vx, po_x)

    # Subsystem Y: 3D isosurface in (py, vy, phi), sliced at p=0
    po_y = PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[p0],
        colorscale="Bluered", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_Y_isosurface"),
    )
    plot_isosurface(gy, Vy, po_y)

    # Subsystem Z: interactive 3D value-function surface (already 2D, no slicing)
    po_z = PlotOptions(
        do_plot=False, plot_type="value", plotDims=[0, 1],
        colorscale="RdBu", save_fig=True, interactive_html=True,
        filename=os.path.join(PLOTS_DIR, f"{config_name}_Z_value"),
    )
    plot_valuefunction(gz, Vz, po_z)

    print("Saved interactive HTML plots to:", PLOTS_DIR)
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.startswith(config_name) and f.endswith(".html"):
            print(" -", os.path.join(PLOTS_DIR, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    main(args.config)
