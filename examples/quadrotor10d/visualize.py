"""
Quick static visualization of the decomposed 10D near-hover quadrotor
reachable sets. Not part of the required deliverables -- just a convenience
script to eyeball results after solving.

X and Y are 4D, so they're shown as a 2D slice at (theta,q)=(0,0) and
(phi,p)=(0,0) respectively (i.e. "if attitude is already level, what
positions/velocities can recover to hover?"). Z is plotted directly since
it's already 2D.

Run with: python3 visualize.py [--config independent_control|shared_l2_control]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from odp.Grid import Grid

from quad_config import (
    GRID_MIN_X, GRID_MAX_X, GRID_N_X, TARGET_MIN_X, TARGET_MAX_X,
    GRID_MIN_Y, GRID_MAX_Y, GRID_N_Y, TARGET_MIN_Y, TARGET_MAX_Y,
    GRID_MIN_Z, GRID_MAX_Z, GRID_N_Z, TARGET_MIN_Z, TARGET_MAX_Z,
    results_dir_for,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def nearest_index(grid, dim, value):
    return int(round((value - grid.min[dim]) / grid.dx[dim]))


def draw_target_box(ax, tmin, tmax):
    rect = plt.Rectangle(
        (tmin[0], tmin[1]), tmax[0] - tmin[0], tmax[1] - tmin[1],
        fill=False, edgecolor="lime", linewidth=2, linestyle="--",
        label="target",
    )
    ax.add_patch(rect)


def main(config_name):
    results_dir = results_dir_for(config_name)

    gx = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])
    gy = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])
    gz = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

    Vx = np.load(os.path.join(results_dir, "Vx.npy"))
    Vy = np.load(os.path.join(results_dir, "Vy.npy"))
    Vz = np.load(os.path.join(results_dir, "Vz.npy"))

    # Slice at theta=0,q=0 saturates fully (100% covered) at the 3.0s X/Y
    # horizon -- starting already level/no residual rate makes position+
    # velocity recovery trivial within 3s. theta=0.3,q=0.5 (near the domain
    # edge, some residual rate) shows a genuine partial boundary instead.
    theta0 = nearest_index(gx, 2, 0.3)
    q0 = nearest_index(gx, 3, 0.5)
    Vx_slice = Vx[:, :, theta0, q0]  # (px, vx)

    phi0 = nearest_index(gy, 2, 0.3)
    p0 = nearest_index(gy, 3, 0.5)
    Vy_slice = Vy[:, :, phi0, p0]  # (py, vy)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, xs, ys, V, tmin, tmax, xlabel, ylabel, title in [
        (axes[0], gx.grid_points[0], gx.grid_points[1], Vx_slice,
         TARGET_MIN_X[:2], TARGET_MAX_X[:2], "px (m)", "vx (m/s)",
         "Subsystem X\nslice at theta=0.3, q=0.5"),
        (axes[1], gy.grid_points[0], gy.grid_points[1], Vy_slice,
         TARGET_MIN_Y[:2], TARGET_MAX_Y[:2], "py (m)", "vy (m/s)",
         "Subsystem Y\nslice at phi=0.3, p=0.5"),
        (axes[2], gz.grid_points[0], gz.grid_points[1], Vz,
         TARGET_MIN_Z, TARGET_MAX_Z, "pz (m)", "vz (m/s)",
         "Subsystem Z\n(full 2D, no slicing needed)"),
    ]:
        v_abs = max(np.max(np.abs(V)), 1e-6)
        cs = ax.contourf(xs, ys, V.T, levels=30, cmap="RdBu", vmin=-v_abs, vmax=v_abs)
        ax.contour(xs, ys, V.T, levels=[0], colors="black", linewidths=2)
        draw_target_box(ax, tmin, tmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(cs, ax=ax, shrink=0.85, label="V (black contour = V=0 boundary)")

    fig.suptitle(
        f"Decomposed 10D near-hover quadrotor -- config: {config_name}\n"
        f"(black contour = reachable-set boundary V=0, dashed green = target box)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{config_name}.png")
    fig.savefig(out_path, dpi=130)
    print("Saved:", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    main(args.config)
