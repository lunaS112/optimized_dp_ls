"""
Solve the backward reachable tube for Subsystem X of the decomposed 10D
near-hover quadrotor: state (px, vx, theta, q), control u_theta.

    dot(px)    = vx
    dot(vx)    = g * theta
    dot(theta) = q
    dot(q)     = u_theta

Two control configurations are supported (see quad_config.py):
    independent_control : |u_theta| <= ux_max                 (exact)
    shared_l2_control    : projected interval |u_theta| <= U_MAX
                            (u_theta is one component of the coupled
                            ux^2+uy^2+uz^2 <= U_MAX^2 constraint --
                            decomposed reconstruction is an approximation)

Run with: python3 solve_subsystem_x.py [--config independent_control|shared_l2_control]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import QuadrotorHoverX4D
from odp.solver import HJSolver

from quad_config import (
    GRID_MIN_X, GRID_MAX_X, GRID_N_X,
    TARGET_MIN_X, TARGET_MAX_X,
    LOOKBACK_LENGTH_XY, T_STEP,
    get_control_bounds, results_dir_for,
)


def solve(config_name="independent_control"):
    bounds = get_control_bounds(config_name)
    ux_max = bounds["ux_max"]

    g = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])

    # Target set: rectangle around hover, |px|<=r1, |vx|<=r2, |theta|<=r3, |q|<=r4
    target = ShapeRectangle(g, TARGET_MIN_X, TARGET_MAX_X)

    sys_x = QuadrotorHoverX4D(uMin=-ux_max, uMax=ux_max, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH_XY + small_number, step=T_STEP)

    # Backward reachable tube: union of target-reaching states over the horizon
    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_x, g, target, tau, compMethod, saveAllTimeSteps=True)

    # NOTE: HJSolver's saveAllTimeSteps output is indexed backwards in time:
    # index 0 is the fully-propagated (final) result, index -1 is the
    # untouched t=0 initial target (see dubins_4d_avoid.py's
    # `last_time_step_result = result[..., 0]` for the same convention).
    Vx = result[..., 0]

    out_dir = results_dir_for(config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Vx.npy")
    np.save(out_path, Vx)
    print(f"[{config_name}] Saved subsystem X value function:", Vx.shape,
          "-> ", out_path)
    return Vx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    solve(args.config)
