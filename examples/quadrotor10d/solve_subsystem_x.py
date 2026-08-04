"""
Solve the backward reachable tube for Subsystem X of the decomposed 10D
near-hover quadrotor: state (px, vx, theta, q), control u_theta.

    dot(px)    = vx
    dot(vx)    = g * theta
    dot(theta) = q
    dot(q)     = u_theta

Run with: python3 solve_subsystem_x.py
"""
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
    U_THETA_MAX, LOOKBACK_LENGTH, T_STEP, RESULTS_DIR,
)


def solve():
    g = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])

    # Target set: rectangle around hover, |px|<=r1, |vx|<=r2, |theta|<=r3, |q|<=r4
    target = ShapeRectangle(g, TARGET_MIN_X, TARGET_MAX_X)

    sys_x = QuadrotorHoverX4D(uMin=-U_THETA_MAX, uMax=U_THETA_MAX, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH + small_number, step=T_STEP)

    # Backward reachable tube: union of target-reaching states over the horizon
    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_x, g, target, tau, compMethod, saveAllTimeSteps=True)

    # NOTE: HJSolver's saveAllTimeSteps output is indexed backwards in time:
    # index 0 is the fully-propagated (final) result, index -1 is the
    # untouched t=0 initial target (see dubins_4d_avoid.py's
    # `last_time_step_result = result[..., 0]` for the same convention).
    Vx = result[..., 0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "Vx.npy"), Vx)
    print("Saved subsystem X value function:", Vx.shape,
          "->", os.path.join(RESULTS_DIR, "Vx.npy"))
    return Vx


if __name__ == "__main__":
    solve()
