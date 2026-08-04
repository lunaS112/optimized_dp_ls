"""
Solve the backward reachable tube for Subsystem Y of the decomposed 10D
near-hover quadrotor: state (py, vy, phi, p), control u_phi.

    dot(py)  = vy
    dot(vy)  = -g * phi
    dot(phi) = p
    dot(p)   = u_phi

Run with: python3 solve_subsystem_y.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import QuadrotorHoverY4D
from odp.solver import HJSolver

from quad_config import (
    GRID_MIN_Y, GRID_MAX_Y, GRID_N_Y,
    TARGET_MIN_Y, TARGET_MAX_Y,
    U_PHI_MAX, LOOKBACK_LENGTH, T_STEP, RESULTS_DIR,
)


def solve():
    g = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])

    # Target set: rectangle around hover, |py|<=r1, |vy|<=r2, |phi|<=r3, |p|<=r4
    target = ShapeRectangle(g, TARGET_MIN_Y, TARGET_MAX_Y)

    sys_y = QuadrotorHoverY4D(uMin=-U_PHI_MAX, uMax=U_PHI_MAX, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH + small_number, step=T_STEP)

    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_y, g, target, tau, compMethod, saveAllTimeSteps=True)

    # NOTE: HJSolver's saveAllTimeSteps output is indexed backwards in time:
    # index 0 is the fully-propagated (final) result, index -1 is the
    # untouched t=0 initial target (see dubins_4d_avoid.py's
    # `last_time_step_result = result[..., 0]` for the same convention).
    Vy = result[..., 0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "Vy.npy"), Vy)
    print("Saved subsystem Y value function:", Vy.shape,
          "->", os.path.join(RESULTS_DIR, "Vy.npy"))
    return Vy


if __name__ == "__main__":
    solve()
