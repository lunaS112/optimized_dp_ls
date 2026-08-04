"""
Solve the backward reachable tube for Subsystem Z of the decomposed 10D
near-hover quadrotor: state (pz, vz), control u_z.

    dot(pz) = vz
    dot(vz) = u_z

Run with: python3 solve_subsystem_z.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import QuadrotorHoverZ2D
from odp.solver import HJSolver

from quad_config import (
    GRID_MIN_Z, GRID_MAX_Z, GRID_N_Z,
    TARGET_MIN_Z, TARGET_MAX_Z,
    U_Z_MAX, LOOKBACK_LENGTH, T_STEP, RESULTS_DIR,
)


def solve():
    g = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

    # Target set: rectangle around hover, |pz|<=r1, |vz|<=r2
    target = ShapeRectangle(g, TARGET_MIN_Z, TARGET_MAX_Z)

    sys_z = QuadrotorHoverZ2D(uMin=-U_Z_MAX, uMax=U_Z_MAX, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH + small_number, step=T_STEP)

    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_z, g, target, tau, compMethod, saveAllTimeSteps=True)

    # NOTE: HJSolver's saveAllTimeSteps output is indexed backwards in time:
    # index 0 is the fully-propagated (final) result, index -1 is the
    # untouched t=0 initial target (see dubins_4d_avoid.py's
    # `last_time_step_result = result[..., 0]` for the same convention).
    Vz = result[..., 0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, "Vz.npy"), Vz)
    print("Saved subsystem Z value function:", Vz.shape,
          "->", os.path.join(RESULTS_DIR, "Vz.npy"))
    return Vz


if __name__ == "__main__":
    solve()
