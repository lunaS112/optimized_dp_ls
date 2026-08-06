"""
Solve the backward reachable tube for Subsystem Y of the decomposed 10D
near-hover quadrotor: state (py, vy, phi, p), control u_phi.

    dot(py)  = vy
    dot(vy)  = -g * phi
    dot(phi) = p
    dot(p)   = u_phi

Two control configurations are supported (see quad_config.py):
    independent_control : |u_phi| <= uy_max                  (exact)
    shared_l2_control    : projected interval |u_phi| <= U_MAX
                            (u_phi is one component of the coupled
                            ux^2+uy^2+uz^2 <= U_MAX^2 constraint --
                            decomposed reconstruction is an approximation)

Run with: python3 solve_subsystem_y.py [--config independent_control|shared_l2_control]
"""
import argparse
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
    LOOKBACK_LENGTH_XY, T_STEP,
    get_control_bounds, results_dir_for,
)


def solve(config_name="independent_control"):
    bounds = get_control_bounds(config_name)
    uy_max = bounds["uy_max"]

    g = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])

    # Target set: rectangle around hover, |py|<=r1, |vy|<=r2, |phi|<=r3, |p|<=r4
    target = ShapeRectangle(g, TARGET_MIN_Y, TARGET_MAX_Y)

    sys_y = QuadrotorHoverY4D(uMin=-uy_max, uMax=uy_max, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH_XY + small_number, step=T_STEP)

    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_y, g, target, tau, compMethod, saveAllTimeSteps=True)

    Vy = result[..., 0]

    out_dir = results_dir_for(config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Vy.npy")
    np.save(out_path, Vy)
    print(f"[{config_name}] Saved subsystem Y value function:", Vy.shape,
          "-> ", out_path)
    return Vy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    solve(args.config)
