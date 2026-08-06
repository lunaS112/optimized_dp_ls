"""
Solve the backward reachable tube for Subsystem Z of the decomposed 10D
near-hover quadrotor: state (pz, vz), control u_z.

    dot(pz) = vz
    dot(vz) = u_z

Two control configurations are supported (see quad_config.py):
    independent_control : |u_z| <= uz_max                    (exact)
    shared_l2_control    : projected interval |u_z| <= U_MAX
                            (u_z is one component of the coupled
                            ux^2+uy^2+uz^2 <= U_MAX^2 constraint --
                            decomposed reconstruction is an approximation)

Run with: python3 solve_subsystem_z.py [--config independent_control|shared_l2_control]
"""
import argparse
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
    LOOKBACK_LENGTH_Z, T_STEP,
    get_control_bounds, results_dir_for,
)


def solve(config_name="independent_control"):
    bounds = get_control_bounds(config_name)
    uz_max = bounds["uz_max"]

    g = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

    # Target set: rectangle around hover, |pz|<=r1, |vz|<=r2
    target = ShapeRectangle(g, TARGET_MIN_Z, TARGET_MAX_Z)

    sys_z = QuadrotorHoverZ2D(uMin=-uz_max, uMax=uz_max, uMode="min")

    small_number = 1e-5
    tau = np.arange(start=0, stop=LOOKBACK_LENGTH_Z + small_number, step=T_STEP)

    compMethod = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(sys_z, g, target, tau, compMethod, saveAllTimeSteps=True)

    Vz = result[..., 0]

    out_dir = results_dir_for(config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Vz.npy")
    np.save(out_path, Vz)
    print(f"[{config_name}] Saved subsystem Z value function:", Vz.shape,
          "-> ", out_path)
    return Vz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=["independent_control", "shared_l2_control"],
        default="independent_control",
    )
    args = parser.parse_args()
    solve(args.config)
