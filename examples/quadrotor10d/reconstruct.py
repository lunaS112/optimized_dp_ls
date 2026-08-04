"""
Reconstruction utility for the decomposed 10D near-hover quadrotor benchmark.

Given the 3 independently-solved subsystem value functions
    Vx(px, vx, theta, q)
    Vy(py, vy, phi, p)
    Vz(pz, vz)
this evaluates the implicit 10D value function

    V(x) = max(Vx(x1), Vy(x2), Vz(x3))

for arbitrary 10D states x = (px,vx,theta,q, py,vy,phi,p, pz,vz), WITHOUT ever
allocating a dense 10D array: each state is projected onto its 3 subsystems,
each (already-solved) subsystem value function is interpolated at that
projection using the existing Grid.get_values API, and the max of the three
results is returned. The reconstructed reachable set is {x : V(x) <= 0}.

Run with: python3 reconstruct.py   (after running the 3 solve_subsystem_*.py
scripts, which populate results/Vx.npy, Vy.npy, Vz.npy)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from odp.Grid import Grid

from quad_config import (
    GRID_MIN_X, GRID_MAX_X, GRID_N_X,
    GRID_MIN_Y, GRID_MAX_Y, GRID_N_Y,
    GRID_MIN_Z, GRID_MAX_Z, GRID_N_Z,
    IDX_X, IDX_Y, IDX_Z,
    RESULTS_DIR,
)


class Quadrotor10DReconstructedValue:
    """
    V(x) = max(Vx(x1), Vy(x2), Vz(x3)) for the decomposed 10D quadrotor,
    evaluated on demand via interpolation -- no dense 10D array is ever built.
    """

    def __init__(self, results_dir=RESULTS_DIR):
        self.grid_x = Grid(GRID_MIN_X, GRID_MAX_X, 4, GRID_N_X, [])
        self.grid_y = Grid(GRID_MIN_Y, GRID_MAX_Y, 4, GRID_N_Y, [])
        self.grid_z = Grid(GRID_MIN_Z, GRID_MAX_Z, 2, GRID_N_Z, [])

        self.Vx = np.load(os.path.join(results_dir, "Vx.npy"))
        self.Vy = np.load(os.path.join(results_dir, "Vy.npy"))
        self.Vz = np.load(os.path.join(results_dir, "Vz.npy"))

    @staticmethod
    def project(state_10d):
        """Splits full 10D state(s) into (x1, x2, x3) subsystem projections."""
        state_10d = np.asarray(state_10d, dtype=float)
        x1 = state_10d[..., IDX_X]
        x2 = state_10d[..., IDX_Y]
        x3 = state_10d[..., IDX_Z]
        return x1, x2, x3

    def value(self, state_10d):
        """
        Args:
            state_10d: shape (10,) or (N, 10), ordered
                       (px,vx,theta,q, py,vy,phi,p, pz,vz)

        Returns:
            V(x) = max(Vx(x1), Vy(x2), Vz(x3)) -- scalar or shape (N,)
        """
        x1, x2, x3 = self.project(state_10d)
        vx = self.grid_x.get_values(self.Vx, x1)
        vy = self.grid_y.get_values(self.Vy, x2)
        vz = self.grid_z.get_values(self.Vz, x3)
        return np.maximum(np.maximum(vx, vy), vz)

    def is_in_reachable_set(self, state_10d):
        """True where V(x) <= 0."""
        return self.value(state_10d) <= 0


if __name__ == "__main__":
    recon = Quadrotor10DReconstructedValue()

    hover = np.zeros(10)
    print("V(hover)              =", recon.value(hover),
          "| in set:", recon.is_in_reachable_set(hover))

    # Far outside the domain in every dimension: not recoverable in the horizon
    # (theta/phi clipped to the +-0.3 rad domain boundary, see quad_config.py)
    far_corner = np.array([1.9, 1.9, 0.29, 1.9, 1.9, 1.9, 0.29, 1.9, 1.9, 1.9])
    print("V(far corner)         =", recon.value(far_corner),
          "| in set:", recon.is_in_reachable_set(far_corner))

    # pz=0.5 is outside the raw +-0.2 target box, but z is a direct double
    # integrator with strong control authority (u_z in [-3,3]), so the BRT
    # should have expanded to cover it within the time horizon.
    pz_out = np.zeros(10)
    pz_out[8] = 0.5
    print("V(pz=0.5, rest hover) =", recon.value(pz_out),
          "| in set:", recon.is_in_reachable_set(pz_out))

    # px=0.5 is the same offset, but the x-chain is 4th order/underactuated
    # (px <- vx <- theta <- q <- u_theta), so it should NOT have expanded as
    # far in the same horizon.
    px_out = np.zeros(10)
    px_out[0] = 0.5
    print("V(px=0.5, rest hover) =", recon.value(px_out),
          "| in set:", recon.is_in_reachable_set(px_out))

    batch = np.stack([hover, far_corner, pz_out, px_out])
    print("V(batch)              =", recon.value(batch))
