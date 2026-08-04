"""
Shared configuration for the decomposed 10D near-hover quadrotor benchmark.

The 10D state
    x = (px, vx, theta, q,   py, vy, phi, p,   pz, vz)
is decomposed into 3 independent subsystems:
    Subsystem X: (px, vx, theta, q)  -- 4D, control u_theta
    Subsystem Y: (py, vy, phi, p)    -- 4D, control u_phi
    Subsystem Z: (pz, vz)            -- 2D, control u_z

This module is imported by both the 3 solve_subsystem_*.py scripts and by
reconstruct.py, so that the grids used when solving exactly match the grids
used when interpolating/reconstructing later.
"""
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---- Control bounds ----
U_THETA_MAX = 2.0   # u_theta in [-2, 2]
U_PHI_MAX = 2.0      # u_phi   in [-2, 2]
U_Z_MAX = 3.0        # u_z     in [-3, 3]

# ---- Subsystem X grid: (px, vx, theta, q) ----
# theta is bounded to +-0.3 rad (~17 deg): the task's dynamics use the
# small-angle approximation g*theta (rather than the g*tan(theta) used in
# the original near-hover quadrotor model this benchmark is based on --
# Chen, Herbert & Tomlin, "Decomposition of Reachable Sets and Tubes for a
# Class of Nonlinear Systems", arXiv:1611.00122), which is only accurate for
# small angles (~3% error at 0.3 rad). Letting theta range to +-1 rad (57
# deg) would take the state well outside where that linearization is valid.
GRID_MIN_X = np.array([-2.0, -2.0, -0.3, -2.0])
GRID_MAX_X = np.array([2.0, 2.0, 0.3, 2.0])
GRID_N_X = np.array([41, 41, 41, 41])

# Target around hover: |px|<=r1, |vx|<=r2, |theta|<=r3, |q|<=r4
TARGET_MIN_X = np.array([-0.2, -0.2, -0.2, -0.2])
TARGET_MAX_X = np.array([0.2, 0.2, 0.2, 0.2])

# ---- Subsystem Y grid: (py, vy, phi, p) ----
GRID_MIN_Y = np.array([-2.0, -2.0, -0.3, -2.0])
GRID_MAX_Y = np.array([2.0, 2.0, 0.3, 2.0])
GRID_N_Y = np.array([41, 41, 41, 41])

TARGET_MIN_Y = np.array([-0.2, -0.2, -0.2, -0.2])
TARGET_MAX_Y = np.array([0.2, 0.2, 0.2, 0.2])

# ---- Subsystem Z grid: (pz, vz) ----
# No small-angle assumption involved here, so the domain is left as-is;
# resolution bumped since 2D grids are cheap.
GRID_MIN_Z = np.array([-2.0, -2.0])
GRID_MAX_Z = np.array([2.0, 2.0])
GRID_N_Z = np.array([81, 81])

TARGET_MIN_Z = np.array([-0.2, -0.2])
TARGET_MAX_Z = np.array([0.2, 0.2])

# ---- Time horizon shared by all 3 subsystem BRT computations ----
LOOKBACK_LENGTH = 1.0
T_STEP = 0.05

# 10D state layout, used by reconstruct.py to project a full state onto
# each subsystem.
IDX_X = [0, 1, 2, 3]
IDX_Y = [4, 5, 6, 7]
IDX_Z = [8, 9]
