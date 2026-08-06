"""
Shared configuration for the decomposed 10D near-hover quadrotor benchmark.

The 10D state
    x = (px, vx, theta, q,   py, vy, phi, p,   pz, vz)
is decomposed into 3 independent subsystems:
    Subsystem X: (px, vx, theta, q)  -- 4D, control u_theta
    Subsystem Y: (py, vy, phi, p)    -- 4D, control u_phi
    Subsystem Z: (pz, vz)            -- 2D, control u_z

Units (SI throughout -- NOT cm): position in meters (m), velocity in m/s,
angle in radians (rad), angular rate in rad/s, control u_theta/u_phi in
rad/s^2, control u_z in m/s^2. This follows from g=9.81 (m/s^2) appearing
directly in dot(vx)=g*theta -- position must be in meters for that to be
dimensionally consistent.

This module is imported by both the 3 solve_subsystem_*.py scripts and by
reconstruct.py, so that the grids used when solving exactly match the grids
used when interpolating/reconstructing later.

Full derivation, reasoning, and the source paper this is based on:
see README.md in this directory.
"""
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---- Control configurations ----
# Two control-set configurations for the same 10D dynamics / decomposition:
#
#   independent_control: Cartesian-product bounds |ux|<=ux_max, |uy|<=uy_max,
#       |uz|<=uz_max. Subsystems share no control, so decomposition is EXACT.
#
#   shared_l2_control: the coupled constraint ux^2+uy^2+uz^2 <= U_MAX^2.
#       Each subsystem is solved with its *projected* 1D interval
#       [-U_MAX, U_MAX] (the exact projection of the L2 ball onto any single
#       axis). Combinations like (U_MAX,U_MAX,U_MAX) satisfy the Cartesian
#       product of projected intervals but violate the true L2 ball -- this
#       is the "leaking-corner" approximation. See reconstruct.py: the
#       decomposed reconstruction for this configuration is an
#       APPROXIMATION, not the exact value function of the coupled system.
#
# U_MAX is a free modeling choice for the coupled ball's radius (it must be a
# single scalar shared by all 3 axes, unlike the independent per-axis bounds
# below); 2.0 is chosen to match the tighter pitch/roll bounds.
U_THETA_MAX = 2.0   # u_theta in [-2, 2]
U_PHI_MAX = 2.0      # u_phi   in [-2, 2]
U_Z_MAX = 3.0        # u_z     in [-3, 3]

U_MAX = 2.0          # shared_l2_control: ux^2 + uy^2 + uz^2 <= U_MAX^2

CONTROL_CONFIGS = {
    "independent_control": {
        "ux_max": U_THETA_MAX,
        "uy_max": U_PHI_MAX,
        "uz_max": U_Z_MAX,
        "exact": True,
    },
    "shared_l2_control": {
        "ux_max": U_MAX,
        "uy_max": U_MAX,
        "uz_max": U_MAX,
        "exact": False,
    },
}


def get_control_bounds(config_name):
    if config_name not in CONTROL_CONFIGS:
        raise ValueError(
            f"Unknown control config '{config_name}', expected one of "
            f"{list(CONTROL_CONFIGS)}"
        )
    return CONTROL_CONFIGS[config_name]


def results_dir_for(config_name):
    """Each control configuration gets its own results/<config_name>/ dir
    so independent_control and shared_l2_control runs never clobber each
    other."""
    return os.path.join(RESULTS_DIR, config_name)

# ---- Subsystem X grid: (px, vx, theta, q) ----
# Units: px in m, vx in m/s, theta in rad, q in rad/s.
# theta is bounded to +-0.3 rad (~17 deg): the task's dynamics use the
# small-angle approximation g*theta (rather than the g*tan(theta) used in
# the original near-hover quadrotor model this benchmark is based on --
# Mo Chen, Sylvia L. Herbert, Claire J. Tomlin, "Decomposition of Reachable
# Sets and Tubes for a Class of Nonlinear Systems", IEEE Transactions on
# Automatic Control, 2018 (arXiv:1611.00122), Section VII-B, Eq. 61-64 --
# which is only accurate for small angles (~3% error at 0.3 rad). Letting
# theta range to +-1 rad (57 deg) would take the state well outside where
# that linearization is valid.
# px,py=+-8.0 confirmed NOT touching (convergence test showed both edges
# clear, X/Y symmetric) -- kept at +-8.0.
# vx,vy settled at +-2.5 m/s: still touches the domain edge in the strict
# "expand until nothing touches" sense (that would require going well past
# realistic near-hover speeds), but +-2.5 m/s is the outer edge of the
# realistic near-hover/low-speed flight range itself (~0.2-2.5 m/s tested
# range in the literature) -- beyond that the state isn't "near-hover"
# anymore regardless of recoverability, and this linearized/drag-free model
# isn't trustworthy there either. So the domain is sized to match the
# physically meaningful range, not to eliminate boundary-touching entirely.
GRID_MIN_X = np.array([-8.0, -2.5, -0.3, -2.0])
GRID_MAX_X = np.array([8.0, 2.5, 0.3, 2.0])
GRID_N_X = np.array([41, 41, 41, 41])

# Target: axis-aligned box around hover (L-infinity/Chebyshev box, built via
# ShapeRectangle -- NOT a ball), same +-0.2 half-width in all 4 components:
# |px|<=0.2 m, |vx|<=0.2 m/s, |theta|<=0.2 rad (~11.5 deg), |q|<=0.2 rad/s.
TARGET_MIN_X = np.array([-0.2, -0.2, -0.2, -0.2])
TARGET_MAX_X = np.array([0.2, 0.2, 0.2, 0.2])

# ---- Subsystem Y grid: (py, vy, phi, p) ----
# Units: py in m, vy in m/s, phi in rad, p in rad/s. Same box shape as X.
GRID_MIN_Y = np.array([-8.0, -2.5, -0.3, -2.0])
GRID_MAX_Y = np.array([8.0, 2.5, 0.3, 2.0])
GRID_N_Y = np.array([41, 41, 41, 41])

TARGET_MIN_Y = np.array([-0.2, -0.2, -0.2, -0.2])
TARGET_MAX_Y = np.array([0.2, 0.2, 0.2, 0.2])

# ---- Subsystem Z grid: (pz, vz) ----
# Units: pz in m, vz in m/s. No small-angle assumption involved here.
# pz widened to +-8.0 (from +-2.0), matching the same fix applied to px,py:
# at pz=+-2.0 the reachable set touched all 4 boundary faces and was ~93%
# saturated -- the same "domain too small, not a real saturation" issue X/Y
# had before their own widening. vz stays at the same +-2.5 m/s realistic
# near-hover ceiling as vx,vy (see GRID_MIN_X's comment).
GRID_MIN_Z = np.array([-8.0, -2.5])
GRID_MAX_Z = np.array([8.0, 2.5])
GRID_N_Z = np.array([81, 81])

# Target box: |pz|<=0.2 m, |vz|<=0.2 m/s.
TARGET_MIN_Z = np.array([-0.2, -0.2])
TARGET_MAX_Z = np.array([0.2, 0.2])

# ---- Time horizon: MUST be shared across all 3 subsystems ----
# The reconstruction V(x) = max(Vx,Vy,Vz) is only meaningful as the joint
# system's BRT at a single, common horizon T if every subsystem was solved
# over that same T. Giving Z a shorter horizon than X/Y (an earlier version
# of this file did, to make Z's plot look less saturated) breaks that: it
# stops corresponding to the joint BRT at any single T, and stops being
# comparable to a DeepReach model trained with one tMax for the whole 10D
# system. Do not split this again without also re-deriving what "the BRT"
# even means in that case.
#
# 3.0s: at the domain edge (2m) with vx bounded to 2 m/s, pure constant-
# velocity travel to the target takes ~0.9s before even accounting for the
# pitch/roll attitude lag needed to produce acceleration (g*theta) or
# decelerate into the target's |v|<=0.2 tolerance -- 1.0s was too short for
# the X/Y chains (4th-order/underactuated) to converge (~38% domain coverage
# at 3.0s vs. barely-expanded at 1.0s). Z (a direct 2nd-order integrator
# with strong control authority relative to its small domain) reaches 100%
# domain coverage under both independent_control and shared_l2_control at
# T=3.0s -- fully saturated, no visible BRT boundary in a Vz plot, and no
# visible difference between the two control configs in that plot. This is
# a real property of the system at this horizon (Z is simply not the
# limiting subsystem at T=3), not a bug -- see README.md.
LOOKBACK_LENGTH_XY = 3.0
LOOKBACK_LENGTH_Z = 3.0
T_STEP = 0.05

# 10D state layout, used by reconstruct.py to project a full state onto
# each subsystem.
IDX_X = [0, 1, 2, 3]
IDX_Y = [4, 5, 6, 7]
IDX_Z = [8, 9]
