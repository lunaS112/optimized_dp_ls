import numpy as np

try:
    import heterocl as hcl
    HCL_AVAILABLE = True
except ImportError:
    hcl = None
    HCL_AVAILABLE = False

""" 10D NEAR-HOVER QUADROTOR - DECOMPOSED SUBSYSTEMS

Full (coupled-only-through-target) state:
    x = (px, vx, theta, q,   py, vy, phi, p,   pz, vz)

Linearized near-hover dynamics, decomposed into 3 independent subsystems
that share no state or control:

Subsystem X (pitch axis drives x-position), state (px, vx, theta, q):
    dot(px)    = vx
    dot(vx)    = g * theta
    dot(theta) = q
    dot(q)     = u_theta

Subsystem Y (roll axis drives y-position), state (py, vy, phi, p):
    dot(py)  = vy
    dot(vy)  = -g * phi
    dot(phi) = p
    dot(p)   = u_phi

Subsystem Z (vertical axis), state (pz, vz):
    dot(pz) = vz
    dot(vz) = u_z

There is no disturbance in this baseline (pure reachability under control).
"""

G = 9.81


class QuadrotorHoverX4D:
    """Subsystem 1: state (px, vx, theta, q), control u_theta."""

    def __init__(self, x=[0, 0, 0, 0], uMin=-2.0, uMax=2.0, uMode="min"):
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        assert uMode in ["min", "max"]
        self.uMode = uMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_u = hcl.scalar(self.uMax, "opt_u")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[3] > 0):
                opt_u[0] = self.uMin
        else:
            with hcl.if_(spat_deriv[3] < 0):
                opt_u[0] = self.uMin

        return (opt_u[0], in2[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        px_dot = hcl.scalar(0, "px_dot")
        vx_dot = hcl.scalar(0, "vx_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        q_dot = hcl.scalar(0, "q_dot")

        px_dot[0] = state[1]
        vx_dot[0] = G * state[2]
        theta_dot[0] = state[3]
        q_dot[0] = uOpt[0]

        return (px_dot[0], vx_dot[0], theta_dot[0], q_dot[0])


class QuadrotorHoverY4D:
    """Subsystem 2: state (py, vy, phi, p), control u_phi."""

    def __init__(self, x=[0, 0, 0, 0], uMin=-2.0, uMax=2.0, uMode="min"):
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        assert uMode in ["min", "max"]
        self.uMode = uMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_u = hcl.scalar(self.uMax, "opt_u")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[3] > 0):
                opt_u[0] = self.uMin
        else:
            with hcl.if_(spat_deriv[3] < 0):
                opt_u[0] = self.uMin

        return (opt_u[0], in2[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        py_dot = hcl.scalar(0, "py_dot")
        vy_dot = hcl.scalar(0, "vy_dot")
        phi_dot = hcl.scalar(0, "phi_dot")
        p_dot = hcl.scalar(0, "p_dot")

        py_dot[0] = state[1]
        vy_dot[0] = -G * state[2]
        phi_dot[0] = state[3]
        p_dot[0] = uOpt[0]

        return (py_dot[0], vy_dot[0], phi_dot[0], p_dot[0])


class QuadrotorHoverZ2D:
    """Subsystem 3: state (pz, vz), control u_z."""

    def __init__(self, x=[0, 0], uMin=-3.0, uMax=3.0, uMode="min"):
        self.x = x
        self.uMin = uMin
        self.uMax = uMax
        assert uMode in ["min", "max"]
        self.uMode = uMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_u = hcl.scalar(self.uMax, "opt_u")
        in2 = hcl.scalar(0, "in2")

        if self.uMode == "min":
            with hcl.if_(spat_deriv[1] > 0):
                opt_u[0] = self.uMin
        else:
            with hcl.if_(spat_deriv[1] < 0):
                opt_u[0] = self.uMin

        return (opt_u[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        pz_dot = hcl.scalar(0, "pz_dot")
        vz_dot = hcl.scalar(0, "vz_dot")

        pz_dot[0] = state[1]
        vz_dot[0] = uOpt[0]

        return (pz_dot[0], vz_dot[0])
