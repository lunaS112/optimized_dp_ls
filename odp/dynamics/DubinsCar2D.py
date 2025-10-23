import heterocl as hcl
import numpy as np

class DubinsCar2D:
    """
    Simplified 2D Dubins Car model for HJ reachability.

    State: [x, y]
    Control: [u_x, u_y]  (bounded velocity)
    Disturbance: [d_x, d_y]  (bounded uncertainty)

    dynamics:
    x_dot = u_x + d_x
    y_dot = u_y + d_y
    """

    def __init__(self, uMin=[-1.0, -1.0], uMax=[1.0, 1.0],
                 dMin=[-0.1, -0.1], dMax=[0.1, 0.1],
                 uMode="min", dMode="max"):
        self.uMin = uMin
        self.uMax = uMax
        self.dMin = dMin
        self.dMax = dMax
        assert uMode in ["min", "max"]
        self.uMode = uMode
        # dMode must be opposite of uMode
        self.dMode = "max" if uMode == "min" else "min"
        self.dims = 2  # [x, y]

    # ------------------------------------------------------
    # Optimal Control
    # ------------------------------------------------------
    def opt_ctrl(self, t, state, spat_deriv):
        """
        Compute optimal control minimizing (or maximizing) Hamiltonian.
        spat_deriv: [∂V/∂x, ∂V/∂y]
        """
        u1 = hcl.scalar(0, "u1")
        u2 = hcl.scalar(0, "u2")

        # For each dimension, choose control sign based on spatial derivative
        for i in range(2):
            if i == 0:
                sd = spat_deriv[0]
                if self.uMode == "min":
                    with hcl.if_(sd > 0):
                        u1[0] = self.uMin[0]
                    with hcl.elif_(sd < 0):
                        u1[0] = self.uMax[0]
                else:  # uMode == "max"
                    with hcl.if_(sd > 0):
                        u1[0] = self.uMax[0]
                    with hcl.elif_(sd < 0):
                        u1[0] = self.uMin[0]
            else:
                sd = spat_deriv[1]
                if self.uMode == "min":
                    with hcl.if_(sd > 0):
                        u2[0] = self.uMin[1]
                    with hcl.elif_(sd < 0):
                        u2[0] = self.uMax[1]
                else:
                    with hcl.if_(sd > 0):
                        u2[0] = self.uMax[1]
                    with hcl.elif_(sd < 0):
                        u2[0] = self.uMin[1]

        return (u1[0], u2[0])

    # ------------------------------------------------------
    # Optimal Disturbance
    # ------------------------------------------------------
    def opt_dstb(self, t, state, spat_deriv):
        """
        Compute optimal disturbance maximizing (or minimizing) Hamiltonian.
        """
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")

        for i in range(2):
            if i == 0:
                sd = spat_deriv[0]
                if self.dMode == "max":
                    with hcl.if_(sd > 0):
                        d1[0] = self.dMax[0]
                    with hcl.elif_(sd < 0):
                        d1[0] = self.dMin[0]
                else:
                    with hcl.if_(sd > 0):
                        d1[0] = self.dMin[0]
                    with hcl.elif_(sd < 0):
                        d1[0] = self.dMax[0]
            else:
                sd = spat_deriv[1]
                if self.dMode == "max":
                    with hcl.if_(sd > 0):
                        d2[0] = self.dMax[1]
                    with hcl.elif_(sd < 0):
                        d2[0] = self.dMin[1]
                else:
                    with hcl.if_(sd > 0):
                        d2[0] = self.dMin[1]
                    with hcl.elif_(sd < 0):
                        d2[0] = self.dMax[1]

        return (d1[0], d2[0])

    # ------------------------------------------------------
    # Continuous Dynamics
    # ------------------------------------------------------
    def dynamics(self, t, state, uOpt, dOpt):
        """
        Compute state derivatives given control and disturbance.
        f(x,u,d) = u + d
        """
        dx = uOpt[0] + dOpt[0]
        dy = uOpt[1] + dOpt[1]
        return (dx, dy)

    # ------------------------------------------------------
    # Python (non-HCL) versions for quick testing
    # ------------------------------------------------------
    def dynamics_inPython(self, state, control, disturbance):
        return np.array([
            control[0] + disturbance[0],
            control[1] + disturbance[1]
        ])
