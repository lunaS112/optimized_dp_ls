'''
Two Dubins cars: A controlled car (ego) and an adversary (obstacle).
2D space: Each car moves in (x, y) with a fixed speed and limited turning radius.
Goal: The ego car avoids colliding with the adversary.
State Representation
Since each car has (x, y, θ\thetaθ), the joint state space is:
(xr,yr,θr)

where:
xr=x2-x1
yr=y2-y1
θr=θ2-θ1
(x1,y1,θ1) is the state of the ego car.
(x2,y2,θ2) is the state of the adversary.
'''
from typing import List
import numpy as np
import math
from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.solver import HJSolver, computeSpatDerivArray
from odp.Plots import PlotOptions, plot_isosurface
import heterocl as hcl

grid_resolution = np.array([50, 50, 30])  

# Define the state space: (x_r, y_r, theta_r)
g = Grid(
    np.array([-4, -4, -math.pi]),  # min values
    np.array([4, 4, math.pi]),  # max values
    3, # number of dimensions
    grid_resolution,  # grid resolution
    [2]  # Periodicity in theta1 and theta2
)

# Define the initial value function
collision_radius = 1
Initial_value_f = CylinderShape(g, [2], np.zeros(3), collision_radius)


"""
Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
"TargetSetMode":
{
"none" -> compute Backward Reachable Set,
"minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
"maxVWithV0" -> max V with V0,
"maxVOverTime" -> compute max V over time,
"minVOverTime" -> compute min V over time,
"minVWithVTarget" -> min V with target set (if target set is different from initial V0)
"maxVWithVTarget" -> max V with target set (if target set is different from initial V0)
}

(optional)
Please specify this mode if you would like to add another target set, which can be an obstacle set
for solving a reach-avoid problem
"ObstacleSetMode":
{
"minVWithObstacle" -> min with obstacle set,
"maxVWithObstacle" -> max with obstacle set
}
"""
class DubinsTwoCar_3d:
    def __init__(self, v1=1.0, v2=1.0, wMax=1.0, uMode="max", dMode="min"):
        self.v1 = v1  # Fixed speed for both cars
        self.v2 = v2
        self.wMax = wMax  # Maximum steering rate
        self.uMode = uMode  # Control strategy
        self.dMode = dMode  # Disturbance strategy

    def dynamics(self, t, state, uOpt, dOpt):
        """
        Computes the dynamics in the relative coordinate system.
        :param state: (x_r, y_r, theta_r)
        :param uOpt: Control input for ego car (turn rate u1)
        :param dOpt: Control input for adversary (turn rate u2)
        :return: Time derivatives of the state variables
        """

        x_r_dot = hcl.scalar(0, "xr_dot")
        y_r_dot = hcl.scalar(0, "yr_dot")
        theta_r_dot = hcl.scalar(0, "thetar_dot")

        x_r_dot[0] = self.v1 * hcl.cos(state[2]) - self.v2  
        y_r_dot[0] = self.v1 * hcl.sin(state[2])  
        theta_r_dot[0] = uOpt[0] - dOpt[0]  

        return (x_r_dot[0], y_r_dot[0], theta_r_dot[0])

    def opt_ctrl(self, t, state, spat_deriv):
        """
        Computes optimal control (steering) for the ego car.
        """
        opt_w1 = hcl.scalar(self.wMax, "opt_w1")
        # Just create and pass back, even though they're not used
        in2   = hcl.scalar(0, "in2")
        in3   = hcl.scalar(0, "in3")
        
        with hcl.if_(self.uMode == "max"):
            with hcl.if_(spat_deriv[2] < 0): # If ∂V/∂θr < 0, turn +wMax
                opt_w1[0] = -self.wMax
        with hcl.elif_(self.uMode == "min"):
            with hcl.if_(spat_deriv[2] > 0): # If ∂V/∂θr > 0, turn -wMax
                opt_w1[0] = -self.wMax

        return (opt_w1[0], in2[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        Computes optimal disturbance (steering) for the adversary.
        """
        opt_w2 = hcl.scalar(self.wMax, "opt_w2")
        # Just create and pass back, even though they're not used
        in2   = hcl.scalar(0, "in2")
        in3   = hcl.scalar(0, "in3")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[2] < 0): # If ∂V/∂θr < 0, adversary turns -wMax
                opt_w2[0] = -self.wMax
        with hcl.elif_(self.dMode == "min"):
            with hcl.if_(spat_deriv[2] > 0): # If ∂V/∂θr > 0, adversary turns +wMax
                opt_w2[0] = -self.wMax

        return (opt_w2[0],  in2[0], in3[0])

lookback_length = 2.0  # Compute avoidance for 2 seconds
t_step = 0.05  # Time step resolution

tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)

sys = DubinsTwoCar_3d()
compMethod = { "TargetSetMode": "minVWithV0"}  # Avoidance problem

po = PlotOptions(
    do_plot=True,
    plot_type="set",
    plotDims=[0,1,2],  # Plot x1 vs y1, We fix the adversary's position and orientation (x2, y2, theta2) to mid-range values.
    # slicesCut=[math.floor(grid_resolution[2] / 2), math.floor(grid_resolution[3] / 2), math.floor(grid_resolution[4] / 2), math.floor(grid_resolution[5] / 2)],  
    slicesCut=[],
    save_fig=True,
    filename="collision_avoidance_3d.png"
)
result = HJSolver(sys, g, Initial_value_f, tau, compMethod, po, saveAllTimeSteps=True)
print("Result shape after solving:", result.shape)

# Extract the value function at the last time step
last_time_step_result = result[..., 0]
print("Last time step shape:", last_time_step_result.shape) 
# # Compute spatial derivatives at every state
# x1_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=1, accuracy="medium")
# y1_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=2, accuracy="medium")
# theta1_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=3, accuracy="medium")
# x2_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=4, accuracy="medium")
# y2_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=5, accuracy="medium")
# theta2_derivative = computeSpatDerivArray(g, last_time_step_result, deriv_dim=6, accuracy="medium")

# # Choose a random state index (10, 10, 7, 10, 10, 7) for analysis
# spat_deriv_vector = (
#     x1_derivative[10, 10, 7, 10, 10, 7], 
#     y1_derivative[10, 10, 7, 10, 10, 7],
#     theta1_derivative[10, 10, 7, 10, 10, 7], 
#     x2_derivative[10, 10, 7, 10, 10, 7], 
#     y2_derivative[10, 10, 7, 10, 10, 7], 
#     theta2_derivative[10, 10, 7, 10, 10, 7]
# )

# # Compute the optimal control for the ego car at this state
# opt_w1 = sys.opt_ctrl(0, None, spat_deriv_vector)
# print("Optimal Control for Ego Car:", opt_w1)

# # Compute the worst-case disturbance (adversary control)
# opt_w2 = sys.opt_dstb(0, None, spat_deriv_vector)
# print("Worst-case Disturbance for Adversary:", opt_w2)

plot_isosurface(g, last_time_step_result, po)
np.save("dp_value_function.npy", last_time_step_result)
