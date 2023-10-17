#!/usr/bin/env python3

import sys

sys.path.append('/pora/aa274_ws/AA274a-HW1')

import numpy as np
import rclpy, scipy.interpolate
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from P1_star import DetOccupancyGrid2D, AStar
# from utils import plot_line_segments

import typing as T

class Navigator(BaseNavigator):
    def __init__(self):
        super().__init__()
        self.kp = 2.0

        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        # self.t_prev = 0

        self.coeffs = np.zeros(8)
        self.v_prev_thres = 0.0001

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        head_error = wrap_angle(goal.theta - state.theta)
        omega_control = (float) (self.kp * head_error)
        turtleBotMsg = TurtleBotControl(omega=omega_control)
        return turtleBotMsg
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        dt = t - self.t_prev

        x, y, th = state

        x_d = scipy.interpolate.splev(plan[0])
        y_d = scipy.interpolate.splev(plan[1])
        xd_d = scipy.interpolate.splev(plan[0], der=1)
        yd_d = scipy.interpolate.splev(plan[1], der=1)
        xdd_d = scipy.interpolate.splev(plan[0], der=2)
        ydd_d = scipy.interpolate.splev(plan[1], der=2)

        if self.V_prev < self.v_prev_thres:
            self.V_prev = self.v_prev_thres
        u1 = xdd_d + self.kpx*(x_d - x) + self.kdx*(xd_d - self.V_prev*np.cos(th))
        u2 = ydd_d + self.kpy*(y_d - y) + self.kdy*(yd_d - self.V_prev*np.sin(th))
        b = np.array([u1, u2]).T
        A = np.array([[np.cos(th), -self.V_prev*np.sin(th)],[np.sin(th), self.V_prev*np.cos(th)]])
        [a, om] = np.linalg.solve(A, b)
        # integrate to get V
        V = self.V_prev + dt*a 
        

        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om
    
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        

    



