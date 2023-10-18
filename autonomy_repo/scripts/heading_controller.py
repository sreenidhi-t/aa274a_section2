#!/usr/bin/env python3

import numpy, rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__('heading_controller')
        self.declare_parameter("kp", 2.0)

    def compute_control_with_goal(self, currState: TurtleBotState, desiredState: TurtleBotState) -> TurtleBotControl:
        head_error = wrap_angle(desiredState.theta - currState.theta)
        omega_control = (float) (self.kp * head_error)
        turtleBotMsg = TurtleBotControl(omega=omega_control)
        return turtleBotMsg
    
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

if __name__ == '__main__':
    rclpy.init()
    node = HeadingController()
    rclpy.spin(node)
    rclpy.shutdown()
