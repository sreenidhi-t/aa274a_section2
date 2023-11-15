#!/usr/bin/env python3

import numpy, rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool

class PerceptionController(BaseHeadingController):
    def __init__(self):
        super().__init__('perception_controller')
        self.declare_parameter("kp", 2.0)
        self.declare_parameter("active", True)
        self.image_detected = False
        self.image_det_sub = self.create_subscription(
            Bool, "/detector_bool", self.image_det_cb, 10
        )

    def compute_control_with_goal(self, currState: TurtleBotState, desiredState: TurtleBotState) -> TurtleBotControl:
        # head_error = wrap_angle(desiredState.theta - currState.theta)
        # omega_control = (float) (self.kp * head_error)
        turtleBotMsg = TurtleBotControl(omega=0.2)

        if self.image_detected:
            turtleBotMsg.omega = 0.0
            return turtleBotMsg


        return turtleBotMsg
    
    def image_det_cb(self, msg:Bool) -> None:
        if msg.data:
            self.image_detected = True

    

    
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value
    
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

if __name__ == '__main__':
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()
