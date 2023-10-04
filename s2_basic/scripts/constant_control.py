#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


class ConstantControl(Node):
    def __init__(self) -> None:
        super().__init__("ConstantControl")  # initialize base class
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.twist_sub = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

        self.timer = self.create_timer(0.25, self.callback)

    def callback(self) -> None:
        """ Twist the turtle """

        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.0

        self.twist_pub.publish(msg)

    def kill_callback(self, msg: Bool) -> None:
        if msg.data:
            self.destroy_timer(self.timer)

            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0

            self.twist_pub.publish(twist_msg)
    


if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = ConstantControl()  
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context