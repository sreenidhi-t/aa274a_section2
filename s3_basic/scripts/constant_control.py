#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

class ConstantControl(Node):
    def __init__(self):
        super().__init__('constant_control_node')
        self.get_logger().info('Constant control node has been created!')

        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist_timer = self.create_timer(0.05, self.publish_control)

        self.kill_sub = self.create_subscription(Bool, '/kill', self.kill_callback, 10)

    def publish_control(self):
        twist_msg = Twist()
        twist_msg.linear.x = 1.0
        twist_msg.angular.z = 0.0    
        self.twist_pub.publish(twist_msg)

    def kill_callback(self, msg: Bool):
        if msg.data == True:
            self.get_logger().info("STOPPING!!")
            self.twist_timer.cancel()
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0    
            self.twist_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    constant_control = ConstantControl()
    rclpy.spin(constant_control)
    rclpy.shutdown()

if __name__ == "__main__":
    main()