#!/usr/bin/env python3

import numpy as np
import typing as T
# import matplotlib.pyplot as plt
import rclpy
# from navigator import Navigator
# from scipy.interpolate import splev, splrep
# from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
# from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D
from scipy.signal import convolve2d
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool


def explore(occupancy: StochOccupancyGrid2D, current_statet: TurtleBotState):
    """ returns potential states to explore
    Args:
        occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

    Returns:
        frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.
    HINTS:
    - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
    - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
    """

    window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
    ########################### Code starts here ###########################
    window = np.ones((window_size, window_size)) / (window_size ** 2)
    unknown_grid = (occupancy.probs == -1)*1  # 1 if unknown
    known_occu_grid = (occupancy.probs >= 0.5)*1 # 1 if known and occupied
    known_unoc_grid = np.multiply((occupancy.probs != -1)*1, (occupancy.probs <= 0.5)*1) # 1 if known and unoccupied

    # Convolve all three conditions
    unknown_frac = convolve2d(unknown_grid, window, mode='same')
    known_occu_frac = convolve2d(known_occu_grid, window, mode='same')
    known_unoc_frac = convolve2d(known_unoc_grid, window, mode='same')

    # Boolean operations
    valid_cells = np.multiply(np.multiply((unknown_frac >= 0.2)*1, (known_occu_frac == 0)*1), (known_unoc_frac >= 0.3)*1)
    valid_cell_ind = np.array(np.where(valid_cells)).T[:, [1,0]]
    frontier_states = occupancy.grid2state(valid_cell_ind)

    # Calculate the closest state
    current_state = np.array([current_statet.x, current_statet.y])
    # min_dist_state = np.argmin(np.linalg.norm(frontier_states - current_state, axis=1))
    #print(min_dist)

    ########################### Code ends here ###########################
    return frontier_states #[min_dist_state, :]


class Explore(Node):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("explore")
        self.goal_post_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)

        
        self.occupancy_sub = self.create_subscription(
            OccupancyGrid, "/map", self.occupancy_callback, 10)
        

        self.current_state_sub = self.create_subscription(
            TurtleBotState, "/state", self.current_state_callback, 10)
        

        self.nav_state_sub = self.create_subscription(
            Bool, "/nav_success", self.explore_callback, 10)
        
        self.nav_state_pub = self.create_publisher(Bool, "/nav_success", 10)


        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        self.message_sent = None


        self.timer_ = self.create_timer(5.0, self.timer_callback)
        self.attempts = 0
        self.have_occu = False
        self.have_state = False

    
    
    def timer_callback(self) -> None:
        # Send message once to start the exploration
        if not self.message_sent:
            msg = Bool()
            msg.data = True
            self.nav_state_pub.publish(msg)
            self.get_logger().info('Published True to /nav_success topic')
            self.message_sent = True 


    def occupancy_callback(self, msg:OccupancyGrid) -> None:
        # self.get_logger().info('Got Occupancy Map!')
        self.have_occu = True
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def current_state_callback(self, msg:TurtleBotState) -> None:
        # self.get_logger().info('Got state')
        self.have_state = True
        self.state = msg

    def explore_callback(self, msg:Bool) -> None:
        if self.have_occu and self.have_state:
            success = msg.data
            self.get_logger().info('exp Bool' + str(success) + str(type(success)))
            x_curr = self.state.x
            y_curr = self.state.y
            xygoal = self.generate_goal_post(success)

            if xygoal is not None:
                x = xygoal[0]
                y = xygoal[1]
                theta = np.arctan2(y-y_curr, x-x_curr)
                msgo = TurtleBotState()
                # msgo.x = 0.2
                # msgo.y = 0.6
                # msgo.theta = 0.1
                # self.goal_post_pub.publish(msgo)
                msgo.x = x
                msgo.y = y
                msgo.theta = theta
                self.goal_post_pub.publish(msgo)
            else:
                self.get_logger().info('Nowhere to go')


    

    def generate_goal_post(self, message):
        if self.occupancy is None:
            self.get_logger().warn("Unable to explore: occupancy map not yet available")
            return
        if self.state is None:
            self.get_logger().warn("Unable to explore: state not yet available")
            return
        posts = explore(self.occupancy, self.state)

        if posts.shape[0] == 0:
            return None
        
        if message:
            self.attempts = 0
            min_dist_state = np.argmin(np.linalg.norm(posts - np.array([self.state.x, self.state.y]), axis=1))
            return posts[min_dist_state]
        elif self.attempts <= 3:
            num_rows = posts.shape[0]
            # Select a random row index
            random_row_index = np.random.choice(num_rows)

            # Use the random row index to select the random row
            random_row = posts[random_row_index, :]   
            self.attempts += 1
            return random_row   
        else:
            # If really can't find a place
            alpha = np.random.random()
            beta = np.random.random() * np.pi * 2
            return alpha * np.array([np.cos(beta), np.sin(beta)])


if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Explore()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits

