import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import qos_profile_sensor_data

from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rosgraph_msgs.msg import Clock

from ament_index_python.packages import get_package_share_directory

import os
import math
import time
import yaml
import numpy as np

sim_clock = 0


class RESET_SERVICE(Node):
    def __init__(self):
        super().__init__("reset_service")

        self.reset_service = self.create_service(Empty, "/rl_env/reset", self.reset_callback)
        self.is_reset = False

    def reset_callback(self, request, response):
        self.is_reset = True

        while self.is_reset:
            # Wait for the reset to finish
            # Note that the reset will be finished in the main thread
            time.sleep(0.01)

        return response


class STEP_SERVICE(Node):
    def __init__(self):
        super().__init__("step_service")

        self.step_service = self.create_service(Empty, "/rl_env/step", self.step_callback)
        self.is_step = False

    def step_callback(self, request, response):
        self.is_step = True

        while self.is_step:
            # Wait for the step to finish
            # Note that the step will be finished in the main thread
            time.sleep(0.01)

        return response


class SIM_CLOCK_SUBSCRIBER(Node):
    def __init__(self):
        super().__init__("sim_clock_subscriber")
        self.sim_clock_subscriber = self.create_subscription(Clock, "/clock", self.sim_clock_callback, qos_profile_sensor_data)

    def sim_clock_callback(self, msg):
        global sim_clock

        sim_clock = msg.clock.sec


class GAZEBO_RL_ENV_NODE(Node):
    def __init__(self):
        super().__init__("gazebo_rl_env_node")

        self.config = {
            "step_time_delta": 1,  # seconds (Simulation time provided by Gazebo)
            "gazebo_service_timeout": 3.0,  # seconds (Real time for waiting the Gazebo service)
            "reach_target_distance": 0.2,  # meters
            "target_reward": 10,
            "penalty_per_step": -0.05,
            "max_step_without_reach_target": 25,
        }

        self.current_timestamp = 0
        self.current_reward = 0.0
        self.step_without_reach_target = 0
        self.is_done = False

        # Create the clients for the Gazebo services
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.get_entity_state_client = self.create_client(GetEntityState, "/get_entity_state")
        self.set_entity_state_client = self.create_client(SetEntityState, "/set_entity_state")

        # Create the publisher for the environment
        self.info_publisher = self.create_publisher(Float32MultiArray, "/rl_env/info", 10)

        # Current path list
        self.endpoints = []
        self.path_list = []
        self.ball_list = []

        # Graph
        self.node_list = []
        self.edge_list = []

        # Initialize
        self.read_graph()
        self.init_ball_list(ball_list_size=12)

    def read_graph(self):
        file_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "map", "small_house_2.yaml")

        # Load the graph from the YAML file
        graph = None
        with open(file_path, "r") as stream:
            try:
                graph = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self.get_logger().error(exc)

        if graph is not None:
            self.node_list = graph["node"]
            self.edge_list = graph["edge"]
            self.get_logger().info("Graph is loaded successfully.")
            # self.get_logger().info(f"Node list: {self.node_list}")
            # self.get_logger().info(f"Edge list: {self.edge_list}")

    def get_node_index(self, node_name: str):
        for node in self.node_list:
            if node["name"] == node_name:
                return node["index"]
        return None

    def get_path(self, target_endpoint: list):
        endpoint = target_endpoint.copy()

        # Ensure the target_endpoint is in ascending order
        if endpoint[0] > endpoint[1]:
            temp = endpoint[0]
            endpoint[0] = endpoint[1]
            endpoint[1] = temp

        # Find the path in the edge list
        path = []
        for edge in self.edge_list:
            if edge["endpoint"] == endpoint:
                path = edge["path"]
                break

        return path

    def wait_one_step_time(self):
        global sim_clock

        current_sim_clock = sim_clock
        target_sim_clock = current_sim_clock + self.config["step_time_delta"]

        # Wait for the simulation clock to reach the target
        while sim_clock < target_sim_clock:
            time.sleep(0.001)

    def reset(self, target_endpoints: list[int, int] = None):
        self.current_timestamp = 0
        self.current_reward = 0.0
        self.step_without_reach_target = 0
        self.is_done = False

        # Generate the new path list randomly
        # NOTE: Change the endpoints to generate specific paths when testing
        self.generate_path_list(endpoints=target_endpoints)

        # Move the Kobuki to the start point.
        kobuki_start_point_index = self.endpoints[np.random.randint(0, 2)]
        kobuki_start_point = self.node_list[kobuki_start_point_index]["position"]
        future = self.set_entity_state("kobuki", kobuki_start_point[0], kobuki_start_point[1], 0.0, np.random.uniform(0, 2 * np.pi))
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Move the endpoint ball to the endpoint
        endpoint_index = self.endpoints[0] if kobuki_start_point_index == self.endpoints[1] else self.endpoints[1]
        endpoint = self.node_list[endpoint_index]["position"]
        self.ball_list[-1].x = endpoint[0]
        self.ball_list[-1].y = endpoint[1]
        self.ball_list[-1].z = 0.2
        future = self.set_entity_state(self.ball_list[-1].name, self.ball_list[-1].x, self.ball_list[-1].y, self.ball_list[-1].z)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Unpause the physics to allow the simulation to run
        while not self.unpause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "unpause" not available, waiting again...')
        self.unpause_client.call_async(Empty.Request())

        # Wait for one step time to stabilize the environment
        self.wait_one_step_time()

        # Pause the physics to stop the simulation at the beginning
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        future = self.pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        self.get_logger().info(f"Reset environment with endpoint {self.endpoints}.")

    def step(self):
        self.current_timestamp += 1

        # Unpause the physics to allow the simulation to run
        while not self.unpause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "unpause" not available, waiting again...')
        self.unpause_client.call_async(Empty.Request())

        # Wait for one step
        self.wait_one_step_time()

        # Pause the physics to stop the simulation
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        self.pause_client.call_async(Empty.Request())

        # Reset the reward
        self.current_reward = self.config["penalty_per_step"]

        # Check whether the Kobuki reaches the target
        is_reach_target = False
        state = self.get_kobuki_state()
        for i in range(len(self.ball_list)):
            # Check whether the ball is still in the environment (above the ground)
            if self.ball_list[i].z > 0.0:
                distance = self.ball_list[i].get_distance(state)
                if distance < self.config["reach_target_distance"]:
                    is_reach_target = True

                    # Move the ball under the ground
                    self.ball_list[i].z = -2.0
                    future = self.set_entity_state(self.ball_list[i].name, self.ball_list[i].x, self.ball_list[i].y, self.ball_list[i].z)
                    rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

                    # If the Kobuki reaches the endpoint, set the done signal to True
                    if self.ball_list[i].name == "ball_endpoint":
                        self.is_done = True

                    # Calculate the reward
                    self.current_reward = self.config["target_reward"]

        # If the Kobuki does not reach the target, increase the step_without_reach_target
        if not is_reach_target:
            self.step_without_reach_target += 1
        else:
            self.step_without_reach_target = 0

        # Publish the information
        self.publish_info()

    def init_ball_list(self, ball_list_size: int):
        self.get_logger().info("Initialize the ball list.")

        # Load the ball URDF
        ball_urdf_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "urdf", "ball.urdf")
        ball_urdf = open(ball_urdf_path, "r").read()

        for i in range(ball_list_size):
            # Set the ball name. (The last ball is the endpoint ball)
            if i == ball_list_size - 1:
                ball_name = "ball_endpoint"
            else:
                ball_name = "ball_" + str(i)

            # Construct the spawn ball request
            self.spawn_ball_request = SpawnEntity.Request()
            self.spawn_ball_request.name = ball_name
            self.spawn_ball_request.xml = ball_urdf
            self.spawn_ball_request.initial_pose.position.x = 0.0
            self.spawn_ball_request.initial_pose.position.y = 0.0
            self.spawn_ball_request.initial_pose.position.z = -2.0

            # Append the ball to the ball list
            self.ball_list.append(BALL(0.0, 0.0, -2.0, ball_name))

            # Spawn the ball
            while not self.spawn_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
                self.get_logger().info('Gazebo service "spawn_entity" not available, waiting again...')
            future = self.spawn_entity_client.call_async(self.spawn_ball_request)
            rclpy.spin_until_future_complete(self, future)

        self.get_logger().info("Ball list is initialized.")

    def set_entity_state(self, name: str, x: float, y: float, z: float, yaw: float = 0.0) -> Future:
        self.set_entity_state_request = SetEntityState.Request()
        self.set_entity_state_request.state.name = name
        self.set_entity_state_request.state.pose.position.x = x
        self.set_entity_state_request.state.pose.position.y = y
        self.set_entity_state_request.state.pose.position.z = z
        self.set_entity_state_request.state.pose.orientation.z = math.sin(yaw / 2)
        self.set_entity_state_request.state.pose.orientation.w = math.cos(yaw / 2)

        # Set the entity state
        while not self.set_entity_state_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "set_entity_state" not available, waiting again...')
        future = self.set_entity_state_client.call_async(self.set_entity_state_request)
        return future

    def get_kobuki_state(self):
        self.get_kobuki_state_request = GetEntityState.Request()
        self.get_kobuki_state_request.name = "kobuki"

        while not self.get_entity_state_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "get_entity_state" not available, waiting again...')

        # Get the state of the Kobuki
        future = self.get_entity_state_client.call_async(self.get_kobuki_state_request)

        # Get the response
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])
        response = future.result()

        if response is None:
            self.get_logger().error("Failed to get the state of the Kobuki.")
            response = GetEntityState.Response()
            response.state.pose.position.x = 1000
            response.state.pose.position.y = 1000
            response.state.pose.position.z = 0

        # Print the state
        # kobuki_pose = response.state.pose
        # self.get_logger().info(f"Kobuki position: ({kobuki_pose.position.x}, {kobuki_pose.position.y}, {kobuki_pose.position.z})")

        return response.state

    def generate_path_list(self, endpoints: list[int, int] = None):
        # If the endpoints is None, randomly select an edge from the edge list
        if endpoints is None:
            random_index = np.random.randint(0, len(self.edge_list))
            endpoints = self.edge_list[random_index]["endpoint"]
        self.endpoints = endpoints

        # Get the path from the graph
        self.path_list = self.get_path(self.endpoints)

        # Move the ball to the path
        future_list = []
        for i in range(len(self.path_list)):
            self.ball_list[i].x = self.path_list[i][0]
            self.ball_list[i].y = self.path_list[i][1]
            self.ball_list[i].z = 0.2
            future_list.append(self.set_entity_state(self.ball_list[i].name, self.ball_list[i].x, self.ball_list[i].y, self.ball_list[i].z))

        # Move the remaining balls under the ground
        for i in range(len(self.path_list), len(self.ball_list) - 1):
            self.ball_list[i].z = -2.0
            future_list.append(self.set_entity_state(self.ball_list[i].name, self.ball_list[i].x, self.ball_list[i].y, self.ball_list[i].z))

        # Wait for all the ball to be moved
        for future in future_list:
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])
            if future.result() is None:
                self.get_logger().error("Failed to move the ball.")
            elif future.result().success is False:
                self.get_logger().error(future.result().status_message)

    def publish_info(self):
        msg = Float32MultiArray()

        # Add the timestamp
        msg.layout.dim.append(MultiArrayDimension(label="timestamp", size=1, stride=1))
        msg.data.append(self.current_timestamp)

        # Add the reward
        msg.layout.dim.append(MultiArrayDimension(label="reward", size=1, stride=1))
        msg.data.append(self.current_reward)

        # Add the done signal
        msg.layout.dim.append(MultiArrayDimension(label="success", size=1, stride=1))
        msg.data.append(self.is_done)

        # Add the max_step_without_reach_target
        msg.layout.dim.append(MultiArrayDimension(label="is_reach_max_step", size=1, stride=1))
        msg.data.append(self.step_without_reach_target > self.config["max_step_without_reach_target"])

        # Publish the message
        self.info_publisher.publish(msg)


class BALL:
    def __init__(self, x: float, y: float, z: float, name: str):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def get_distance(self, state):
        # Calculate the Euclidean distance between the target and the state,
        # note that we only consider the x and y coordinates.
        return math.sqrt((self.x - state.pose.position.x) ** 2 + (self.y - state.pose.position.y) ** 2)
