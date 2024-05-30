import rclpy
from rclpy.node import Node
from rclpy.task import Future

from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty

from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ament_index_python.packages import get_package_share_directory

import os
import math
import time
import yaml
import numpy as np


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
            time.sleep(0.05)

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
            time.sleep(0.05)

        return response


class GAZEBO_RL_ENV_NODE(Node):
    def __init__(self):
        super().__init__("gazebo_rl_env_node")

        self.config = {
            "step_time_delta": 0.5,  # seconds
            "gazebo_service_timeout": 1.0,  # seconds
            "reach_target_distance": 0.2,  # meters
            "target_reward": 100,
            "penalty_per_step": -0.01,
        }

        self.current_timestamp = 0
        self.current_reward = 0.0

        # Create the clients for the Gazebo services
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.reset_world_client = self.create_client(Empty, "/reset_world")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.get_entity_state_client = self.create_client(GetEntityState, "/get_entity_state")
        self.set_entity_state_client = self.create_client(SetEntityState, "/set_entity_state")

        # Create the publisher for the environment
        self.info_publisher = self.create_publisher(Float32MultiArray, "/rl_env/info", 10)

        # Load the ball URDF
        ball_urdf_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "urdf", "ball.urdf")
        self.ball_urdf = open(ball_urdf_path, "r").read()

        # Current path list
        self.endpoints = []
        self.path_list = []
        self.ball_list = []

        # Graph
        self.node_list = []
        self.edge_list = []

        # Initialize
        self.read_graph()
        self.reset()

    def read_graph(self):
        file_path = os.path.join(get_package_share_directory("gazebo_rl_env"), "map", "small_house.yaml")

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
        # Ensure the target_endpoint is in ascending order
        if target_endpoint[0] > target_endpoint[1]:
            temp = target_endpoint[0]
            target_endpoint[0] = target_endpoint[1]
            target_endpoint[1] = temp

        # Find the path in the edge list
        path = []
        for edge in self.edge_list:
            if edge["endpoint"] == target_endpoint:
                path = edge["path"]
                break

        return path

    def reset(self):
        self.current_timestamp = 0
        self.current_reward = 0.0

        # Reset the gazebo environment
        while not self.reset_world_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "reset_world" not available, waiting again...')
        future = self.reset_world_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Generate the new path list randomly
        # NOTE: Change the endpoints to generate specific paths when testing
        self.generate_path_list(endpoints=None)

        # Move the Kobuki to the start point.
        kobuki_start_point_index = self.endpoints[np.random.randint(0, 2)]
        kobuki_start_point = self.node_list[kobuki_start_point_index]["position"]
        future = self.set_entity_state("kobuki", kobuki_start_point[0], kobuki_start_point[1], 0.0, np.random.uniform(0, 2 * np.pi))
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Spawn the other endpoint
        endpoint_index = self.endpoints[0] if kobuki_start_point_index == self.endpoints[1] else self.endpoints[1]
        endpoint = self.node_list[endpoint_index]["position"]
        future = self.spawn_ball(endpoint[0], endpoint[1], 0.2, "ball_endpoint")
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        # Unpause the physics to allow the simulation to run
        while not self.unpause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "unpause" not available, waiting again...')
        self.unpause_client.call_async(Empty.Request())

        # Wait for the world stable
        time.sleep(self.config["step_time_delta"])

        # Pause the physics to stop the simulation at the beginning
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        future = self.pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        self.get_logger().info("Reset environment.")

    def step(self):
        self.current_timestamp += 1
        # self.get_logger().info(f"Current timestamp: {self.current_timestamp}")

        # Unpause the physics to allow the simulation to run
        while not self.unpause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "unpause" not available, waiting again...')
        self.unpause_client.call_async(Empty.Request())

        # Wait for the Gazebo to run for a certain amount of time.
        # Note that Gazebo will run 1000 iterations per second by default,
        # so the real simulation timestamp will be: 0.001 * self.config["step_time_delta"] * current_timestamp
        time.sleep(self.config["step_time_delta"])

        # Pause the physics to stop the simulation
        while not self.pause_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "pause" not available, waiting again...')
        self.pause_client.call_async(Empty.Request())

        # Reset the reward
        self.current_reward = self.config["penalty_per_step"]

        # Check whether the Kobuki reaches the target
        state = self.get_kobuki_state()
        for i in range(len(self.ball_list)):
            if self.ball_list[i] is not None:
                distance = self.ball_list[i].get_distance(state)
                if distance < self.config["reach_target_distance"]:
                    self.get_logger().info(f"Kobuki reaches target {self.ball_list[i].name} at timestamp {self.current_timestamp}")
                    self.delete_ball(self.ball_list[i].name)
                    self.current_reward = self.config["target_reward"]
                    self.ball_list[i] = None

        # Publish the information
        self.publish_info()

    def spawn_ball(self, x: float, y: float, z: float, name: str) -> Future:
        self.spawn_ball_request = SpawnEntity.Request()
        self.spawn_ball_request.name = name
        self.spawn_ball_request.xml = self.ball_urdf
        self.spawn_ball_request.initial_pose.position.x = x
        self.spawn_ball_request.initial_pose.position.y = y
        self.spawn_ball_request.initial_pose.position.z = z

        # Append the ball to the ball list
        self.ball_list.append(TARGET(x, y, z, name))

        # Spawn the ball
        while not self.spawn_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "spawn_entity" not available, waiting again...')
        future = self.spawn_entity_client.call_async(self.spawn_ball_request)
        return future

    def delete_ball(self, name: str) -> Future:
        self.delete_ball_request = DeleteEntity.Request()
        self.delete_ball_request.name = name

        # Delete the ball
        while not self.delete_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "delete_entity" not available, waiting again...')
        future = self.delete_entity_client.call_async(self.delete_ball_request)
        return future

    def spawn_kobuki(self, x: float, y: float, z: float, yaw: float):
        # Use system call to spawn the Kobuki robot
        os.system(f"ros2 run gazebo_ros spawn_entity.py -entity kobuki -topic /robot_description -x {x} -y {y} -z {z} -Y {yaw}")

    def delete_kobuki(self):
        self.delete_kobuki_request = DeleteEntity.Request()
        self.delete_kobuki_request.name = "kobuki"

        # Delete the Kobuki
        while not self.delete_entity_client.wait_for_service(timeout_sec=self.config["gazebo_service_timeout"]):
            self.get_logger().info('Gazebo service "delete_entity" not available, waiting again...')
        self.delete_entity_client.call_async(self.delete_kobuki_request)

    def set_entity_state(self, name: str, x: float, y: float, z: float, yaw: float) -> Future:
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

        response = None
        while response is None:
            future = self.get_entity_state_client.call_async(self.get_kobuki_state_request)

            # Get the response
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])
            response = future.result()

            if response is None:
                self.get_logger().info('Gazebo service "get_entity_state" call failed, waiting again...')

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

        # Clear the ball list if it is not empty
        if len(self.ball_list) > 0:
            self.clear_ball_list()

        # Get the path from the graph
        self.path_list = self.get_path(self.endpoints)

        # Spawn the targets
        future_list = []
        for i in range(len(self.path_list)):
            future_list.append(self.spawn_ball(self.path_list[i][0], self.path_list[i][1], 0.2, "ball_" + str(i)))

        # Wait for all the targets to be spawned
        for future in future_list:
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

    def clear_ball_list(self):
        # Delete all the balls in the ball list
        future_list = []
        for i in range(len(self.ball_list)):
            if self.ball_list[i] is not None:
                future_list.append(self.delete_ball(self.ball_list[i].name))

        # Wait for all the balls to be deleted
        for future in future_list:
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.config["gazebo_service_timeout"])

        self.ball_list = []

    def publish_info(self):
        msg = Float32MultiArray()

        # Add the timestamp
        msg.layout.dim.append(MultiArrayDimension(label="timestamp", size=1, stride=1))
        msg.data.append(self.current_timestamp)

        # Add the reward
        msg.layout.dim.append(MultiArrayDimension(label="reward", size=1, stride=1))
        msg.data.append(self.current_reward)

        # Publish the message
        self.info_publisher.publish(msg)


class TARGET:
    def __init__(self, x: float, y: float, z: float, name: str):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def get_distance(self, state):
        # Calculate the Euclidean distance between the target and the state,
        # note that we only consider the x and y coordinates.
        return math.sqrt((self.x - state.pose.position.x) ** 2 + (self.y - state.pose.position.y) ** 2)
