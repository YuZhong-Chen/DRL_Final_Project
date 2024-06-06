import rclpy
from rclpy.node import Node

from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from gazebo_msgs.msg import ContactsState

from cv_bridge import CvBridge

import os
import math
import time
import numpy as np

timestamp = 0
reward = 0
info = {"is_collision": False, "is_success": False, "is_max_step": False}
done = False
observation = None


class OBSERVATION_SUBSCRIBER(Node):
    def __init__(self):
        super().__init__("observation_subscriber")
        self.observation_subscriber = self.create_subscription(Image, "/zed/zed_node/left/image_rect_color", self.observation_callback, 10)

        # OpenCV bridge
        # Reference: https://wiki.ros.org/cv_bridge
        self.cv_bridge = CvBridge()

    def observation_callback(self, msg):
        global observation
        observation = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")


class ENV_INFO_SUBSCRIBER(Node):
    def __init__(self):
        super().__init__("env_info_subscriber")
        self.env_info_subscriber = self.create_subscription(Float32MultiArray, "/rl_env/info", self.env_info_callback, 10)
        self.collision_subscriber = self.create_subscription(ContactsState, "/collision", self.collision_callback, 10)

    def env_info_callback(self, msg):
        global timestamp, reward, done, info
        timestamp = msg.data[0]
        reward = msg.data[1]
        info["is_success"] = msg.data[2]
        info["is_max_step"] = msg.data[3]
        done = info["is_success"] or info["is_max_step"]

    def collision_callback(self, msg):
        global info
        info["is_collision"] = msg.states != []


class RL_ENV(Node):
    def __init__(self):
        super().__init__("rl_env")

        self.config = {
            "service_timeout": 1.0,  # seconds
        }

        # Create a publisher for the action
        self.action_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Create a service client for controlling the env
        self.step_service = self.create_client(Empty, "/rl_env/step")
        self.reset_service = self.create_client(Empty, "/rl_env/reset")

        # Action space
        # 0 - stop
        # 1 - forward
        # 2 - forward right
        # 3 - forward left
        # 4 - turn right
        # 5 - turn left
        self.action_space = 6

        self.timestamp = 0
        self.reward = 0

        self.data_timeout_count = 0
        self.wait_for_data_update_timeout = 1.0

    def publish_action(self, action: int):
        twist = Twist()

        if action == 0:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif action == 1:
            twist.linear.x = 0.239
            twist.angular.z = 0.0
        elif action == 2:
            twist.linear.x = 0.239
            twist.angular.z = -0.205
        elif action == 3:
            twist.linear.x = 0.239
            twist.angular.z = 0.205
        elif action == 4:
            twist.linear.x = 0.0
            twist.angular.z = -0.205
        elif action == 5:
            twist.linear.x = 0.0
            twist.angular.z = 0.205

        self.action_publisher.publish(twist)

    def reset(self):
        # Stop the car
        self.publish_action(0)

        # Call the reset service
        while not self.reset_service.wait_for_service(timeout_sec=self.config["service_timeout"]):
            self.get_logger().info('Env service "reset" not available, waiting again...')
        future = self.reset_service.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        # Call the step service for the first time
        observation = None
        while observation is None:
            observation, _, done, info = self.step(0)
            if done or info["is_max_step"]:
                break

        # Check if the observation is received
        if observation is None:
            self.get_logger().error('Env service "reset" failed to receive data.')
            self.get_logger().error("Recalling the reset function...")
            observation = self.reset()

        return observation

    def step(self, action: int):
        global observation, reward, done, info

        # Since we need to wait for the data coming from the subscriber,
        # we set them to None first, and then wait for them to be updated.
        old_observation = observation
        old_reward = reward
        old_info = info

        # Since the info come from the same subscriber, we only need to checkout one of them.
        observation = reward = info["is_collision"] = None

        self.publish_action(action)

        # Call the step service
        while not self.step_service.wait_for_service(timeout_sec=self.config["service_timeout"]):
            self.get_logger().info('Env service "step" not available, waiting again...')
        future = self.step_service.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        # Wait for the data to be updated.
        # If the data is not updated, use the old data.
        is_data_updated = False
        timeout_period = self.wait_for_data_update_timeout / 100.0
        for _ in range(100):
            if observation is None or reward is None or info["is_collision"] is None:
                # self.get_logger().info(f'Env service "step" is waiting for data...')
                time.sleep(timeout_period)
            else:
                is_data_updated = True
                break

        if is_data_updated is False:
            self.data_timeout_count += 1
            self.get_logger().error('Env service "step" failed to receive data, use the old data.')

            # Check which data is not received
            is_observation = "Received" if observation is not None else "None"
            is_reward = "Received" if reward is not None else "None"
            is_collision = "Received" if info["is_collision"] is not None else "None"
            self.get_logger().error(f"Observation: {is_observation}, Reward: {is_reward}, Info: {is_collision}")
            self.get_logger().error(f"Timeout: {self.wait_for_data_update_timeout} seconds.")

            return old_observation, old_reward, done, old_info
        else:
            # If the data is received, decrease the waiting time for the next time.
            self.wait_for_data_update_timeout -= timeout_period
            self.wait_for_data_update_timeout = max(self.wait_for_data_update_timeout, 1.0)

        # Dynamic increase the waiting time if the data is not updated
        if self.data_timeout_count > 5:
            self.data_timeout_count = 0
            self.wait_for_data_update_timeout += 1.0
            self.get_logger().warn(f"Data timeout count is over 5, increase the waiting time to {self.wait_for_data_update_timeout} seconds.")

        return observation, reward, done, info
