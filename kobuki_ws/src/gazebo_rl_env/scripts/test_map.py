#!/usr/bin/env python3

import rclpy
from gazebo_rl_env.gazebo_rl_env import GAZEBO_RL_ENV_NODE, RESET_SERVICE

from ament_index_python.packages import get_package_share_directory

import os
import time


def main(args=None):
    rclpy.init(args=args)

    gazebo_rl_env = GAZEBO_RL_ENV_NODE()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
