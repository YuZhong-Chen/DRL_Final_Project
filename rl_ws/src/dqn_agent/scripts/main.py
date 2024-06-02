#!/usr/bin/env python3

import rclpy

import os
import cv2
import time
import datetime
import math
import torch
import threading
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dqn_agent.rl_env import RL_ENV, OBSERVATION_SUBSCRIBER, ENV_INFO_SUBSCRIBER
from dqn_agent.wrapper import RESIZE_OBSERVATION, GRAY_SCALE_OBSERVATION, FRAME_STACKING, BLUR_OBSERVATION
from dqn_agent.agent import AGENT
from dqn_agent.logger import LOGGER

#############################################################################################
EPISODES = 10000
SAVE_INTERVAL = 50

PROJECT = "drl-final-project"
PROJECT_NAME = PROJECT + "-dqn-" + datetime.datetime.now().strftime("%m-%d-%H-%M")

# LOAD_MODEL_PATH = None
LOAD_MODEL_PATH = "drl-final-project-dqn-06-01-20-17/models/episode_850"

USE_LOGGER = True
USE_WANDB = True
#############################################################################################

checkpoint_dir = Path("/home/DRL_Final_Project/rl_ws/checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
project_dir = checkpoint_dir / PROJECT_NAME
project_dir.mkdir(exist_ok=True)

print("Project Name:", PROJECT_NAME)

agent = AGENT(project_dir=project_dir, load_path=(checkpoint_dir / LOAD_MODEL_PATH) if LOAD_MODEL_PATH is not None else None)
logger = LOGGER(project=PROJECT, project_name=PROJECT_NAME, config=agent.config, project_dir=project_dir, enable=USE_LOGGER, use_wandb=USE_WANDB)

RESIZE_OBSERVATION_ = RESIZE_OBSERVATION(shape=(320, 320))
GRAY_SCALE_OBSERVATION_ = GRAY_SCALE_OBSERVATION()
FRAME_STACKING_ = FRAME_STACKING(stack_size=4)
BLUR_OBSERVATION_ = BLUR_OBSERVATION(kernel_size=5)


def ProcessObservation(env, observation):
    try:
        observation = BLUR_OBSERVATION_.forward(observation)
        observation = RESIZE_OBSERVATION_.forward(observation)
        observation = GRAY_SCALE_OBSERVATION_.forward(observation)
        observation = FRAME_STACKING_.forward(observation)
    except Exception as e:
        env.get_logger().error(e)
        env.get_logger().info(f"Observation {observation}")
        observation = np.zeros((4, 320, 320), dtype=np.uint8)
    return observation


def main(args=None):
    rclpy.init(args=args)

    observation_subscriber_node = OBSERVATION_SUBSCRIBER()
    env_info_subscriber_node = ENV_INFO_SUBSCRIBER()

    # Create a MultiThreadedExecutor
    # Reference: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html
    # TODO: This is a temporary solution, and it may not be the best solution,
    # since the main thread use busy waiting, it may consume a lot of CPU resources.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(observation_subscriber_node)
    executor.add_node(env_info_subscriber_node)
    executer_thread = threading.Thread(target=executor.spin)
    executer_thread.start()

    env = RL_ENV()

    average_reward = 0
    average_step = 0
    collision_rate = 1.0
    success_rate = 0.0

    for episode in tqdm(range(agent.current_episode + 1, EPISODES + 1)):
        observation = env.reset()
        observation = ProcessObservation(env, observation)

        episode_step = 0
        episode_reward = 0
        loss_list = []
        td_error_list = []
        td_estimation_list = []

        while True:
            action = agent.Act(observation)

            next_observation, reward, done, info = env.step(action)
            next_observation = ProcessObservation(env, next_observation)

            # If the agent collides with the wall, the episode is terminated.
            # And we give a penalty to the agent.
            if info["is_collision"] or info["is_max_step"]:
                done = True
                reward -= 3

            agent.AddToReplayBuffer(observation, action, reward, done, next_observation)

            loss, td_error, td_estimation = agent.Train()

            episode_reward += reward
            loss_list.append(loss)
            td_error_list.append(td_error)
            td_estimation_list.append(td_estimation)

            observation = next_observation

            episode_step += 1

            if done:
                break

        agent.current_episode = episode
        average_loss = np.mean(loss_list)
        average_td_error = np.mean(td_error_list)
        average_td_estimation = np.mean(td_estimation_list)

        episode_average_reward = episode_reward / episode_step
        average_reward = 0.97 * average_reward + 0.03 * episode_average_reward
        average_step = 0.97 * average_step + 0.03 * episode_step

        collision_rate = 0.97 * collision_rate + 0.03 * (1 if info["is_collision"] else 0)
        success_rate = 0.97 * success_rate + 0.03 * (1 if info["is_success"] else 0)

        log_data = {}
        log_data["Train/Loss"] = average_loss
        log_data["Train/TD Error"] = average_td_error
        log_data["Train/TD Estimation"] = average_td_estimation
        log_data["Reward/Episode"] = episode_average_reward
        log_data["Reward/Average"] = average_reward
        log_data["Step/Episode"] = episode_step
        log_data["Step/Average"] = average_step
        log_data["Step/Total"] = agent.current_step
        log_data["Rate/Collision"] = collision_rate
        log_data["Rate/Success"] = success_rate
        logger.Log(episode, log_data)

        env.get_logger().info("")
        env.get_logger().info(f"Episode: {episode}, Step: {round(episode_step, 3)}, Reward: {round(episode_average_reward, 3)}")
        env.get_logger().info(f"Average Reward: {round(average_reward, 3)}, Average Step: {round(average_step, 3)}, Total Step: {agent.current_step}")
        env.get_logger().info(f"Loss: {round(average_loss, 3)}, TD Error: {round(average_td_error, 3)}, TD Estimation: {round(average_td_estimation, 3)}")
        env.get_logger().info(f"Collision Rate: {round(collision_rate, 3)}, Success Rate: {round(success_rate, 3)}")

        if episode % SAVE_INTERVAL == 0:
            env.get_logger().info("Save model.")
            agent.SaveModel()


if __name__ == "__main__":
    main()
