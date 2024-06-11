# DRL_Final_Project

This repo contains the final project of NTHU -  CS565700 - DRL.

## Overview

In this study, we advance the field of Vision and Language Navigation (VLN) by identifying the optimal reinforcement learning (RL) strategy for visual navigation agents. We developed a comprehensive pipeline for VLN tasks within a simulated residential environment and trained three distinct RL algorithms. Our findings demonstrate that the ReDo algorithm Sokar et al. (2023) outperforms all other evaluated agents, establishing it as the most effective approach among those tested.

## How to run the code

### lang_graph_ws

Please follow the instructions in `/lang_graph_ws`.

### kobuki_ws

#### Docker

We have provided the docker compose file to help you easily launch the container. Please navigate to the `/kobuki_ws/docker` folder and use `docker compose up` to build and run the container. Additionally, we have provided the DevContainer configuration. If you are familiar with it, we highly recommend using DevContainer for added convenience.

#### Build the workspace

```=
cd /home/DRL_Final_Project/kobuki_ws
colcon build --symlink-install 
```

#### Run the Gazebo Simulator

```=
ros2 launch gazebo_rl_env rl_env.launch.py
```

### rl_ws

#### Docker

Just like in the `kobuki_ws` folder, we have provided both the Docker Compose file and the DevContainer configuration file. Please choose the one you prefer.

#### Build the workspace

```=
cd /home/DRL_Final_Project/rl_ws
colcon build --symlink-install 
```

#### Run the RL Agent

```=
ros2 launch dqn_agent agent.launch.py
```
