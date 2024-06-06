# python related
import numpy as np
import random
from collections import deque

# training related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# gym related
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import wandb

from test import Agent

n_episode = 30000

LOG_FREQ = 100

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    wandb.init(project="mario-with-dqn", mode="disabled")
    # Create the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Create the agent
    agent = Agent()
    agent.model.train()
    agent.init_target_model()
    wandb.watch(agent.model, log_freq=LOG_FREQ)
    
    state_stack_temp = None
    action_temp = None
    reward_temp = None
    done_temp = None

    show_state = None
    show_next_state = None
    score_per_5_episode = 0
    # show_flag = True

    # Train the agent
    for episode in (range(n_episode)):
        state = env.reset()
        done = False
        score = 0
        first_action = True
        _4_frames_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            _4_frames_reward += reward

            if agent.pick_action_flag:
                if first_action:
                    
                    first_action = False

                    # just store
                    state_stack_temp = agent.stacked_img_buf
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done

                    _4_frames_reward = 0
                else:
                    # call remember() (to remember the last transition)
                    next_state_stack = agent.stacked_img_buf
                    action_temp = torch.tensor(action_temp).to(torch.int8)
                    reward_temp = torch.tensor(reward_temp).to(torch.int8)
                    done_temp = torch.tensor(done_temp).to(torch.int8)

                    agent.remember(state_stack_temp, \
                                    action_temp, \
                                    reward_temp, \
                                    next_state_stack, \
                                    done_temp)
                    
                    # for plotting
                    # if show_flag and np.random.rand() < 0.01:
                    #     show_flag = False
                    #     show_state = state_stack_temp.cpu().numpy()
                    #     show_next_state = next_state_stack.cpu().numpy()

                    # store
                    state_stack_temp = next_state_stack
                    action_temp = action
                    reward_temp = _4_frames_reward
                    done_temp = done

                    agent.replay()
                    _4_frames_reward = 0

            state = next_state
            score += reward
        print(f"Episode: {episode}, Score: {score}")

        score_per_5_episode += score
        if episode % 5 == 0:
            wandb.log({"score per 5 epi": score_per_5_episode / 5})
            score_per_5_episode = 0

    # Save the trained model
    agent.save("try.pt")

    env.close()

    # matplotlib.use('TkAgg')
    # fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # for i in range(2):
    #     for j in range(4):
    #         if i == 0:
    #             axes[i, j].imshow(show_state[j], cmap='gray')
    #         else:
    #             axes[i, j].imshow(show_next_state[j], cmap='gray')
    # plt.axis('off')  # Turn off axis
    # plt.tight_layout()
    # plt.show()
    # Close the environment