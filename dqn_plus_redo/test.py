
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np

import torch
import math
import wandb

from typing import NamedTuple
from agent import QNetwork
from redo import run_redo

ACTION_SIZE = 12  # Number of valid actions in the game
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.0001  # Learning rate
BATCH_SIZE = 32  # Batch size for training
MEMORY_SIZE = 20000  # Size of the replay memory buffer
REDO_CHECK_INTERVAL = 1000
REDO_TAU = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBufferSamples(NamedTuple):
    """Container for replay buffer samples."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class ReplayMemory_Per(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity=MEMORY_SIZE, a=0.6, e=0.01):
        self.tree = SumTree(capacity)
        self.memory_size = capacity
        self.prio_max = 0.1
        self.a = a
        self.e = e

    def push(self, transition):
        p = (np.abs(self.prio_max) + self.e) ** self.a  # proportional priority
        self.tree.add(p, transition)

    def sample(self, batch_size):
        idxs = []
        priorities = []
        sample_datas = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if not isinstance(data, tuple):
                print(idx, p, data, self.tree.write)
            idxs.append(idx)
            priorities.append(p)
            sample_datas.append(data)
        return idxs, priorities, sample_datas

    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries
    
class Agent:
    def __init__(self):
        
        self.memory = ReplayMemory_Per(capacity=MEMORY_SIZE)
        self.model = QNetwork().to(device)
        self.target_model = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.steps_counter = 0

        self.frames_counter = 0

        self.stacked_img = None
        self.stacked_img_buf = None
        self.prev_action = 0 # initialize as NOOP
        self.pick_action_flag = False

        # epsilon greedy
        self.eps_end = 0.01
        self.eps_decay = 0.997
        self.epsilon = 1

        # self.load("data.py")

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        # if copy:
        #     return torch.tensor(array, device=device)
        return torch.as_tensor(array, device=device)
    
    def init_target_model(self): # used only before training
        self.target_model = QNetwork().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
    
    def act(self, observation):
        # grayscale the image
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, axis=2)


        if self.frames_counter != 12:
            
            # stack image
            if self.frames_counter == 0:
                self.stacked_img = observation
            elif self.frames_counter % 4 == 0:
                self.stacked_img = np.concatenate((self.stacked_img, observation), axis=2)

            # update member variables
            self.pick_action_flag = False

            # update frames counter
            self.frames_counter += 1

            return self.prev_action
        
        else: # self.frames_counter == 12
            
            # stack image
            self.stacked_img = np.concatenate((self.stacked_img, observation), axis=2)
            self.stacked_img = np.int8(self.stacked_img)
            self.stacked_img = torch.from_numpy(self.stacked_img).float()
            self.stacked_img = self.stacked_img.permute(2, 0, 1)
            self.stacked_img = self.stacked_img.unsqueeze(0).to(device)
            
            # pick new action
            if np.random.rand() <= self.epsilon:
                new_action = np.random.randint(0, 12)
            else:
                q_values = self.model(self.stacked_img)
                new_action = q_values.max(1)[1].item()

            # update member variables
            self.stacked_img_buf = self.stacked_img.squeeze(0).to(torch.int8)
            self.stacked_img = None
            self.prev_action = new_action
            self.pick_action_flag = True

            # update frames counter
            self.frames_counter = 0

            return new_action

    def replay(self):
        if self.memory.size() < BATCH_SIZE:
            return

        idxs, priorities, sample_datas = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample_datas)

        # compute weights for loss update
        weights = np.power(np.array(priorities) + self.memory.e, -self.memory.a)
        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        states = torch.stack(states).float()
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).float().to(device)
        next_states = torch.stack(next_states).float()
        dones = torch.FloatTensor(dones).float().to(device)

        
        # print(type(states), type(actions), type(rewards), type(next_states), type(dones))
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach() # max Q (the maximum q-value picking the best action)
        expected_q_values = (rewards + GAMMA * next_q_values * (1 - dones)) # y = r + gamma * max Q
        loss = (weights * torch.nn.MSELoss()(q_values, expected_q_values)).mean()
        
        wandb.log({"loss": loss.item()})
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1000, norm_type=2)
        self.optimizer.step()

        # update PER
        td_errors = (q_values - expected_q_values).detach().squeeze().tolist()
        self.memory.update(idxs, td_errors)

        # copy online network to target network
        if self.steps_counter % 10000 == 0:
            # print("copy!")
            self.target_model.load_state_dict(self.model.state_dict())

        # epsilon decay
        if self.steps_counter % 1000 == 0:
            # print("decay!")
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
            wandb.log({"epsilon": self.epsilon})

        # ReDO mechanism
        if self.steps_counter % REDO_CHECK_INTERVAL == 0:
            # print("redo!")
            data = (states, actions, rewards, next_states, dones)
            RB_sample = ReplayBufferSamples(*tuple(map(self.to_torch, data)))
            redo_out = run_redo(
                RB_sample,
                model=self.model,
                optimizer=self.optimizer,
                tau=REDO_TAU,
                re_initialize=True,
                use_lecun_init=False
            )

            self.model = redo_out["model"]
            self.optimizer = redo_out["optimizer"]


        self.steps_counter += 1
        self.model.reset_noise()
        self.target_model.reset_noise()

    def load(self, name):
        self.model.eval()
        # self.model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        weights = self.model.state_dict()
        torch.save(weights, name)

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state = env.reset()

    agent = Agent()
    
    agent.model.eval()
    done = False
    total_reward = 0
    for step in range(5000):
        if done:
            state = env.reset()
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = np.expand_dims(state, axis=2)
            print(total_reward)
            total_reward = 0
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    env.close()
#