import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from dqn_agent.dqn import DQN
from dqn_agent.per import PRIORITIZED_EXPERIENCE_REPLAY
from dqn_agent.redo import run_redo


class AGENT:
    def __init__(self, project_dir=None, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = {
            "batch_size": 128,
            "learning_rate": 0.0003,
            "gamma": 0.85,
            "replay_buffer_size": 10000,
            "warmup_steps": 5000,
            "tau": 0.001,
            "optimizer": "Adam",
            "loss": "MSE",
            "max_grad_norm": 10.0,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 2000,
            "enable_redo": False,
            "redo_steps": 1000,
            "redo_tau": 0.1,
        }

        self.network = None
        self.replay_buffer = None
        self.optimizer = None
        self.loss = None
        self.action_space_len = 6

        self.current_step = 0
        self.current_episode = 0

        if project_dir is not None:
            self.model_dir = project_dir / "models"
            self.model_dir.mkdir(exist_ok=True)

        if load_path is not None:
            self.LoadModel(load_path)
        else:
            self.Init()

    def Init(self):
        self.network = DQN(tau=self.config["tau"])
        self.replay_buffer = PRIORITIZED_EXPERIENCE_REPLAY(capacity=self.config["replay_buffer_size"])

        if self.config["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.network.learning_network.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "AdamW":
            self.optimizer = optim.AdamW(self.network.learning_network.parameters(), lr=self.config["learning_rate"], amsgrad=True)
        elif self.config["optimizer"] == "RMSprop":
            self.optimizer = optim.RMSprop(self.network.learning_network.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.network.learning_network.parameters(), lr=self.config["learning_rate"])

        if self.config["loss"] == "MSE":
            self.loss = nn.MSELoss(reduction="mean")
        elif self.config["loss"] == "L1Loss":
            self.loss = nn.L1Loss()
        elif self.config["loss"] == "SmoothL1Loss":
            self.loss = nn.SmoothL1Loss()

        # Add the epsilon parameters
        self.epsilon_start = self.config["epsilon_start"]
        self.epsilon_end = self.config["epsilon_end"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.epsilon = self.epsilon_start

    def UpdateEpsilon(self) -> None:
        # Update the epsilon using the exponential decay function.
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.0 * self.current_episode / self.epsilon_decay)

    def Act(self, state):
        # With epsilon-greedy policy
        self.UpdateEpsilon()

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space_len)
        else:
            with torch.no_grad():
                state = torch.tensor(np.array(state), dtype=torch.int8).unsqueeze(0).to(self.device)
                action = torch.argmax(self.network.learning_network(state)).item()

        return action

    def AddToReplayBuffer(self, state, action, reward, done, next_state):
        self.replay_buffer.Add(state, action, reward, done, next_state)

    def Train(self):
        self.current_step += 1

        if self.replay_buffer.current_size < self.config["warmup_steps"]:
            return 0, 0, 0, None

        # Sample batch data from replay buffer.
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = self.replay_buffer.Sample(self.config["batch_size"])

        # Move the data to the device.
        next_state_batch = next_state_batch.to(self.device)

        # Calculate TD target.
        with torch.no_grad():
            max_action = torch.argmax(self.network.learning_network(next_state_batch), dim=1)
            next_q = self.network.target_network(next_state_batch).gather(1, max_action.unsqueeze(-1))
            td_target = reward_batch + (~done_batch) * self.config["gamma"] * next_q

        # Calculate TD estimation.
        td_estimation = self.network.learning_network(state_batch).gather(1, action_batch)

        # Compute loss
        loss = self.loss(td_estimation, td_target)

        # Update network.
        # Clip the gradient to prevent the gradient explosion.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.learning_network.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()

        # Update target network. (Soft update)
        self.network.UpdateTargetNetwork()

        # Update the priority of the replay buffer.
        with torch.no_grad():
            td_error = torch.abs(td_target - td_estimation).cpu()
            self.replay_buffer.UpdatePriority(td_error)

        # Apply the REDO algorithm
        dormant_fraction = None
        if self.current_step % self.config["redo_steps"] == 0:
            result = run_redo(state_batch=state_batch, model=self.network.learning_network, optimizer=self.optimizer, tau=self.config["redo_tau"], re_initialize=self.config["enable_redo"], use_lecun_init=False)

            self.network.learning_network = result["model"]
            self.optimizer = result["optimizer"]
            dormant_fraction = result["dormant_fraction"].item()

        return loss.item(), td_error.mean().item(), td_estimation.mean().item(), dormant_fraction

    def SaveModel(self):
        save_dir = self.model_dir / f"episode_{self.current_episode}"
        save_dir.mkdir(exist_ok=True)

        # Save model parameters and config
        learning_network_path = save_dir / "learning_network.pth"
        target_network_path = save_dir / "target_network.pth"
        config_path = save_dir / "config.pth"

        torch.save({"model": self.network.learning_network.state_dict()}, learning_network_path)
        torch.save({"model": self.network.target_network.state_dict()}, target_network_path)
        torch.save(
            {
                "config": self.config,
                "current_step": self.current_step,
                "current_episode": self.current_episode,
            },
            config_path,
        )

    def LoadModel(self, path):
        config_path = path / "config.pth"
        checkpoint = torch.load(config_path)
        self.config = checkpoint["config"]
        self.current_step = checkpoint["current_step"]
        self.current_episode = checkpoint["current_episode"]

        # If you want to modify the model's configuration,
        # add the configuration to the config here directly.
        # Ex: self.config["batch_size"] = 128

        self.Init()

        learning_network_path = path / "learning_network.pth"
        checkpoint = torch.load(learning_network_path)
        self.network.learning_network.load_state_dict(checkpoint["model"])

        target_network_path = path / "target_network.pth"
        checkpoint = torch.load(target_network_path)
        self.network.target_network.load_state_dict(checkpoint["model"])
