import torch
import math
import torch.nn as nn


class NETWORK(nn.Module):
    def __init__(self, use_dropout=False):
        super(NETWORK, self).__init__()

        # State shape: 4x320x320 (4 frames of 320x320 pixels)
        # Action space: 6

        self.feature_map = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=10, stride=3, padding=0),  # 4x320x320 -> 16x104x104
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=10, stride=3, padding=0),  # 16x104x104 -> 32x32x32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),  # 32x32x32 -> 64x14x14
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0),  # 64x14x14 -> 64x5x5
            nn.LeakyReLU(),
            nn.Flatten(),  # 64x5x5 -> 1600
        )

        # Directly use simple DQN without Dueling DQN
        if use_dropout:
            self.linear = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1600, 512),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 6),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(1600, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 6),
            )

        # Initialize network's parameters
        self.InitNetwork()

    def InitNetwork(self):
        for layer in (self.feature_map, self.linear):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("leaky_relu"))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Transform the range of x from [0, 255] to [0, 1]
        x = x / 255.0

        # DQN network
        feature_map = self.feature_map(x)
        q_value = self.linear(feature_map)

        return q_value


class DQN:
    def __init__(self, tau=0.001):
        # Network
        self.learning_network = None
        self.target_network = None

        self.tau = 0.001
        self.tau_minus = 1.0 - self.tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Init(tau)

    def Init(self, tau=0.001):
        self.learning_network = NETWORK().to(self.device)
        self.target_network = NETWORK().to(self.device)

        self.tau = tau
        self.tau_minus = 1.0 - self.tau

        # Init target_network's parameters
        self.target_network.load_state_dict(self.learning_network.state_dict())

        # Frozen target_network's parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

    def UpdateTargetNetwork(self):
        # Use soft update
        for target_parameter, learning_parameter in zip(self.target_network.parameters(), self.learning_network.parameters()):
            target_parameter.data.copy_(self.tau * learning_parameter.data + self.tau_minus * target_parameter.data)
