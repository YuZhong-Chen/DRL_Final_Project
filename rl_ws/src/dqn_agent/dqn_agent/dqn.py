import torch
import math
import torch.nn as nn


class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()

        # State shape: 4x320x320 (4 frames of 320x320 pixels)
        # Action space: 6

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=10, stride=3, padding=0)  # 4x320x320 -> 16x104x104
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=10, stride=3, padding=0)  # 16x104x104 -> 32x32x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)  # 32x32x32 -> 64x14x14
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=0)  # 64x14x14 -> 64x5x5

        self.advantage1 = nn.Linear(1600, 512)
        self.advantage2 = nn.Linear(512, 6)

        self.value1 = nn.Linear(1600, 512)
        self.value2 = nn.Linear(512, 1)

        # Initialize network's parameters
        self.InitNetwork()

    def InitNetwork(self):
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.advantage1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.advantage2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.value1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.xavier_uniform_(self.value2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.conv4.bias, 0)
        nn.init.constant_(self.advantage1.bias, 0)
        nn.init.constant_(self.advantage2.bias, 0)
        nn.init.constant_(self.value1.bias, 0)
        nn.init.constant_(self.value2.bias, 0)

    def forward(self, x):
        # Transform the range of x from [0, 255] to [0, 1]
        x = x / 255.0

        # Feature map
        feature_map = nn.functional.leaky_relu(self.conv1(x))
        feature_map = nn.functional.leaky_relu(self.conv2(feature_map))
        feature_map = nn.functional.leaky_relu(self.conv3(feature_map))
        feature_map = nn.functional.leaky_relu(self.conv4(feature_map))
        feature_map = torch.flatten(feature_map, start_dim=1)
        feature_map = nn.functional.dropout(feature_map, p=0.2)

        # Dueling DQN -> Q(s, a) = V(s) + A(s, a)
        advantage = nn.functional.leaky_relu(self.advantage1(feature_map))
        advantage = nn.functional.dropout(advantage, p=0.2)
        advantage = self.advantage2(advantage)
        value = nn.functional.leaky_relu(self.value1(feature_map))
        value = nn.functional.dropout(value, p=0.2)
        value = self.value2(value)
        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)

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
