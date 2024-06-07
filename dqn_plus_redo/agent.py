import torch
import math

ACTION_SIZE = 12

class NoisyLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = torch.nn.Parameter(torch.full((out_features, in_features), sigma_init), requires_grad=True)
        self.sigma_bias = torch.nn.Parameter(torch.full((out_features,), sigma_init), requires_grad=True)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        if self.training:
            weight = self.weight + self.sigma_weight * self.epsilon_weight.data.to(self.weight.device)
            bias = self.bias + self.sigma_bias * self.epsilon_bias.data.to(self.bias.device)
        else:
            weight = self.weight
            bias = self.bias

        output = torch.nn.functional.linear(input, weight, bias)
        return output

    def reset_noise(self):
        epsilon_weight = torch.randn(self.out_features, self.in_features)
        epsilon_bias = torch.randn(self.out_features)
        self.epsilon_weight = torch.nn.Parameter(epsilon_weight, requires_grad=False)
        self.epsilon_bias = torch.nn.Parameter(epsilon_bias, requires_grad=False)

# Define the Double DQN model
class QNetwork(torch.nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=3)
        self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3, stride=2)
        # Noisy Net
        self.fc1 = torch.nn.Linear(14 * 9 * 10, 512)
        self.q = NoisyLinear(512, ACTION_SIZE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        q_values = self.q(x)
        return q_values

    def reset_noise(self):
        # reset the parameters for NoisyNet mechanism
        self.q.reset_noise()


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
