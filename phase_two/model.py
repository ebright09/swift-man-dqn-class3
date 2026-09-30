"""The Deep Q-Network: four stacked 84x84 grayscale screens in, one Q-value per action out.

The outputs are *estimated action values*: "how many (clipped, discounted) points do I
expect from here if I press this button now and play well afterwards?" They are not
class probabilities: they don't sum to 1, can be negative, and have no softmax.
The agent simply picks the largest one.
"""
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),   # 84x84 -> 20x20: coarse shapes
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # 20x20 -> 9x9: corridors, sprites
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  # 9x9 -> 7x7: local situations
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),  # 64 * 7 * 7 = 3136
            nn.Linear(512, n_actions))

    def forward(self, screens):
        # Replay keeps uint8 pixels to save memory; scale to 0-1 here, on the device.
        return self.layers(screens.float() / 255.0)


def build_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
