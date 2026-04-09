import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat import logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
import torchvision
import vlnce_baselines.models.encoders.clip as clip


class CLIPEncoderB16(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, device,
    ):
        super().__init__()
        self.model, _ = clip.load("ViT-B/16", device=device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

        from torchvision import transforms
        self.rgb_transform = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            )

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """
        rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        rgb_observations = self.rgb_transform(rgb_observations)
        rgb_vector, rgb_grid = self.model.encode_image(rgb_observations.contiguous())

        return rgb_vector.float(), rgb_grid.float()