import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class Generator(ModelBase):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.a * x


class Discriminator(ModelBase):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.a * x
