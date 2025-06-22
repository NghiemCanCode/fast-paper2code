import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        self._name = "undefined_model"

    def forward(self, x):
        pass

    @property
    @abstractmethod
    def name(self):
        """Subclass must define the name of the model."""
        pass

    def __str__(self):
        """
        Model prints with a number of trainable parameters
        """
        name_str = "Model: " + self.name() + "\n"
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return name_str + super().__str__() + '\nTrainable parameters: {}'.format(params)