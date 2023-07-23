import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

class DenseModel(nn.Module):
    def __init__(
        self,
        feature_size: int,
        output_shape: tuple,
        layers: int,
        hidden_size: int,
        dist="normal_fixed_std",
        min_std: float = 0.0,
        activation=nn.ELU,
    ):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self._min_std = min_std
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        if self._dist == "normal_fixed_std":
            model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        elif self._dist == "normal_learned_std":
            model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)) * 2)]
        elif self._dist == "deterministic":
            model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, *features):
        x = self.model(torch.cat(features, -1))
        if self._dist == "normal_learned_std":
            mean, std = torch.chunk(x, 2, -1)
            std = F.softplus(std) + self._min_std
            return td.independent.Independent(
                td.Normal(mean, std), len(self._output_shape)
            )
        elif self._dist == "normal_fixed_std":
            mean = x
            return td.independent.Independent(
                td.Normal(mean, 1), len(self._output_shape)
            )
        elif self._dist == "binary":
            return td.independent.Independent(
                td.Bernoulli(logits=x), len(self._output_shape)
            )
        elif self._dist == "deterministic":
            return x
        else:
            raise NotImplementedError(self._dist)
