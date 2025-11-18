"""Neural network architectures for dynamics modeling."""

from dynax.architectures.mlp import MLPNeuralModel
from dynax.architectures.residual import ResidualNeuralModel
from dynax.architectures.resnet import ResNetNeuralModel
from dynax.architectures.transformer import TransformerNeuralModel

__all__ = [
    "MLPNeuralModel",
    "ResidualNeuralModel",
    "ResNetNeuralModel",
    "TransformerNeuralModel",
]

