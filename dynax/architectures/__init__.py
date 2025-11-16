"""Neural network architectures for dynamics modeling."""

from dynax.architectures.mlp import MLPDynamicsModel
from dynax.architectures.resnet import ResNetDynamicsModel

__all__ = ["MLPDynamicsModel", "ResNetDynamicsModel"]

