"""ResNet architecture for dynamics modeling with residual connections."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynax.base import BaseDynamicsModel


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection.

    Attributes:
        dim: Dimension of the hidden layers.
        activation: Activation function name.
    """

    dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply residual block transformation.

        Args:
            x: Input tensor, shape (..., dim).

        Returns:
            Output tensor with residual connection, shape (..., dim).
        """
        # Get activation function
        if self.activation == "swish":
            act_fn = nn.swish
        elif self.activation == "relu":
            act_fn = nn.relu
        elif self.activation == "tanh":
            act_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # First layer
        residual = x
        x = nn.Dense(self.dim, name="fc1")(x)
        x = act_fn(x)

        # Second layer (no activation, will be added after skip)
        x = nn.Dense(self.dim, name="fc2")(x)

        # Skip connection and activation
        x = act_fn(x + residual)

        return x


class ResNetDynamicsModel(BaseDynamicsModel):
    """ResNet architecture predicting state deltas with residual connections.

    Architecture:
    - Input: [state, action] concatenation
    - Initial projection layer
    - Multiple residual blocks with skip connections
    - Output layer predicts state delta

    The model predicts Δs = f(s, a), and next_state = s + Δs.

    Attributes:
        env: Environment providing model dimensions.
        hidden_dim: Hidden dimension for all layers (default: 500).
        num_blocks: Number of residual blocks (default: 2).
        activation: Activation function name ("relu", "swish", or "tanh").
    """

    hidden_dim: int = 500
    num_blocks: int = 2
    activation: str = "relu"

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Predict state delta from state and action.

        Args:
            state: Current state, shape (state_dim,).
            action: Current action, shape (action_dim,).

        Returns:
            Predicted state delta, shape (state_dim,).
        """
        # Get activation function
        if self.activation == "swish":
            act_fn = nn.swish
        elif self.activation == "relu":
            act_fn = nn.relu
        elif self.activation == "tanh":
            act_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)

        # Initial projection to hidden dimension
        x = nn.Dense(self.hidden_dim, name="input_proj")(x)
        x = act_fn(x)

        # Residual blocks
        for i in range(self.num_blocks):
            x = ResidualBlock(
                dim=self.hidden_dim,
                activation=self.activation,
                name=f"res_block_{i}",
            )(x)

        # Output layer predicts state delta
        state_dim = self.env.model.nq + self.env.model.nv
        delta = nn.Dense(state_dim, name="output")(x)

        return delta

