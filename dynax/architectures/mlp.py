"""Simple MLP architecture for dynamics modeling (matching MBRL paper)."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynax.base import BaseDynamicsModel


class MLPDynamicsModel(BaseDynamicsModel):
    """Simple feedforward MLP predicting state deltas.

    Architecture from "Neural Network Dynamics for Model-Based Deep RL":
    - Input: [state, action] concatenation
    - Hidden layers with ReLU activation
    - Output: state delta (next_state - state)

    The model predicts Δs = f(s, a), and next_state = s + Δs.

    Attributes:
        env: Environment providing model dimensions.
        hidden_dims: Hidden layer dimensions (default: (500, 500) from paper).
        activation: Activation function name ("relu", "swish", or "tanh").
    """

    hidden_dims: Tuple[int, ...] = (500, 500)
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

        # Hidden layers
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"fc{i}")(x)
            x = act_fn(x)

        # Output layer predicts state delta
        delta = nn.Dense(self.env.model.nq + self.env.model.nv, name="output")(x)

        return delta
