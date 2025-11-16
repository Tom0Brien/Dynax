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
    def __call__(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        """Predict state delta from state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).

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

        # Interleave state-action pairs: [s_0, a_0, s_1, a_1, ..., s_h, a_h]
        # This makes it easier for the model to associate states with actions
        # Reshape to (history_length, state_dim + action_dim) then flatten
        state_action_pairs = jnp.concatenate([states, actions], axis=-1)
        x = state_action_pairs.flatten()

        # Hidden layers
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"fc{i}")(x)
            x = act_fn(x)

        # Output layer predicts state delta
        output_dim = self.env.model.nq + self.env.model.nv
        delta = nn.Dense(output_dim, name="output")(x)

        return delta
