"""Simple MLP architecture for dynamics modeling."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from dynax.base import BaseDynamicsModel, DynamicsModelParams
from dynax.utils.normalization import (
    denormalize_output,
    normalize_action,
    normalize_state,
)


class MLPDynamicsModel(BaseDynamicsModel):
    """Simple MLP for predicting accelerations.

    This model predicts accelerations given state and action,
    then uses semi-implicit Euler integration to compute the next state.

    Attributes:
        state_dim: Total state dimension (nq + nv).
        nq: Number of position dimensions.
        action_dim: Action dimension.
        hidden_dims: Hidden layer dimensions.
        activation: Activation function name ("swish", "relu", or "tanh").
    """

    state_dim: int
    nq: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "swish"

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Predict acceleration from state and action.

        Args:
            state: Current state [q, v], shape (state_dim,).
            action: Current action, shape (action_dim,).

        Returns:
            Predicted acceleration, shape (nv,).
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
            x = nn.Dense(dim, name=f"fc_{i}")(x)
            x = act_fn(x)

        # Output layer predicts accelerations (nv dimensions)
        nv = self.state_dim - self.nq
        acceleration = nn.Dense(nv, name="output")(x)

        return acceleration

    def step(
        self,
        params: DynamicsModelParams,
        state: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """Predict next state using semi-implicit Euler integration.

        Args:
            params: Model parameters with normalization stats and dt.
            state: Current state [q, v], shape (state_dim,).
            action: Current action, shape (action_dim,).

        Returns:
            Next state [q_new, v_new], shape (state_dim,).
        """
        # Split state into position and velocity
        q = state[: self.nq]
        v = state[self.nq :]

        # Normalize state and action
        state_norm = normalize_state(
            state, params.state_mean, params.state_std
        )
        action_norm = normalize_action(
            action, params.action_mean, params.action_std
        )

        # Predict normalized acceleration
        accel_norm = self.apply(
            params.network_params, state_norm, action_norm
        )

        # Denormalize acceleration
        acceleration = denormalize_output(
            accel_norm, params.output_mean, params.output_std
        )

        # Semi-implicit Euler integration
        dt = params.dt
        v_new = v + acceleration * dt
        q_new = q + v_new * dt

        # Concatenate to form next state
        next_state = jnp.concatenate([q_new, v_new])

        return next_state

