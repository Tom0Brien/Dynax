"""Residual architecture that combines physics model with learned correction."""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from mujoco import mjx

from dynax.base import BaseDynamicsModel
from dynax.utils.data import extract_state_features


class ResidualDynamicsModel(BaseDynamicsModel):
    """Residual dynamics model using physics model + learned correction.

    This model uses the underlying physics model (MuJoCo) to predict the next
    state, then learns a residual correction to compensate for modeling
    errors:
    - next_state = physics_model(state, action) + residual(state, action)

    The residual is learned by a neural network, allowing the model to:
    - Leverage known physics structure
    - Learn corrections for unmodeled effects (friction, contact, etc.)
    - Improve sample efficiency by starting from a good baseline

    Attributes:
        env: Environment providing model dimensions and physics model.
        hidden_dims: Hidden layer dimensions for residual network
            (default: (500, 500)).
        activation: Activation function name ("relu", "swish", or "tanh").
    """

    hidden_dims: Tuple[int, ...] = (500, 500)
    activation: str = "relu"

    @nn.compact
    def __call__(
        self, states: jax.Array, actions: jax.Array, training: bool = False
    ) -> jax.Array:
        """Predict residual correction from state-action history.

        Args:
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).
            training: Whether in training mode (unused for MLP).

        Returns:
            Predicted residual correction, shape (state_dim,).
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

        # Use most recent state and action from history
        current_state = states[-1]
        current_action = actions[-1]

        # Concatenate state and action
        x = jnp.concatenate([current_state, current_action], axis=-1)

        # Hidden layers
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"fc{i}")(x)
            x = act_fn(x)

        # Output layer predicts residual correction
        output_dim = self.env.model.nq + self.env.model.nv
        residual = nn.Dense(output_dim, name="output")(x)

        return residual

    def prepare_training_targets(self, dataset) -> jax.Array:
        """Prepare residual targets: true_next_state - physics_next_state.

        Computes the residual correction needed by comparing true next states
        with physics model predictions.

        Args:
            dataset: DynamicsDataset with states, actions, next_states, etc.
                States may be windowed (N, history_length, state_dim) or
                single-step (N, state_dim).

        Returns:
            Residual targets with shape (N, state_dim).
        """
        # Handle windowed states: use most recent state from history
        if dataset.states.ndim == 3:
            # Most recent state
            current_states = dataset.states[:, -1, :]
            # Most recent action
            current_actions = dataset.actions[:, -1, :]
        else:
            current_states = dataset.states
            current_actions = dataset.actions

        true_next_states = dataset.next_states

        # Compute physics predictions using mjx.step
        def physics_step(state, action):
            """Step physics model forward."""
            # Create mjx.Data from state features
            nq = self.env.model.nq
            data = mjx.make_data(self.env.model)
            data = data.replace(
                qpos=state[:nq],
                qvel=state[nq:],
                ctrl=action,
            )
            data = mjx.forward(self.env.model, data)

            # Step physics
            next_data = mjx.step(self.env.model, data)

            # Extract state features
            return extract_state_features(next_data)

        # Vectorize over batch
        physics_next_states = jax.vmap(physics_step)(
            current_states, current_actions
        )

        # Residual = true - physics
        residuals = true_next_states - physics_next_states

        return residuals

    def step(
        self,
        params,
        states: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """Predict next state using physics model + residual correction.

        Args:
            params: Model parameters.
            states: State history, shape (history_length, state_dim).
            actions: Action history, shape (history_length, action_dim).

        Returns:
            Next state, shape (state_dim,).
        """
        from dynax.utils.normalization import (
            denormalize_output,
            normalize_action,
            normalize_state,
        )

        # Use most recent state and action
        current_state = states[-1]
        current_action = actions[-1]

        # Step physics model forward
        nq = self.env.model.nq
        data = mjx.make_data(self.env.model)
        data = data.replace(
            qpos=current_state[:nq],
            qvel=current_state[nq:],
            ctrl=current_action,
        )
        data = mjx.forward(self.env.model, data)
        physics_next_data = mjx.step(self.env.model, data)
        physics_next_state = extract_state_features(physics_next_data)

        # Predict residual correction
        # Normalize state and action
        state_norm = normalize_state(
            current_state, params.state_mean, params.state_std
        )
        action_norm = normalize_action(
            current_action, params.action_mean, params.action_std
        )

        # Create state/action history for model (just single step)
        states_norm = state_norm[None, :]  # (1, state_dim)
        actions_norm = action_norm[None, :]  # (1, action_dim)

        # Predict normalized residual
        residual_norm = self.apply(
            params.network_params, states_norm, actions_norm, training=False
        )

        # Denormalize residual
        residual = denormalize_output(
            residual_norm, params.output_mean, params.output_std
        )

        # Combine physics prediction with residual
        next_state = physics_next_state + residual

        return next_state

