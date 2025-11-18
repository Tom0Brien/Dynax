"""Neural task wrapper for hydrax that uses learned dynamics models."""

from pathlib import Path
from typing import Optional

import jax
from mujoco import mjx

from dynax.base import BaseDynamicsModel, DynamicsModelParams
from dynax.utils.data import extract_state_features
from hydrax.task_base import Task


class NeuralTask(Task):
    """A hydrax Task wrapper that uses a learned neural dynamics model.

    This class wraps a hydrax Task and overrides the step function to use
    a neural dynamics model instead of the default MuJoCo physics. This
    enables MPC planning with learned models.

    The neural model must have history_length=1 for hydrax compatibility.
    """

    def __init__(
        self,
        base_task: Task,
        dynamics_model: BaseDynamicsModel,
        model_params: Optional[DynamicsModelParams] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the neural task wrapper.

        Args:
            base_task: The base hydrax Task to wrap (provides cost functions).
            dynamics_model: The neural dynamics model to use for stepping.
            model_params: Optional pre-loaded model parameters. If None and
                model_path is provided, parameters will be loaded from disk.
            model_path: Optional path to saved model parameters file.

        Raises:
            ValueError: If dynamics_model has history_length != 1.
            FileNotFoundError: If model_path is provided but file doesn't exist.
        """
        # Validate history_length
        if dynamics_model.history_length != 1:
            raise ValueError(
                f"NeuralTask only supports history_length=1 for hydrax "
                f"compatibility, got {dynamics_model.history_length}"
            )

        # Initialize base Task with the same MuJoCo model
        # Get trace site names from base task if available
        trace_site_names = None
        if len(base_task.trace_site_ids) > 0:
            # Convert site IDs back to names
            trace_site_names = [
                base_task.mj_model.site(i).name
                for i in base_task.trace_site_ids
            ]
        super().__init__(base_task.mj_model, trace_sites=trace_site_names)

        self.base_task = base_task
        self.dynamics_model = dynamics_model

        # Load or set model parameters
        if model_params is not None:
            self.model_params = model_params
        elif model_path is not None:
            self.model_params = self.load_model(model_path)
        else:
            raise ValueError(
                "Either model_params or model_path must be provided"
            )

        # JIT-compile the step function for performance
        self._step_fn = jax.jit(self._neural_step)

    def _neural_step(
        self, state_features: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Step the neural dynamics model forward.

        Args:
            state_features: Current state as [qpos, qvel], shape (state_dim,).
            action: Control action, shape (action_dim,).

        Returns:
            Next state features as [qpos, qvel], shape (state_dim,).
        """
        # Reshape for model: (state_dim,) -> (1, state_dim) for history_length=1
        states = state_features[None, :]  # (1, state_dim)
        actions = action[None, :]  # (1, action_dim)

        # Predict next state using neural model
        next_state = self.dynamics_model.step(
            self.model_params, states, actions
        )

        return next_state

    def step(self, model: mjx.Model, state: mjx.Data) -> mjx.Data:
        """Custom step function using neural dynamics model.

        Overrides the default MuJoCo step to use the learned neural model.

        Args:
            model: The MuJoCo MJX model (unused, kept for interface
                compatibility).
            state: The current state xₜ.

        Returns:
            The next state xₜ₊₁ after applying neural dynamics.
        """
        # Extract state features and action
        state_features = extract_state_features(state)
        action = state.ctrl

        # Step neural model
        next_state_features = self._step_fn(state_features, action)

        # Convert back to mjx.Data
        nq = self.model.nq
        next_state = state.replace(
            qpos=next_state_features[:nq],
            qvel=next_state_features[nq:],
            time=state.time + self.dt,
        )

        # Forward kinematics to update dependent quantities
        next_state = mjx.forward(self.model, next_state)

        return next_state

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Delegate running cost to base task."""
        return self.base_task.running_cost(state, control)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Delegate terminal cost to base task."""
        return self.base_task.terminal_cost(state)

    def domain_randomize_model(
        self, rng: jax.Array
    ) -> dict[str, jax.Array]:
        """Delegate domain randomization to base task."""
        return self.base_task.domain_randomize_model(rng)

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> dict[str, jax.Array]:
        """Delegate domain randomization to base task."""
        return self.base_task.domain_randomize_data(data, rng)

    def save_model(self, path: str | Path) -> None:
        """Save the model parameters to disk.

        Args:
            path: Path to save the model parameters (uses .pkl extension
                if not provided).
        """
        self.dynamics_model.save_model(self.model_params, path)

    @staticmethod
    def load_model(path: str | Path) -> DynamicsModelParams:
        """Load model parameters from disk.

        Args:
            path: Path to the saved model parameters file.

        Returns:
            Loaded DynamicsModelParams.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        return BaseDynamicsModel.load_model(path)

