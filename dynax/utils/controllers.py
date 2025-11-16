"""Controller interfaces for data collection."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx


class Controller(ABC):
    """Generic controller interface for data collection.

    Controllers must be functional (stateless) to work with JAX's vmap/jit.
    State is managed externally and passed through scan carries.
    """

    @abstractmethod
    def init_state(self, rng: jax.Array) -> Any:
        """Initialize controller state for a rollout.

        Args:
            rng: Random number generator key.

        Returns:
            Initial controller state (can be any JAX-compatible type).
        """
        pass

    @abstractmethod
    def get_action(
        self, state: mjx.Data, controller_state: Any
    ) -> Tuple[jax.Array, Any]:
        """Compute control action and update controller state.

        Args:
            state: Current mujoco state (mjx.Data).
            controller_state: Current controller state.

        Returns:
            Tuple of (action, new_controller_state).
            Action shape: (nu,).
        """
        pass


class RandomController(Controller):
    """Random controller that samples actions uniformly."""

    def __init__(self, action_min: jax.Array, action_max: jax.Array):
        """Initialize random controller.

        Args:
            action_min: Minimum action values, shape (nu,).
            action_max: Maximum action values, shape (nu,).
        """
        self.action_min = action_min
        self.action_max = action_max

    def init_state(self, rng: jax.Array) -> jax.Array:
        """Initialize RNG state."""
        return rng

    def get_action(
        self, state: mjx.Data, controller_state: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample random action."""
        rng, subrng = jax.random.split(controller_state)
        action = jax.random.uniform(
            subrng,
            (len(self.action_min),),
            minval=self.action_min,
            maxval=self.action_max,
        )
        return action, rng


class HydraxController(Controller):
    """Wrapper for hydrax SamplingBasedController.

    Adapts hydrax controllers to the generic Controller interface.
    """

    def __init__(self, hydrax_controller):
        """Initialize hydrax controller wrapper.

        Args:
            hydrax_controller: A hydrax SamplingBasedController instance.
        """
        self.hydrax_controller = hydrax_controller
        # JIT compile controller methods once
        self._jit_optimize = jax.jit(hydrax_controller.optimize)
        self._jit_get_action = jax.jit(hydrax_controller.get_action)

    def init_state(self, rng: jax.Array) -> Any:
        """Initialize hydrax controller parameters."""
        params = self.hydrax_controller.init_params(seed=0)
        return params.replace(rng=rng)

    def get_action(
        self, state: mjx.Data, controller_state: Any
    ) -> Tuple[jax.Array, Any]:
        """Get action from hydrax controller with warm-starting.

        Args:
            state: Current mujoco state.
            controller_state: Controller parameters (from hydrax).

        Returns:
            Tuple of (action, updated_controller_state).
        """
        # Run MPC optimization (with warm-starting from previous step)
        params, _ = self._jit_optimize(state, controller_state)

        # Get action at current time
        action = self._jit_get_action(params, state.time)
        action = jnp.clip(
            action,
            self.hydrax_controller.task.u_min,
            self.hydrax_controller.task.u_max,
        )

        return action, params

