"""Pendulum environment."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from dynax.envs.envs import Env


class PendulumEnv(Env):
    """Pendulum environment."""

    def __init__(self):
        """Initialize the pendulum environment."""
        super().__init__(model_name="pendulum")

    def _reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset pendulum to random initial state."""
        rng, angle_rng, vel_rng = jax.random.split(rng, 3)

        angle = jax.random.uniform(angle_rng, (), minval=-jnp.pi, maxval=jnp.pi)
        velocity = jax.random.uniform(vel_rng, (), minval=-8.0, maxval=8.0)

        return data.replace(
            qpos=jnp.array([angle]),
            qvel=jnp.array([velocity]),
        )

