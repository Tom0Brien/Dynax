"""Cart pole environment."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from dynax.envs.envs import Env


class CartPoleEnv(Env):
    """Cart pole environment."""

    def __init__(self):
        """Initialize the cart pole environment."""
        super().__init__(model_name="cart_pole")

    def _reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset cart pole to random initial state."""
        (
            rng,
            cart_rng,
            angle_rng,
            cart_vel_rng,
            angle_vel_rng,
        ) = jax.random.split(rng, 5)

        # Cart position: random in range [-1.5, 1.5] (within rail limits)
        cart_pos = jax.random.uniform(
            cart_rng, (), minval=-1.5, maxval=1.5
        )

        # Pole angle: random in range [-pi, pi]
        angle = jax.random.uniform(
            angle_rng, (), minval=-jnp.pi, maxval=jnp.pi
        )

        # Velocities: random in range [-3, 3]
        cart_vel = jax.random.uniform(
            cart_vel_rng, (), minval=-3.0, maxval=3.0
        )
        angle_vel = jax.random.uniform(
            angle_vel_rng, (), minval=-3.0, maxval=3.0
        )

        return data.replace(
            qpos=jnp.array([cart_pos, angle]),
            qvel=jnp.array([cart_vel, angle_vel]),
        )

