"""Double cart pole environment."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from dynax.envs.envs import Env


class DoubleCartPoleEnv(Env):
    """Double cart pole environment."""

    def __init__(self):
        """Initialize the double cart pole environment."""
        super().__init__(model_name="double_cart_pole")

    def _reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset double cart pole to random initial state."""
        (
            rng,
            cart_rng,
            angle1_rng,
            angle2_rng,
            cart_vel_rng,
            angle1_vel_rng,
            angle2_vel_rng,
        ) = jax.random.split(rng, 7)

        # Cart position: random in range [-2, 2]
        cart_pos = jax.random.uniform(
            cart_rng, (), minval=-2.0, maxval=2.0
        )

        # Pole angles: random in range [-pi, pi]
        angle1 = jax.random.uniform(
            angle1_rng, (), minval=-jnp.pi, maxval=jnp.pi
        )
        angle2 = jax.random.uniform(
            angle2_rng, (), minval=-jnp.pi, maxval=jnp.pi
        )

        # Velocities: random in range [-5, 5]
        cart_vel = jax.random.uniform(
            cart_vel_rng, (), minval=-5.0, maxval=5.0
        )
        angle1_vel = jax.random.uniform(
            angle1_vel_rng, (), minval=-5.0, maxval=5.0
        )
        angle2_vel = jax.random.uniform(
            angle2_vel_rng, (), minval=-5.0, maxval=5.0
        )

        return data.replace(
            qpos=jnp.array([cart_pos, angle1, angle2]),
            qvel=jnp.array([cart_vel, angle1_vel, angle2_vel]),
        )

