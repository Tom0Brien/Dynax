# Dynax

Training neural network dynamics models for robotic systems with [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [JAX](https://jax.readthedocs.io/).

![Cart Pole Trajectory](img/cart_pole_trajectory.gif)

## Installation

```bash
uv sync
uv pip install -e .
```

## Quick Start

```python
import jax
from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import MLPDynamicsModel
from dynax.envs import PendulumEnv
from dynax.utils import collect_and_prepare_data

# Create environment
env = PendulumEnv()

# Collect data
rng = jax.random.PRNGKey(0)
train_dataset, val_dataset = collect_and_prepare_data(
    env=env,
    num_rollouts=50,
    rollout_length=100,
    rng=rng,
)

# Create model
model = MLPDynamicsModel(env=env, hidden_dims=(500, 500))

# Train
trained_params = train_dynamics_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(num_epochs=100, batch_size=512),
    rng=rng,
    env=env,
)
```

## Examples

```bash
python examples/train_pendulum.py
python examples/train_cart_pole.py
```

## Model Formulation

Dynax models predict **state deltas** and use residual dynamics:

$$
s_{t+1} = s_t + f_\theta(s_t, a_t)
$$

where $f_\theta$ is the learned dynamics model predicting $\Delta s_t = s_{t+1} - s_t$.

All inputs/outputs are automatically normalized during training.

## Architectures

- **MLPDynamicsModel**: Simple feedforward MLP with configurable hidden layers
- **ResNetDynamicsModel**: Residual network architecture with skip connections
- **TransformerDynamicsModel**: GPT-2 style transformer with causal self-attention

## Custom Models

```python
from dynax.base import BaseDynamicsModel
from flax import linen as nn

class MyModel(BaseDynamicsModel):
    @nn.compact
    def __call__(self, states, actions, training=False):
        x = jnp.concatenate([states.flatten(), actions.flatten()])
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(self.state_dim)(x)
```

## TODO

- [ ] Improve performance
- [ ] Support contact information in input
- [ ] Support robot centric state representation

## License

MIT

