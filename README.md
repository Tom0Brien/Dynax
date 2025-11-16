# Dynax

Training neural network models for robotic systems with [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [JAX](https://jax.readthedocs.io/).

## About

Dynax provides a lightweight, clean framework for training neural network dynamics models from data collected in MuJoCo simulations. It leverages JAX for automatic differentiation and GPU acceleration, and MJX for fast parallel simulation.

Key features:
- Simple, clean API for data collection and model training
- Physics-informed models with semi-implicit Euler integration
- Automatic normalization of states, actions, and outputs
- GPU-accelerated training with JAX
- Modular architecture supporting custom models

## Installation

Using `uv` (recommended):

```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

Using `pip`:

```bash
pip install -e .
```

## Quick Start

Here's a minimal example of training a dynamics model:

```python
import jax
from dynax.architectures import MLPDynamicsModel
from dynax.models import TrainingConfig, train_dynamics_model
from dynax.utils import collect_random_rollouts, create_dataset, split_dataset

# Create your MuJoCo model (mjx.Model)
model = ...

# Collect training data
rng = jax.random.PRNGKey(0)
states, actions, next_states, accelerations = collect_random_rollouts(
    model=model,
    num_rollouts=50,
    rollout_length=100,
    action_min=action_min,
    action_max=action_max,
    rng=rng,
)

# Create and split dataset
dataset = create_dataset(model, states, actions, next_states, accelerations, dt=0.02)
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)

# Create model architecture
dynamics_model = MLPDynamicsModel(
    state_dim=dataset.state_dim,
    nq=dataset.nq,
    action_dim=dataset.action_dim,
    hidden_dims=(128, 128),
    activation="swish",
)

# Train model
config = TrainingConfig(num_epochs=100, batch_size=128)
trained_params = train_dynamics_model(
    model=dynamics_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
    rng=rng,
)

# Use trained model
next_state = dynamics_model.step(trained_params, state, action)
```

## Examples

Run the pendulum example:

```bash
python examples/train_pendulum.py
```

This will:
1. Create a simple pendulum model
2. Collect random rollout data
3. Train an MLP dynamics model
4. Evaluate single-step prediction accuracy

## Project Structure

```
dynax/
├── dynax/              # Source code
│   ├── models/         # Base model classes and training utilities
│   │   ├── base.py           # BaseDynamicsModel and DynamicsModelParams
│   │   └── training.py       # Training loop and utilities
│   ├── utils/          # Utility functions
│   │   ├── data.py           # Data collection and dataset management
│   │   └── normalization.py # Normalization utilities
│   └── architectures/  # Neural network architectures
│       └── mlp.py            # Simple MLP model
├── examples/           # Example scripts
│   └── train_pendulum.py
└── README.md
```

## Creating Custom Architectures

To create a custom dynamics model, inherit from `BaseDynamicsModel`:

```python
from dynax.models import BaseDynamicsModel
from flax import linen as nn

class MyCustomModel(BaseDynamicsModel):
    state_dim: int
    nq: int
    action_dim: int
    
    @nn.compact
    def __call__(self, state, action):
        # Predict accelerations
        x = jnp.concatenate([state, action])
        x = nn.Dense(256)(x)
        x = nn.swish(x)
        acceleration = nn.Dense(self.state_dim - self.nq)(x)
        return acceleration
    
    def step(self, params, state, action):
        # Override if you need custom integration
        # Default uses semi-implicit Euler
        return super().step(params, state, action)
```

## Key Concepts

### Physics-Informed Models

Dynax models predict **accelerations** rather than state deltas, which are then integrated using semi-implicit Euler:

```
v_new = v + acceleration * dt
q_new = q + v_new * dt
```

This approach:
- Respects the underlying physics structure
- Improves long-horizon prediction accuracy
- Reduces the dimensionality of the output (nv vs state_dim)

### Normalization

All inputs and outputs are automatically normalized during training using dataset statistics. This improves training stability and convergence.

### Dataset Structure

The `DynamicsDataset` stores:
- `states`: [q, v] concatenated
- `actions`: Control inputs
- `next_states`: Next [q, v]
- `accelerations`: qacc from MuJoCo (used as training targets)

## Advanced Usage

### Custom Data Collection

```python
from dynax.utils import extract_state_features

# Define custom initial state sampler
def my_initial_state_sampler(model, rng):
    state = mjx.make_data(model)
    # Customize initial state here
    return mjx.forward(model, state)

# Use in data collection
states, actions, next_states, accelerations = collect_random_rollouts(
    model=model,
    num_rollouts=100,
    rollout_length=200,
    action_min=action_min,
    action_max=action_max,
    rng=rng,
    initial_state_sampler=my_initial_state_sampler,
)
```

### Training Configuration

```python
from dynax.models import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=100,
    weight_decay=1e-4,
    grad_clip=1.0,  # Optional gradient clipping
)
```

## License

MIT

## Citation

If you use Dynax in your research, please cite:

```
@misc{dynax2024,
  title={Dynax: Neural dynamics models with MuJoCo MJX},
  author={Your Name},
  year={2024},
  note={https://github.com/yourusername/dynax}
}
```

