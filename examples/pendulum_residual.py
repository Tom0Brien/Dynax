"""Train residual dynamics model for pendulum with varying friction/mass."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import ResidualDynamicsModel
from dynax.envs import PendulumEnv
from dynax.evaluation import create_rollout_fn, create_true_rollout_fn
from dynax.utils import collect_and_prepare_data
from dynax.utils.data import extract_state_features


class PendulumEnvWithParams(PendulumEnv):
    """Pendulum environment with configurable mass and friction."""

    def __init__(self, mass: float = None, friction: float = None):
        """Initialize pendulum with optional mass and friction parameters.

        Args:
            mass: Pendulum mass multiplier (default: 1.0, uses model default).
            friction: Joint friction loss (default: uses model default).
        """
        # Load base model manually to modify before creating MJX model
        from dynax.envs.envs import MODELS_DIR

        model_dir = MODELS_DIR / "pendulum"
        xml_path = model_dir / "pendulum.xml"
        if not xml_path.exists():
            xml_path = model_dir / "scene.xml"

        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model_name = "pendulum"

        # Modify mass if specified
        if mass is not None:
            # Find pendulum body (usually body 1, after world body)
            # For pendulum, typically the bob is body 1
            if self.mj_model.nbody > 1:
                self.mj_model.body_mass[1] = (
                    self.mj_model.body_mass[1] * mass
                )
                # Also scale inertia proportionally (inertia is 3D vector)
                self.mj_model.body_inertia[1, :] = (
                    self.mj_model.body_inertia[1, :] * mass
                )

        # Modify friction if specified
        if friction is not None:
            # Set DOF friction loss (stored per DOF, not per joint)
            # For pendulum, there's typically one DOF (nv=1)
            if self.mj_model.nv > 0:
                self.mj_model.dof_frictionloss[:] = friction

        # Create MJX model after modifications
        self.model = mjx.put_model(self.mj_model)

        # Extract action bounds
        self.action_min = jnp.where(
            self.mj_model.actuator_ctrllimited,
            self.mj_model.actuator_ctrlrange[:, 0],
            jnp.full((self.mj_model.nu,), -1.0),
        )
        self.action_max = jnp.where(
            self.mj_model.actuator_ctrllimited,
            self.mj_model.actuator_ctrlrange[:, 1],
            jnp.full((self.mj_model.nu,), 1.0),
        )

        # Timestep
        self.dt = float(self.mj_model.opt.timestep)




# Create environment with varied parameters for data collection
# This single varied model will be used for both data collection and prediction
varied_mass = 1.2  # 20% heavier than default
varied_friction = 0.05  # Some friction loss
varied_env = PendulumEnvWithParams(mass=varied_mass, friction=varied_friction)

print(
    f"Using varied parameters: mass={varied_mass:.2f}, "
    f"friction={varied_friction:.3f}"
)

# Collect data with the varied model using random actions
# (Using random actions avoids model mismatch issues with hydrax controller)
rng = jax.random.PRNGKey(0)
print("\nCollecting data with varied parameters...")
train_dataset, val_dataset = collect_and_prepare_data(
    env=varied_env,
    num_rollouts=500,
    rollout_length=100,
    rng=rng,
    dataset_path="data/pendulum_varied_dataset.pkl",
    controller=None,  # Use random actions only
    num_controlled_rollouts=0,
    force_recollect=False,
)

print(
    f"\nCollected {len(train_dataset)} training samples and "
    f"{len(val_dataset)} validation samples"
)

# Use the BASE model for the residual model physics baseline
# The residual model will use the base physics model, and the NN will learn
# to correct for the difference between base model and true dynamics
# (which has varied parameters)
base_env = PendulumEnv()
dynamics_model = ResidualDynamicsModel(
    env=base_env,
    hidden_dims=(500, 500),
    activation="relu",
)

rng, train_rng = jax.random.split(rng)
trained_params = train_dynamics_model(
    model=dynamics_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(
        num_epochs=200,
        batch_size=512,
        learning_rate=1e-3,
        noise_std=0.0,  # Disable noise
        log_dir="logs/pendulum_residual",
        render_videos=False,
    ),
    rng=train_rng,
    env=base_env,
    eval_num_rollouts=5,
)

# Compare learned model vs base physics model
print("\n" + "=" * 80)
print("Comparing Learned Residual Model vs Base Physics Model")
print("=" * 80)

# Create rollout functions
learned_rollout_fn = create_rollout_fn(dynamics_model, trained_params)
base_physics_rollout_fn = create_true_rollout_fn(base_env)
true_physics_rollout_fn = create_true_rollout_fn(varied_env)

# Generate test rollouts
num_test_rollouts = 10
rollout_length = 100
rng, test_rng = jax.random.split(rng)
test_rngs = jax.random.split(test_rng, num_test_rollouts)

# Create initial states
def create_initial_state_history(rng_key, env):
    """Create initial state history for rollout."""
    data = mjx.make_data(env.model)
    data = env.reset(data, rng_key)
    data = mjx.forward(env.model, data)
    initial_state = extract_state_features(data)
    # Repeat for history length
    history_length = dynamics_model.history_length
    return jnp.repeat(initial_state[None, :], history_length, axis=0)

# Generate random actions
rng, action_rng = jax.random.split(rng)
actions_batch = jax.random.uniform(
    action_rng,
    (num_test_rollouts, rollout_length, varied_env.model.nu),
    minval=varied_env.action_min,
    maxval=varied_env.action_max,
)

# Run rollouts
all_true_trajs = []
all_base_trajs = []
all_learned_trajs = []

for i in range(num_test_rollouts):
    # Create initial state history
    initial_state_history = create_initial_state_history(
        test_rngs[i], varied_env
    )

    # Create initial MJX data for physics rollouts
    initial_state = initial_state_history[-1, :]
    nq = varied_env.model.nq

    # True physics (varied model)
    true_data = mjx.make_data(varied_env.model)
    true_data = true_data.replace(
        qpos=initial_state[:nq],
        qvel=initial_state[nq:],
    )
    true_data = mjx.forward(varied_env.model, true_data)

    # Base physics (base model)
    base_data = mjx.make_data(base_env.model)
    base_data = base_data.replace(
        qpos=initial_state[:nq],
        qvel=initial_state[nq:],
    )
    base_data = mjx.forward(base_env.model, base_data)

    # Run rollouts
    true_traj = true_physics_rollout_fn(true_data, actions_batch[i])
    base_traj = base_physics_rollout_fn(base_data, actions_batch[i])
    learned_traj = learned_rollout_fn(
        initial_state_history, actions_batch[i]
    )

    all_true_trajs.append(true_traj)
    all_base_trajs.append(base_traj)
    all_learned_trajs.append(learned_traj)

# Convert to numpy arrays
true_trajs = np.array(all_true_trajs)
base_trajs = np.array(all_base_trajs)
learned_trajs = np.array(all_learned_trajs)

# Compute statistics
base_errors = np.abs(base_trajs - true_trajs)
learned_errors = np.abs(learned_trajs - true_trajs)

base_mae_over_time = np.mean(base_errors, axis=(0, 2))
learned_mae_over_time = np.mean(learned_errors, axis=(0, 2))

base_rmse_over_time = np.sqrt(
    np.mean(np.square(base_trajs - true_trajs), axis=(0, 2))
)
learned_rmse_over_time = np.sqrt(
    np.mean(np.square(learned_trajs - true_trajs), axis=(0, 2))
)

base_final_mae = np.mean(base_errors[:, -1, :])
learned_final_mae = np.mean(learned_errors[:, -1, :])

base_final_rmse = np.sqrt(
    np.mean(np.square(base_trajs[:, -1, :] - true_trajs[:, -1, :]))
)
learned_final_rmse = np.sqrt(
    np.mean(np.square(learned_trajs[:, -1, :] - true_trajs[:, -1, :]))
)

print(f"\nStatistics (averaged over {num_test_rollouts} rollouts):")
print("\nBase Physics Model (with modeling error):")
print(f"  Final MAE:  {base_final_mae:.6f}")
print(f"  Final RMSE: {base_final_rmse:.6f}")
print(f"  Mean MAE over time: {np.mean(base_mae_over_time):.6f}")
print(f"  Mean RMSE over time: {np.mean(base_rmse_over_time):.6f}")

print("\nLearned Residual Model:")
print(f"  Final MAE:  {learned_final_mae:.6f}")
print(f"  Final RMSE: {learned_final_rmse:.6f}")
print(f"  Mean MAE over time: {np.mean(learned_mae_over_time):.6f}")
print(f"  Mean RMSE over time: {np.mean(learned_rmse_over_time):.6f}")

print("\nImprovement:")
mae_reduction = (1 - learned_final_mae / base_final_mae) * 100
rmse_reduction = (1 - learned_final_rmse / base_final_rmse) * 100
print(f"  MAE reduction:  {mae_reduction:.2f}%")
print(f"  RMSE reduction: {rmse_reduction:.2f}%")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
time_steps = np.arange(rollout_length + 1) * varied_env.dt

# Plot 1: MAE over time
axes[0, 0].plot(
    time_steps, base_mae_over_time, label="Base Physics", linewidth=2
)
axes[0, 0].plot(
    time_steps,
    learned_mae_over_time,
    label="Learned Residual",
    linewidth=2,
)
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Mean Absolute Error")
axes[0, 0].set_title("MAE Over Time")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: RMSE over time
axes[0, 1].plot(
    time_steps, base_rmse_over_time, label="Base Physics", linewidth=2
)
axes[0, 1].plot(
    time_steps,
    learned_rmse_over_time,
    label="Learned Residual",
    linewidth=2,
)
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Root Mean Squared Error")
axes[0, 1].set_title("RMSE Over Time")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sample trajectory (position)
traj_idx = 0
axes[1, 0].plot(
    time_steps,
    true_trajs[traj_idx, :, 0],
    label="True",
    linewidth=2,
    linestyle="--",
)
axes[1, 0].plot(
    time_steps,
    base_trajs[traj_idx, :, 0],
    label="Base Physics",
    linewidth=1.5,
    alpha=0.7,
)
axes[1, 0].plot(
    time_steps,
    learned_trajs[traj_idx, :, 0],
    label="Learned Residual",
    linewidth=1.5,
    alpha=0.7,
)
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Position (q)")
axes[1, 0].set_title(f"Sample Trajectory: Position (Rollout {traj_idx+1})")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Sample trajectory (velocity)
axes[1, 1].plot(
    time_steps,
    true_trajs[traj_idx, :, 1],
    label="True",
    linewidth=2,
    linestyle="--",
)
axes[1, 1].plot(
    time_steps,
    base_trajs[traj_idx, :, 1],
    label="Base Physics",
    linewidth=1.5,
    alpha=0.7,
)
axes[1, 1].plot(
    time_steps,
    learned_trajs[traj_idx, :, 1],
    label="Learned Residual",
    linewidth=1.5,
    alpha=0.7,
)
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Velocity (v)")
axes[1, 1].set_title(f"Sample Trajectory: Velocity (Rollout {traj_idx+1})")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "logs/pendulum_residual/comparison_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {plot_path}")
plt.close()

