"""Train residual dynamics model for pendulum with varying friction/mass."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

from dynax import TrainingConfig, train_dynamics_model
from dynax.architectures import ResidualNeuralModel
from dynax.envs import PendulumEnv
from dynax.evaluation import create_rollout_fn, create_true_rollout_fn
from dynax.mpc import NeuralTask
from dynax.utils import (
    collect_and_prepare_data,
    evaluate_mpc_controllers,
    render_mpc_trajectories,
)
from dynax.utils.data import extract_state_features
from hydrax.algs import PredictiveSampling
from hydrax.tasks.pendulum import Pendulum as HydraxPendulumTask


class PendulumEnvVaried(PendulumEnv):
    """Pendulum environment with hardcoded varied mass and friction."""

    mass: float = 1.25
    friction: float = 0.1

    def _load_model(self, model_name: str, use_scene: bool) -> mujoco.MjModel:
        """Load and modify the model with varied parameters."""
        # Load the base model using parent's method
        model = super()._load_model(model_name, use_scene)

        # Modify mass
        # Find pendulum body (usually body 1, after world body)
        # For pendulum, typically the bob is body 1
        if model.nbody > 1:
            model.body_mass[1] = model.body_mass[1] * self.mass
            # Also scale inertia proportionally (inertia is 3D vector)
            model.body_inertia[1, :] = model.body_inertia[1, :] * self.mass

        # Modify friction
        # Set DOF friction loss (stored per DOF, not per joint)
        # For pendulum, there's typically one DOF (nv=1)
        if model.nv > 0:
            model.dof_frictionloss[:] = self.friction

        return model


def train_model(model_path: str = "models/pendulum_residual.pkl"):
    """Train a residual dynamics model and save it to disk.

    Args:
        model_path: Path to save the trained model.
    """
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    # Create environment with varied parameters for data collection
    # This single varied model will be used for both data collection
    # and prediction
    varied_env = PendulumEnvVaried()

    print(
        f"Using varied parameters: mass={varied_env.mass}, "
        f"friction={varied_env.friction}"
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
        force_recollect=True,
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
    dynamics_model = ResidualNeuralModel(
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
        # Use varied_env for evaluation to compare against true dynamics
        env=varied_env,
        eval_num_rollouts=5,
    )

    # Save model
    print(f"\nSaving model to {model_path}...")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dynamics_model.save_model(trained_params, model_path)
    print(f"Model saved successfully to {model_path}")

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

    print("=" * 60)


def eval_model(model_path: str = "models/pendulum_residual.pkl"):
    """Evaluate MPC performance with base vs learned models.

    Args:
        model_path: Path to the saved model.
    """
    print("=" * 60)
    print("EVALUATION MODE - MPC Performance Comparison")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")

    # Create environments
    varied_env = PendulumEnvVaried()
    base_env = PendulumEnv()

    # Create model architecture (must match training)
    dynamics_model = ResidualNeuralModel(
        env=base_env,
        hidden_dims=(500, 500),
        activation="relu",
    )

    # Load model parameters
    print("Loading model parameters...")
    model_params = dynamics_model.load_model(model_path_obj)

    # Create tasks
    base_task = HydraxPendulumTask()
    neural_task = NeuralTask(
        base_task=base_task,
        dynamics_model=dynamics_model,
        model_params=model_params,
    )

    # Create MPC controllers
    print("Creating MPC controllers...")
    base_ctrl = PredictiveSampling(
        base_task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    learned_ctrl = PredictiveSampling(
        neural_task,
        num_samples=32,
        noise_level=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    # Evaluation parameters
    num_episodes = 20
    episode_length = 250  # steps
    rng = jax.random.PRNGKey(42)

    print(f"\nRunning {num_episodes} episodes in parallel...")
    print("(Using true varied dynamics for simulation)")

    # Create initial states for all episodes
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_episodes)

    def reset_episode(rng_key):
        """Reset a single episode."""
        data = mjx.make_data(varied_env.model)
        data = varied_env.reset(data, rng_key)
        data = mjx.forward(varied_env.model, data)
        return data

    initial_states = jax.vmap(reset_episode)(reset_rngs)

    # Evaluate controllers and generate report
    print("Running episodes with base physics MPC...")
    print("Running episodes with learned neural MPC...")
    evaluate_mpc_controllers(
        base_ctrl,
        learned_ctrl,
        initial_states,
        base_task,
        neural_task,
        varied_env,
        base_env,
        episode_length,
        varied_env.dt,
        rng,
        plot_path="logs/pendulum_residual/mpc_comparison.png",
        replan_freq_hz=10.0,
    )

    # Render trajectory videos
    print("\nRendering videos for visualization...")
    render_mpc_trajectories(
        base_ctrl,
        learned_ctrl,
        initial_states,
        base_task,
        neural_task,
        varied_env,
        base_env,
        episode_length,
        varied_env.dt,
        rng,
        "logs/pendulum_residual",
        num_videos=1,
        replan_freq_hz=10.0,
    )

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate a residual dynamics model for pendulum."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="models/pendulum_residual.pkl",
        help="Path to save the trained model",
    )

    # Eval mode
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="models/pendulum_residual.pkl",
        help="Path to the saved model",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(model_path=args.model_path)
    elif args.mode == "eval":
        eval_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
