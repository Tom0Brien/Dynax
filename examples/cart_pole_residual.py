"""Train residual dynamics model for cart pole with varying friction/mass."""

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
from dynax.envs import CartPoleEnv
from dynax.evaluation import (
    create_rollout_fn,
    create_true_rollout_fn,
    render_single_trajectory_video,
)
from dynax.mpc import NeuralTask
from dynax.utils import collect_and_prepare_data
from dynax.utils.data import extract_state_features
from hydrax.algs import PredictiveSampling
from hydrax.tasks.cart_pole import CartPole as HydraxCartPoleTask


class CartPoleEnvVaried(CartPoleEnv):
    """Cart pole environment with hardcoded varied mass and friction."""

    mass: float = 1.5
    friction: float = 0.1

    def _load_model(self, model_name: str, use_scene: bool) -> mujoco.MjModel:
        """Load and modify the model with varied parameters."""
        # Load the base model using parent's method
        model = super()._load_model(model_name, use_scene)

        # Modify mass of pole (usually body 2, after world and cart)
        # For cart pole: body 0 = world, body 1 = cart, body 2 = pole
        if model.nbody > 2:
            model.body_mass[2] = model.body_mass[2] * self.mass
            # Also scale inertia proportionally
            model.body_inertia[2, :] = model.body_inertia[2, :] * self.mass

        # Modify friction
        # Set DOF friction loss (stored per DOF, not per joint)
        # For cart pole, there are typically 2 DOFs
        # (cart translation, pole rotation)
        if model.nv > 0:
            model.dof_frictionloss[:] = self.friction

        return model


def train_model(model_path: str = "models/cart_pole_residual.pkl"):
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
    varied_env = CartPoleEnvVaried()

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
        dataset_path="data/cart_pole_varied_dataset.pkl",
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
    base_env = CartPoleEnv()
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
            log_dir="logs/cart_pole_residual",
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

    # Plot 3: Sample trajectory (cart position)
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
    axes[1, 0].set_ylabel("Cart Position (q)")
    axes[1, 0].set_title(
        f"Sample Trajectory: Cart Position (Rollout {traj_idx+1})"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Sample trajectory (pole angle)
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
    axes[1, 1].set_ylabel("Pole Angle (q)")
    axes[1, 1].set_title(
        f"Sample Trajectory: Pole Angle (Rollout {traj_idx+1})"
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "logs/cart_pole_residual/comparison_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    print("=" * 60)


def eval_model(model_path: str = "models/cart_pole_residual.pkl"):
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
    varied_env = CartPoleEnvVaried()
    base_env = CartPoleEnv()

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
    base_task = HydraxCartPoleTask()
    neural_task = NeuralTask(
        base_task=base_task,
        dynamics_model=dynamics_model,
        model_params=model_params,
    )

    # Create MPC controllers (using parameters from hydrax examples)
    print("Creating MPC controllers...")
    base_ctrl = PredictiveSampling(
        base_task,
        num_samples=128,
        noise_level=0.3,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=4,
    )
    learned_ctrl = PredictiveSampling(
        neural_task,
        num_samples=128,
        noise_level=0.3,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=4,
    )

    # Evaluation parameters
    num_episodes = 20
    episode_length = 500  # steps
    rng = jax.random.PRNGKey(42)

    print(f"\nRunning {num_episodes} episodes in parallel...")
    print("(Using true varied dynamics for simulation)")

    # Create initial states for all episodes
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_episodes)

    def reset_episode_eval(rng_key):
        """Reset episode for evaluation with zero initial velocities."""
        (
            rng_key,
            cart_rng,
            angle_rng,
        ) = jax.random.split(rng_key, 3)

        # Cart position: random in range [-1.5, 1.5] (within rail limits)
        cart_pos = jax.random.uniform(
            cart_rng, (), minval=-0.5, maxval=0.5
        )

        # Pole angle: random in range [-pi, pi]
        angle = jax.random.uniform(
            angle_rng, (), minval=-jnp.pi, maxval=jnp.pi
        )

        # Velocities: zero for evaluation
        cart_vel = 0.0
        angle_vel = 0.0

        data = mjx.make_data(varied_env.model)
        data = data.replace(
            qpos=jnp.array([cart_pos, angle]),
            qvel=jnp.array([cart_vel, angle_vel]),
        )
        data = mjx.forward(varied_env.model, data)
        return data

    initial_states = jax.vmap(reset_episode_eval)(reset_rngs)

    # JIT-compiled step function for true dynamics
    @jax.jit
    def true_step(data, action):
        """Step true (varied) dynamics."""
        data = data.replace(ctrl=action)
        return mjx.step(varied_env.model, data)

    # Functions to sync state from true model to controller-specific models
    @jax.jit
    def sync_to_base_model(true_state):
        """Sync state from varied model to base model."""
        ctrl_data = mjx.make_data(base_env.model)
        # Copy qpos, qvel, time, and ctrl from true state
        ctrl_data = ctrl_data.replace(
            qpos=true_state.qpos,
            qvel=true_state.qvel,
            time=true_state.time,
            ctrl=true_state.ctrl,
        )
        # Forward kinematics to update dependent quantities
        return mjx.forward(base_env.model, ctrl_data)

    @jax.jit
    def sync_to_neural_model(true_state):
        """Sync state from varied model to neural task model."""
        # Neural task uses the same model as base task
        ctrl_data = mjx.make_data(neural_task.model)
        ctrl_data = ctrl_data.replace(
            qpos=true_state.qpos,
            qvel=true_state.qvel,
            time=true_state.time,
            ctrl=true_state.ctrl,
        )
        return mjx.forward(neural_task.model, ctrl_data)

    # JIT-compiled cost function (uses base task, works with any model)
    @jax.jit
    def compute_cost(state, action):
        """Compute running cost."""
        return base_task.running_cost(state, action)

    @jax.jit
    def compute_terminal_cost(state):
        """Compute terminal cost."""
        return base_task.terminal_cost(state)

    # Create JIT-compiled optimize functions for each controller
    base_jit_optimize = jax.jit(base_ctrl.optimize)
    learned_jit_optimize = jax.jit(learned_ctrl.optimize)

    # Create JIT-compiled interpolation functions
    base_jit_interp = jax.jit(base_ctrl.interp_func)
    learned_jit_interp = jax.jit(learned_ctrl.interp_func)

    # Run single episode with a controller
    def run_episode_base(initial_state, episode_rng, return_trajectory=False):
        """Run a single MPC episode with base controller."""
        return _run_episode(
            initial_state,
            base_ctrl,
            base_jit_optimize,
            base_jit_interp,
            sync_to_base_model,
            return_trajectory=return_trajectory,
        )

    def run_episode_learned(
        initial_state, episode_rng, return_trajectory=False
    ):
        """Run a single MPC episode with learned controller."""
        return _run_episode(
            initial_state,
            learned_ctrl,
            learned_jit_optimize,
            learned_jit_interp,
            sync_to_neural_model,
            return_trajectory=return_trajectory,
        )

    def _run_episode(
        initial_state,
        ctrl,
        jit_optimize,
        jit_interp,
        sync_fn,
        return_trajectory=False,
    ):
        """Run a single MPC episode."""
        true_state = initial_state
        total_cost = 0.0

        # Initialize controller with synced state
        ctrl_state = sync_fn(true_state)
        ctrl_params = ctrl.init_params(seed=0)

        # Replanning frequency (replan every step for simplicity)
        replan_period = 1.0 / 10.0  # 10 Hz
        sim_steps_per_replan = max(1, int(replan_period / varied_env.dt))

        def step_fn(carry, step_idx):
            """Single MPC step."""
            true_state, ctrl_state, ctrl_params, total_cost = carry

            # Sync controller state from true state
            ctrl_state = sync_fn(true_state)

            # Replan if needed
            should_replan = step_idx % sim_steps_per_replan == 0

            def replan(s, p):
                """Replan."""
                new_params, _ = jit_optimize(s, p)
                return new_params

            def no_replan(s, p):
                """Don't replan."""
                return p

            # Conditionally replan using synced controller state
            ctrl_params = jax.lax.cond(
                should_replan,
                replan,
                no_replan,
                ctrl_state,
                ctrl_params,
            )

            # Get action from current policy
            t_curr = ctrl_state.time
            tq = jnp.array([t_curr])
            tk = ctrl_params.tk
            knots = ctrl_params.mean[None, ...]  # (1, num_knots, nu)
            actions = jit_interp(tq, tk, knots)
            action = actions[0, 0]  # (nu,)

            # Clip action to limits
            action = jnp.clip(action, ctrl.task.u_min, ctrl.task.u_max)

            # Compute cost using synced state (for consistency)
            cost = compute_cost(ctrl_state, action)
            total_cost = total_cost + cost

            # Step true dynamics (this is what actually happens)
            next_true_state = true_step(true_state, action)

            # Extract state features for trajectory tracking (after step)
            state_features = extract_state_features(next_true_state)

            return (
                (next_true_state, ctrl_state, ctrl_params, total_cost),
                state_features,
            )

        # Run episode
        (final_true_state, _, _, total_cost), trajectory = jax.lax.scan(
            step_fn,
            (true_state, ctrl_state, ctrl_params, total_cost),
            jnp.arange(episode_length),
        )

        # Add initial state to trajectory
        initial_features = extract_state_features(initial_state)
        trajectory = jnp.concatenate(
            [initial_features[None, :], trajectory], axis=0
        )

        # Add terminal cost using synced final state
        final_ctrl_state = sync_fn(final_true_state)
        terminal_cost = compute_terminal_cost(final_ctrl_state)
        total_cost = total_cost + terminal_cost

        if return_trajectory:
            return total_cost, trajectory
        return total_cost

    # Run episodes in parallel for both controllers
    rng, base_rng = jax.random.split(rng)
    rng, learned_rng = jax.random.split(rng)
    base_rngs = jax.random.split(base_rng, num_episodes)
    learned_rngs = jax.random.split(learned_rng, num_episodes)

    print("Running episodes with base physics MPC...")
    base_costs = jax.vmap(run_episode_base)(initial_states, base_rngs)

    print("Running episodes with learned neural MPC...")
    learned_costs = jax.vmap(run_episode_learned)(
        initial_states, learned_rngs
    )

    # Run a single episode with trajectory tracking for visualization
    print("\nRendering videos for visualization...")
    num_videos = 1  # Start with 1 video

    for vid_idx in range(num_videos):
        # Run base controller episode with trajectory
        base_cost, base_traj = run_episode_base(
            initial_states[vid_idx],
            base_rngs[vid_idx],
            return_trajectory=True,
        )

        # Run learned controller episode with trajectory
        learned_cost, learned_traj = run_episode_learned(
            initial_states[vid_idx],
            learned_rngs[vid_idx],
            return_trajectory=True,
        )

        # Render base controller video
        base_video_path = (
            f"logs/cart_pole_residual/base_mpc_episode_{vid_idx}.mp4"
        )
        print(f"  Rendering base MPC video: {base_video_path}")
        render_single_trajectory_video(
            env=varied_env,
            states=base_traj,
            fps=30,
            save_path=base_video_path,
            max_frames=500,
        )

        # Render learned controller video
        learned_video_path = (
            f"logs/cart_pole_residual/learned_mpc_episode_{vid_idx}.mp4"
        )
        print(f"  Rendering learned MPC video: {learned_video_path}")
        render_single_trajectory_video(
            env=varied_env,
            states=learned_traj,
            fps=30,
            save_path=learned_video_path,
            max_frames=500,
        )

        print(
            f"  Base MPC cost: {base_cost:.2f}, "
            f"Learned MPC cost: {learned_cost:.2f}"
        )

    # Compute statistics
    base_costs_np = np.array(base_costs)
    learned_costs_np = np.array(learned_costs)

    base_mean = np.mean(base_costs_np)
    base_std = np.std(base_costs_np)
    learned_mean = np.mean(learned_costs_np)
    learned_std = np.std(learned_costs_np)

    improvement = ((base_mean - learned_mean) / base_mean) * 100

    print("\n" + "=" * 80)
    print("MPC Performance Comparison")
    print("=" * 80)
    print(f"\nEpisodes: {num_episodes}")
    print(f"Episode length: {episode_length} steps")
    print("\nBase Physics MPC (wrong model):")
    print(f"  Mean cost: {base_mean:.4f} ± {base_std:.4f}")
    print(f"  Min cost:  {np.min(base_costs_np):.4f}")
    print(f"  Max cost:  {np.max(base_costs_np):.4f}")

    print("\nLearned Neural MPC:")
    print(f"  Mean cost: {learned_mean:.4f} ± {learned_std:.4f}")
    print(f"  Min cost:  {np.min(learned_costs_np):.4f}")
    print(f"  Max cost:  {np.max(learned_costs_np):.4f}")

    # Identify worst-performing learned neural MPC episodes
    num_worst_episodes = min(3, num_episodes)  # Render top 3 worst
    worst_indices = np.argsort(learned_costs_np)[-num_worst_episodes:][::-1]
    worst_costs = learned_costs_np[worst_indices]

    print(f"\nWorst {num_worst_episodes} learned neural MPC episodes:")
    for idx, cost in zip(worst_indices, worst_costs):
        print(f"  Episode {idx}: cost = {cost:.2f}")

    # Render videos for worst-performing episodes
    if num_worst_episodes > 0:
        print("\nRendering videos for worst-performing learned MPC episodes...")
        for rank, (ep_idx, cost) in enumerate(zip(worst_indices, worst_costs)):
            print(
                f"\n  Episode {ep_idx} (rank {rank+1} worst, cost={cost:.2f}):"
            )

            # Re-run learned controller episode with trajectory tracking
            learned_cost, learned_traj = run_episode_learned(
                initial_states[ep_idx],
                learned_rngs[ep_idx],
                return_trajectory=True,
            )

            # Also run base controller for comparison
            base_cost, base_traj = run_episode_base(
                initial_states[ep_idx],
                base_rngs[ep_idx],
                return_trajectory=True,
            )

            # Render learned controller video
            learned_video_path = (
                f"logs/cart_pole_residual/"
                f"learned_mpc_worst_{rank+1}_episode_{ep_idx}.mp4"
            )
            print(f"    Rendering learned MPC video: {learned_video_path}")
            render_single_trajectory_video(
                env=varied_env,
                states=learned_traj,
                fps=30,
                save_path=learned_video_path,
                max_frames=500,
            )

            # Render base controller video for comparison
            base_video_path = (
                f"logs/cart_pole_residual/"
                f"base_mpc_worst_{rank+1}_episode_{ep_idx}.mp4"
            )
            print(f"    Rendering base MPC video: {base_video_path}")
            render_single_trajectory_video(
                env=varied_env,
                states=base_traj,
                fps=30,
                save_path=base_video_path,
                max_frames=500,
            )

            print(
                f"    Base MPC cost: {base_cost:.2f}, "
                f"Learned MPC cost: {learned_cost:.2f}"
            )

    print(f"\nImprovement: {improvement:.2f}% cost reduction")
    print("=" * 80)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cost distribution
    axes[0].hist(
        base_costs_np,
        bins=15,
        alpha=0.7,
        label="Base Physics MPC",
        color="red",
    )
    axes[0].hist(
        learned_costs_np,
        bins=15,
        alpha=0.7,
        label="Learned Neural MPC",
        color="blue",
    )
    axes[0].axvline(base_mean, color="red", linestyle="--", linewidth=2)
    axes[0].axvline(learned_mean, color="blue", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Total Episode Cost")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Cost Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Comparison bar plot
    categories = ["Base Physics\nMPC", "Learned Neural\nMPC"]
    means = [base_mean, learned_mean]
    stds = [base_std, learned_std]
    colors = ["red", "blue"]

    bars = axes[1].bar(categories, means, yerr=stds, capsize=10, color=colors)
    axes[1].set_ylabel("Mean Episode Cost")
    axes[1].set_title("MPC Performance Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot_path = "logs/cart_pole_residual/mpc_comparison.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate a residual dynamics model for cart pole."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train mode
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="models/cart_pole_residual.pkl",
        help="Path to save the trained model",
    )

    # Eval mode
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="models/cart_pole_residual.pkl",
        help="Path to the saved model",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(model_path=args.model_path)
    elif args.mode == "eval":
        eval_model(model_path=args.model_path)


if __name__ == "__main__":
    main()

