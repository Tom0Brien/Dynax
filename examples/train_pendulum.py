"""Train a neural dynamics model for a simple pendulum.

This example demonstrates the basic workflow:
1. Create a MuJoCo model
2. Collect training data
3. Train a neural dynamics model
4. Evaluate the model

Usage:
    python examples/train_pendulum.py
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from dynax.architectures import MLPDynamicsModel
from dynax.models import TrainingConfig, train_dynamics_model
from dynax.utils import (
    collect_random_rollouts,
    create_dataset,
    split_dataset,
)


def create_pendulum_model():
    """Create a simple pendulum MuJoCo model."""
    xml = """
    <mujoco model="pendulum">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.02" integrator="implicitfast"/>
      
      <default>
        <geom type="cylinder" size="0.01"/>
        <joint type="hinge" damping="0.1" limited="false"/>
      </default>
      
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        
        <body name="pole" pos="0 0 0.5">
          <joint name="hinge" axis="0 1 0"/>
          <geom name="pole" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0.7 0.3 0.3 1"/>
          <geom name="mass" pos="0 0 -0.5" type="sphere" size="0.05" rgba="0.3 0.3 0.7 1"/>
        </body>
      </worldbody>
      
      <actuator>
        <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-3 3"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def main():
    print("=" * 60)
    print("TRAINING PENDULUM DYNAMICS MODEL")
    print("=" * 60)

    # Create MuJoCo model
    print("\nCreating pendulum model...")
    mj_model = create_pendulum_model()
    model = mjx.put_model(mj_model)
    print(f"  nq: {model.nq}, nv: {model.nv}, nu: {model.nu}")

    # Collect training data
    print("\nCollecting training data...")
    rng = jax.random.PRNGKey(0)

    num_rollouts = 50
    rollout_length = 100
    action_min = jnp.array([-3.0])
    action_max = jnp.array([3.0])

    states, actions, next_states, accelerations = collect_random_rollouts(
        model=model,
        num_rollouts=num_rollouts,
        rollout_length=rollout_length,
        action_min=action_min,
        action_max=action_max,
        rng=rng,
    )

    print(f"  Collected {len(states)} transitions")

    # Create dataset
    dt = mj_model.opt.timestep
    dataset = create_dataset(
        model=model,
        states=states,
        actions=actions,
        next_states=next_states,
        accelerations=accelerations,
        dt=dt,
    )

    # Split into train and validation
    rng, split_rng = jax.random.split(rng)
    train_dataset, val_dataset = split_dataset(
        dataset, train_ratio=0.8, rng=split_rng
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print("\nCreating MLP model...")
    model_arch = MLPDynamicsModel(
        state_dim=dataset.state_dim,
        nq=dataset.nq,
        action_dim=dataset.action_dim,
        hidden_dims=(128, 128),
        activation="swish",
    )
    print(f"  Hidden dims: (128, 128)")
    print(f"  Activation: swish")

    # Train model
    print("\nTraining model...")
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=100,
        weight_decay=1e-4,
    )

    rng, train_rng = jax.random.split(rng)
    trained_params = train_dynamics_model(
        model=model_arch,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        rng=train_rng,
        verbose=True,
    )

    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    # Single-step prediction accuracy
    num_test = min(100, len(val_dataset))
    test_states = val_dataset.states[:num_test]
    test_actions = val_dataset.actions[:num_test]
    test_next_states = val_dataset.next_states[:num_test]

    # Predict next states
    pred_next_states = jax.vmap(
        lambda s, a: model_arch.step(trained_params, s, a)
    )(test_states, test_actions)

    # Compute errors
    errors = jnp.abs(pred_next_states - test_next_states)
    mae = jnp.mean(errors, axis=0)

    print(f"\nSingle-step prediction MAE:")
    print(f"  Position: {mae[0]:.6f} rad")
    print(f"  Velocity: {mae[1]:.6f} rad/s")
    print(f"  Overall: {jnp.mean(mae):.6f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(
        "\nThe model has been trained successfully! "
        "You can now use it for predictions or control."
    )


if __name__ == "__main__":
    main()

