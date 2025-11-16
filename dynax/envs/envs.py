"""Environment classes for managing MuJoCo models and data collection."""

from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# Path to the models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


class Env:
    """Base environment class for managing MuJoCo models and data collection.

    This class provides:
    - Model loading from XML files
    - Reset functionality for initializing states
    - Action bounds from model actuators
    - Timestep information
    """

    def __init__(
        self,
        model_name: str,
        use_scene: bool = False,
    ):
        """Initialize the environment.

        Args:
            model_name: Name of the model (e.g., "pendulum", "cart_pole").
            use_scene: Whether to load the scene.xml instead of the model.xml.
        """
        self.model_name = model_name
        self.mj_model = self._load_model(model_name, use_scene)
        self.model = mjx.put_model(self.mj_model)

        # Extract action bounds from actuators
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

    def _load_model(self, model_name: str, use_scene: bool) -> mujoco.MjModel:
        """Load a MuJoCo model from the models directory."""
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            available = list_available_models()
            raise ValueError(
                f"Model '{model_name}' not found in {MODELS_DIR}. "
                f"Available models: {available}"
            )

        if use_scene:
            xml_path = model_dir / "scene.xml"
        else:
            xml_path = model_dir / f"{model_name}.xml"

        if not xml_path.exists():
            raise ValueError(f"XML file not found: {xml_path}")

        return mujoco.MjModel.from_xml_path(str(xml_path))

    def reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Reset the environment to an initial state.

        Args:
            data: MJX data object.
            rng: Random number generator key.

        Returns:
            Reset MJX data object.
        """
        return self._reset(data, rng)

    @abstractmethod
    def _reset(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        """Subclass-specific reset logic."""
        pass


def list_available_models() -> list[str]:
    """List all available models in the models directory.

    Returns:
        List of model names.
    """
    if not MODELS_DIR.exists():
        return []

    return [
        d.name
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

