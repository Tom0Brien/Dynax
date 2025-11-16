"""Helper functions for loading MuJoCo models."""

import os
from pathlib import Path

import mujoco
from mujoco import mjx

# Path to the models directory
MODELS_DIR = Path(__file__).parent / "models"


def load_model(model_name: str, use_scene: bool = False) -> mujoco.MjModel:
    """Load a MuJoCo model from the models directory.

    Args:
        model_name: Name of the model (e.g., "pendulum", "cart_pole").
        use_scene: Whether to load the scene.xml instead of the model.xml.

    Returns:
        MuJoCo model.
    """
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        raise ValueError(
            f"Model '{model_name}' not found in {MODELS_DIR}. "
            f"Available models: {list_available_models()}"
        )

    if use_scene:
        xml_path = model_dir / "scene.xml"
    else:
        xml_path = model_dir / f"{model_name}.xml"

    if not xml_path.exists():
        raise ValueError(f"XML file not found: {xml_path}")

    return mujoco.MjModel.from_xml_path(str(xml_path))


def load_mjx_model(model_name: str, use_scene: bool = False) -> mjx.Model:
    """Load a MuJoCo MJX model from the models directory.

    Args:
        model_name: Name of the model (e.g., "pendulum", "cart_pole").
        use_scene: Whether to load the scene.xml instead of the model.xml.

    Returns:
        MJX model.
    """
    mj_model = load_model(model_name, use_scene)
    return mjx.put_model(mj_model)


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

