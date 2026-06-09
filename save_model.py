"""
Save and load trained CNN model (parameters + architecture).

Usage:
    1. Train your model in the notebook.
    2. Add this cell at the end:

        from save_model import save_model
        save_model(parameters, architecture)
"""

import pickle
from pathlib import Path


def save_model(parameters, architecture, path="model/parameters.pkl"):
    """Save trained parameters and architecture to disk."""
    Path(path).parent.mkdir(exist_ok=True)

    bundle = (parameters, architecture)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model saved to {path}")
    _print_params(parameters)


def load_model(path="model/parameters.pkl"):
    """
    Load model from disk.

    Returns
    -------
    parameters : dict
    architecture : list of layer-spec dicts
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        parameters, architecture = data
    else:
        raise RuntimeError(
            "Old model format detected (DNN parameters only). "
            "Please retrain the CNN model."
        )

    return parameters, architecture


def _print_params(parameters):
    """Print model parameter summary."""
    trainable = {k: v for k, v in sorted(parameters.items())}
    total = sum(v.size for v in trainable.values())
    print(f"Trainable parameters: {total:,}")
    for key in sorted(parameters.keys()):
        print(f"  {key:12s}  {str(parameters[key].shape):20s}")


if __name__ == "__main__":
    try:
        params, arch = load_model()
        print("Model loaded successfully!")
        _print_params(params)
    except FileNotFoundError:
        print("No saved model found. Train the model first.")
