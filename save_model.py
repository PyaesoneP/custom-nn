"""
Run this script after training to save the model parameters.

Usage:
    1. Train your model in the notebook
    2. Add this cell at the end:
    
        import pickle
        with open("model/parameters.pkl", "wb") as f:
            pickle.dump(parameters, f)
        print("Model saved!")

Or run this script if parameters are in memory.
"""

import pickle
from pathlib import Path

def save_parameters(parameters, path="model/parameters.pkl"):
    """Save trained parameters to disk."""
    Path(path).parent.mkdir(exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(parameters, f)
    
    print(f"Parameters saved to {path}")
    print(f"Layers: {len(parameters) // 2}")
    for key in sorted(parameters.keys()):
        print(f"  {key}: {parameters[key].shape}")


def load_parameters(path="model/parameters.pkl"):
    """Load parameters from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Example: verify saved model
    try:
        params = load_parameters()
        print("Model loaded successfully!")
        print(f"Layers: {len(params) // 2}")
    except FileNotFoundError:
        print("No saved model found. Train the model first and save parameters.")
