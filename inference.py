"""Minimal inference functions for the trained CNN."""

import numpy as np
from utils import cnn_forward


# Default architecture — must match the one used during training.
# This is a fallback; the saved model file also stores the architecture
# and that version always takes precedence.
DEFAULT_ARCHITECTURE = [
    # Block 1: 64×64×3 → 64×64×16 → 32×32×16
    {"type": "conv",   "name": "conv1",  "filters": 16, "kernel": 3, "stride": 1, "padding": "same"},
    {"type": "relu",   "name": "relu1"},
    {"type": "conv",   "name": "conv2",  "filters": 16, "kernel": 3, "stride": 1, "padding": "same"},
    {"type": "relu",   "name": "relu2"},
    {"type": "pool",   "name": "pool1",  "pool_size": 2, "stride": 2},
    # Block 2: 32×32×16 → 32×32×32 → 16×16×32
    {"type": "conv",   "name": "conv3",  "filters": 32, "kernel": 3, "stride": 1, "padding": "same"},
    {"type": "relu",   "name": "relu3"},
    {"type": "conv",   "name": "conv4",  "filters": 32, "kernel": 3, "stride": 1, "padding": "same"},
    {"type": "relu",   "name": "relu4"},
    {"type": "pool",   "name": "pool2",  "pool_size": 2, "stride": 2},
    # Flatten + Dense (regularised via augmentation + light L2 during training)
    {"type": "flatten", "name": "flat"},
    {"type": "dense",  "name": "fc1",    "units": 128, "activation": "relu"},
    {"type": "dense",  "name": "fc2",    "units": 64,  "activation": "relu"},
    {"type": "dense",  "name": "fc3",    "units": 1,   "activation": "sigmoid"},
]


def forward_propagation(X, parameters, architecture=None):
    """
    Forward pass through the CNN.

    Parameters
    ----------
    X : ndarray  shape (1, 64, 64, 3) for a single image
    parameters : dict  model weights & biases
    architecture : list | None  layer specs (uses DEFAULT_ARCHITECTURE if None)

    Returns
    -------
    AL : ndarray  probability (1, 1)
    """
    if architecture is None:
        architecture = DEFAULT_ARCHITECTURE
    AL, _ = cnn_forward(X, parameters, architecture, training=False)
    return AL


def predict(X, parameters, architecture=None):
    """
    Predict binary class for a single preprocessed image.

    Parameters
    ----------
    X : ndarray  shape (1, 64, 64, 3) — preprocessed image
    parameters : dict
    architecture : list | None

    Returns
    -------
    prediction : int     0 or 1
    confidence : float   probability of the predicted class
    """
    AL = forward_propagation(X, parameters, architecture)
    pred = int(AL[0, 0] > 0.5)
    conf = float(AL[0, 0]) if pred == 1 else float(1 - AL[0, 0])
    return pred, conf
