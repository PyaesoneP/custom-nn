"""Minimal inference functions for the trained neural network."""

import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z


def forward_propagation(X, parameters):
    """
    Forward propagation: [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID
    
    Arguments:
        X: input data of shape (n_x, 1)
        parameters: dict containing W1, b1, W2, b2, ...
    
    Returns:
        AL: probability (output of sigmoid)
    """
    A = X
    L = len(parameters) // 2
    
    # Hidden layers with ReLU
    for l in range(1, L):
        Z = linear_forward(A, parameters[f'W{l}'], parameters[f'b{l}'])
        A = relu(Z)
    
    # Output layer with Sigmoid
    Z = linear_forward(A, parameters[f'W{L}'], parameters[f'b{L}'])
    AL = sigmoid(Z)
    
    return AL


def predict(X, parameters):
    """
    Predict binary class for input X.
    
    Arguments:
        X: input image flattened to shape (12288, 1)
        parameters: trained model parameters
    
    Returns:
        prediction: 0 or 1
        confidence: probability score
    """
    AL = forward_propagation(X, parameters)
    prediction = int(AL[0, 0] > 0.5)
    confidence = float(AL[0, 0]) if prediction == 1 else float(1 - AL[0, 0])
    
    return prediction, confidence
