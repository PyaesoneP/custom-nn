# Deep Neural Network from Scratch

A 4-layer deep neural network built entirely from scratch using NumPy for binary image classification. This project demonstrates the fundamental building blocks of neural networks without relying on deep learning frameworks.

## Overview

This implementation classifies images as either "cat" or "non-cat" using a deep neural network with the following architecture:

```
Input (12288) → Dense (20, ReLU) → Dense (7, ReLU) → Dense (5, ReLU) → Output (1, Sigmoid)
```

The input layer takes flattened 64×64×3 RGB images (12,288 features), passes them through three hidden layers with ReLU activation, and outputs a probability through a sigmoid activation.

## Features

**Implemented from scratch:**
- Forward propagation with vectorized operations
- Backward propagation with gradient computation
- Parameter initialization (He initialization for ReLU layers)
- Gradient descent optimization
- Cost function (binary cross-entropy)

**Activation functions:**
- ReLU for hidden layers
- Sigmoid for output layer

## Project Structure

```
├── custom-nn.ipynb    # Main notebook with training pipeline
├── utils.py           # Core neural network functions
├── datasets/
│   ├── train_catvnoncat.h5
│   └── test_catvnoncat.h5
└── README.md
```

## Requirements

```
numpy
matplotlib
h5py
scipy
pillow
```

Install dependencies:
```bash
pip install numpy matplotlib h5py scipy pillow
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/pyaesonep/custom-nn.git
cd custom-nn
```

2. Run the Jupyter notebook:
```bash
jupyter notebook custom-nn.ipynb
```

3. Execute cells sequentially to:
   - Load and preprocess the dataset
   - Train the neural network
   - Evaluate on test data

## How It Works

### Forward Propagation

For each layer $l$:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Where $g$ is ReLU for hidden layers and sigmoid for the output layer.

### Backward Propagation

Gradients are computed using the chain rule, propagating from the output layer back to the input:

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T}$$

$$db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$$

$$dA^{[l-1]} = W^{[l]T} dZ^{[l]}$$

### Cost Function

Binary cross-entropy loss:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{[L](i)}) + (1 - y^{(i)}) \log(1 - a^{[L](i)}) \right]$$

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Test Accuracy | ~80% |
| Iterations | 2500 |
| Learning Rate | 0.0075 |

The gap between training and test accuracy indicates some overfitting, which is expected given the small dataset size and network capacity.

## Key Learnings

This project reinforced understanding of:
- Vectorized implementation of neural network operations
- The mechanics of backpropagation and gradient flow
- Impact of network depth on learning capacity
- Importance of proper weight initialization
- Trade-offs between model complexity and generalization

## Acknowledgments

This project was completed as part of the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI on Coursera.

## License

MIT License
