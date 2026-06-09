"""
Utility functions for a custom CNN built from scratch with NumPy.

Provides:
  - Activation functions (sigmoid, relu) and their gradients
  - Data loading from HDF5
  - im2col / col2im for efficient Conv2D
  - Conv2D, MaxPool, Flatten, Dense layers (forward + backward)
  - Parameter initialisation (He)
  - Forward / backward orchestration via an architecture-spec list
  - Adam and vanilla GD optimisers
  - Training loop and prediction helpers
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image, ImageEnhance


# =============================================================================
# 1. Activation functions
# =============================================================================

def sigmoid(Z):
    """Sigmoid activation."""
    A = 1 / (1 + np.exp(-Z))
    return A, Z


def relu(Z):
    """ReLU activation."""
    A = np.maximum(0, Z)
    return A, Z


def sigmoid_backward(dA, cache):
    """Backward pass for sigmoid.  cache = Z."""
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """Backward pass for ReLU.  cache = Z."""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# =============================================================================
# 2. Data loading
# =============================================================================

def load_data():
    """Load cat / non-cat dataset from HDF5 files."""
    train_ds = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_ds["train_set_x"][:])
    train_y = np.array(train_ds["train_set_y"][:])

    test_ds = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_ds["test_set_x"][:])
    test_y = np.array(test_ds["test_set_y"][:])
    classes = np.array(test_ds["list_classes"][:])

    train_y = train_y.reshape((1, -1))
    test_y = test_y.reshape((1, -1))

    return train_x, train_y, test_x, test_y, classes


# =============================================================================
# 3. Data augmentation  (pre-generate expanded training set)
# =============================================================================

def augment_dataset(X, Y, factor=4):
    """
    Expand a small image dataset via randomised augmentations.

    Each original image produces *factor* variants (including the original).
    Augmentations are safe for cat/non-cat classification:
    horizontal flip, slight rotation, brightness jitter, zoom crop.

    Parameters
    ----------
    X : ndarray  shape (m, 64, 64, 3)  dtype uint8  (pre-normalisation)
    Y : ndarray  shape (1, m)
    factor : int  how many variants per image (default 4)

    Returns
    -------
    X_aug : ndarray  shape (m * factor, 64, 64, 3)  uint8
    Y_aug : ndarray  shape (1, m * factor)
    """
    m = X.shape[0]
    X_aug = np.empty((m * factor, 64, 64, 3), dtype=X.dtype)
    Y_aug = np.empty((1, m * factor), dtype=Y.dtype)

    for i in range(m):
        img = Image.fromarray(X[i])                 # PIL image (64x64 RGB)
        label = Y[0, i]
        base_idx = i * factor

        # Variant 0: original
        X_aug[base_idx] = X[i]
        Y_aug[0, base_idx] = label

        # Variant 1: horizontal flip
        if factor >= 2:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            X_aug[base_idx + 1] = np.array(flipped)
            Y_aug[0, base_idx + 1] = label

        # Variant 2: slight rotation + brightness jitter
        if factor >= 3:
            angle = np.random.uniform(-12, 12)
            rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            brightness = ImageEnhance.Brightness(rotated)
            rotated = brightness.enhance(np.random.uniform(0.8, 1.2))
            X_aug[base_idx + 2] = np.array(rotated)
            Y_aug[0, base_idx + 2] = label

        # Variant 3: zoom-in crop + optional horizontal flip
        if factor >= 4:
            crop_margin = np.random.randint(2, 7)   # crop 2-6 px from each side
            cropped = img.crop((crop_margin, crop_margin,
                                64 - crop_margin, 64 - crop_margin))
            cropped = cropped.resize((64, 64), Image.BILINEAR)
            if np.random.rand() < 0.5:
                cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
            X_aug[base_idx + 3] = np.array(cropped)
            Y_aug[0, base_idx + 3] = label

    # Shuffle the augmented dataset
    perm = np.random.permutation(m * factor)
    return X_aug[perm], Y_aug[:, perm]


# =============================================================================
# 4. im2col / col2im  (efficient Conv2D via matrix multiply)
# =============================================================================

def im2col(X, filter_h, filter_w, stride, pad):
    """
    Rearrange 4-D image tensor into columns for GEMM-based convolution.

    Parameters
    ----------
    X : ndarray  shape (m, H, W, C)
    filter_h, filter_w : int   kernel spatial dimensions
    stride : int
    pad : int   amount of zero-padding on each side

    Returns
    -------
    cols : ndarray  shape (m * out_h * out_w, filter_h * filter_w * C)
    """
    m, H, W, C = X.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')

    cols = np.empty((m, out_h, out_w, filter_h, filter_w, C), dtype=X.dtype)

    for i in range(filter_h):
        i_end = i + stride * out_h
        for j in range(filter_w):
            j_end = j + stride * out_w
            cols[:, :, :, i, j, :] = X_pad[:, i:i_end:stride, j:j_end:stride, :]

    cols = cols.transpose(0, 1, 2, 5, 3, 4)
    cols = cols.reshape(m * out_h * out_w, -1)
    return cols


def col2im(dcol, X_shape, filter_h, filter_w, stride, pad):
    """
    Inverse of im2col — map column gradients back to 4-D image gradient.

    Parameters
    ----------
    dcol : ndarray  shape (m * out_h * out_w, filter_h * filter_w * C)
    X_shape : tuple  (m, H, W, C) of the *original* (pre-pad) input
    filter_h, filter_w, stride, pad : int   same as used in forward im2col

    Returns
    -------
    dX : ndarray  shape (m, H, W, C)
    """
    m, H, W, C = X_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    cols = dcol.reshape(m, out_h, out_w, C, filter_h, filter_w)

    dX_pad = np.zeros((m, H + 2 * pad, W + 2 * pad, C), dtype=dcol.dtype)

    for i in range(filter_h):
        i_end = i + stride * out_h
        for j in range(filter_w):
            j_end = j + stride * out_w
            dX_pad[:, i:i_end:stride, j:j_end:stride, :] += (
                cols[:, :, :, :, i, j]
            )

    if pad == 0:
        return dX_pad
    return dX_pad[:, pad:-pad, pad:-pad, :]


# =============================================================================
# 4. Conv2D layer
# =============================================================================

def conv2d_forward(A_prev, W, b, stride, padding):
    """
    Conv2D forward pass (im2col + matmul).

    Parameters
    ----------
    A_prev : ndarray  shape (m, H, W, C)
    W : ndarray  shape (filters, f_h, f_w, C)
    b : ndarray  shape (filters,)
    stride : int
    padding : int | str  'same' or integer pad amount

    Returns
    -------
    Z : ndarray  shape (m, out_h, out_w, filters)
    cache : tuple  for backward pass
    """
    m, H_in, W_in, C = A_prev.shape
    filters, f_h, f_w, _ = W.shape

    if padding == 'same':
        pad_h = (f_h - 1) // 2
        pad_w = (f_w - 1) // 2
    else:
        pad_h = pad_w = int(padding)

    out_h = (H_in + 2 * pad_h - f_h) // stride + 1
    out_w = (W_in + 2 * pad_w - f_w) // stride + 1

    cols = im2col(A_prev, f_h, f_w, stride, pad_h)       # (m*oh*ow, f_h*f_w*C)
    W_col = W.reshape(filters, -1).T                       # (f_h*f_w*C, filters)
    Z = cols @ W_col + b                                   # (m*oh*ow, filters)
    Z = Z.reshape(m, out_h, out_w, filters)

    cache = (A_prev, W, b, cols, stride, pad_h, pad_w)
    return Z, cache


def conv2d_backward(dZ, cache):
    """
    Conv2D backward pass.

    Parameters
    ----------
    dZ : ndarray  shape (m, out_h, out_w, filters)
    cache : tuple  from conv2d_forward

    Returns
    -------
    dA_prev : ndarray  shape (m, H, W, C)
    dW : ndarray  shape (filters, f_h, f_w, C)
    db : ndarray  shape (filters,)
    """
    A_prev, W, b, cols, stride, pad_h, pad_w = cache
    m, H_in, W_in, C = A_prev.shape
    filters, f_h, f_w, _ = W.shape
    _, out_h, out_w, _ = dZ.shape

    dZ_col = dZ.reshape(m * out_h * out_w, filters)        # (m*oh*ow, filters)

    # dW
    dW_col = cols.T @ dZ_col                                # (f_h*f_w*C, filters)
    dW = dW_col.T.reshape(filters, f_h, f_w, C)

    # db
    db = np.sum(dZ, axis=(0, 1, 2)).reshape(filters)

    # dA_prev
    W_col = W.reshape(filters, -1)                          # (filters, f_h*f_w*C)
    dcols = dZ_col @ W_col                                  # (m*oh*ow, f_h*f_w*C)
    dA_prev = col2im(dcols, (m, H_in, W_in, C), f_h, f_w, stride, pad_h)

    return dA_prev, dW, db


# =============================================================================
# 5. MaxPool layer
# =============================================================================

def maxpool_forward(A_prev, pool_size, stride):
    """
    Max-pooling forward pass.

    Parameters
    ----------
    A_prev : ndarray  shape (m, H, W, C)
    pool_size : int
    stride : int

    Returns
    -------
    out : ndarray  shape (m, out_h, out_w, C)
    cache : tuple  for backward pass (contains argmax mask)
    """
    m, H, W, C = A_prev.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    out = np.zeros((m, out_h, out_w, C))
    max_mask = np.zeros_like(A_prev)

    for i in range(out_h):
        for j in range(out_w):
            h_s = i * stride
            h_e = h_s + pool_size
            w_s = j * stride
            w_e = w_s + pool_size

            patch = A_prev[:, h_s:h_e, w_s:w_e, :]          # (m, ps, ps, C)
            max_vals = np.max(patch, axis=(1, 2))           # (m, C)
            out[:, i, j, :] = max_vals

            # Build mask: for each sample & channel, mark the argmax
            patch_flat = patch.reshape(m, pool_size * pool_size, C)
            argmax_flat = np.argmax(patch_flat, axis=1)     # (m, C)
            for b in range(m):
                for c in range(C):
                    am = argmax_flat[b, c]
                    hi = am // pool_size
                    wi = am % pool_size
                    max_mask[b, h_s + hi, w_s + wi, c] = 1

    cache = (max_mask, pool_size, stride)
    return out, cache


def maxpool_backward(dA, cache):
    """
    Max-pooling backward pass — routes gradient to the max positions.

    Parameters
    ----------
    dA : ndarray  shape (m, out_h, out_w, C)
    cache : tuple  (max_mask, pool_size, stride)

    Returns
    -------
    dA_prev : ndarray  shape (m, H_in, W_in, C)
    """
    max_mask, pool_size, stride = cache
    m, out_h, out_w, C = dA.shape
    H_in = (out_h - 1) * stride + pool_size
    W_in = (out_w - 1) * stride + pool_size
    dA_prev = np.zeros((m, H_in, W_in, C))

    for i in range(out_h):
        for j in range(out_w):
            h_s = i * stride
            h_e = h_s + pool_size
            w_s = j * stride
            w_e = w_s + pool_size
            dA_prev[:, h_s:h_e, w_s:w_e, :] += (
                dA[:, i:i+1, j:j+1, :] * max_mask[:, h_s:h_e, w_s:w_e, :]
            )

    return dA_prev


# =============================================================================
# 6. Dropout layer
# =============================================================================

def dropout_forward(A, keep_prob):
    """
    Inverted dropout — scales activations so no adjustment at test time.

    During training: randomly zeroes (1 - keep_prob) fraction of units
    and scales the rest by 1/keep_prob to keep expected sum unchanged.

    During inference (keep_prob = 1.0): identity pass-through.

    Parameters
    ----------
    A : ndarray  any shape
    keep_prob : float  probability of keeping a unit (1.0 = no dropout)

    Returns
    -------
    A_out : ndarray  same shape as A
    cache : tuple  (mask, keep_prob) for backward, or (None, 1.0) if no dropout
    """
    if keep_prob >= 1.0:
        return A, (None, 1.0)

    mask = (np.random.rand(*A.shape) < keep_prob).astype(A.dtype)
    A_out = (A * mask) / keep_prob
    return A_out, (mask, keep_prob)


def dropout_backward(dA, cache):
    """
    Backward pass for dropout — route gradient only through active units.
    """
    mask, keep_prob = cache
    if keep_prob >= 1.0:
        return dA
    return (dA * mask) / keep_prob


# =============================================================================
# 7. Flatten layer
# =============================================================================

def flatten_forward(A_prev):
    """
    Flatten 4-D tensor → 2-D matrix for dense layers.

    Input  : (m, H, W, C)
    Output : (H*W*C,  m)    (feature-first convention for dense layers)
    """
    input_shape = A_prev.shape                    # (m, H, W, C)
    m = input_shape[0]
    A = A_prev.reshape(m, -1).T                   # (H*W*C, m)
    return A, input_shape


def flatten_backward(dA, cache):
    """
    Inverse of flatten.

    Input  : (features, m)
    Output : (m, H, W, C)
    """
    input_shape = cache                           # (m, H, W, C)
    return dA.T.reshape(input_shape)


# =============================================================================
# 8. Dense (fully-connected) layer helpers
# =============================================================================

def linear_forward(A, W, b):
    """Linear forward: Z = W @ A + b   where A is (n_prev, m)."""
    Z = W @ A + b
    cache = (A, W, b)
    return Z, cache


def linear_backward(dZ, cache):
    """Linear backward."""
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1.0 / m) * (dZ @ A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def linear_activation_forward(A_prev, W, b, activation):
    """Dense layer forward: LINEAR → ACTIVATION."""
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "relu":
        A, act_cache = relu(Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    cache = (linear_cache, act_cache)
    return A, cache


def linear_activation_backward(dA, cache, activation):
    """Dense layer backward."""
    linear_cache, act_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, act_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, act_cache)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


# =============================================================================
# 9. Cost function (with optional L2 penalty)
# =============================================================================

def compute_cost(AL, Y):
    """
    Binary cross-entropy cost (no regularisation).

    AL : (1, m)  predictions
    Y  : (1, m)  ground-truth labels
    """
    m = Y.shape[1]
    cost = (1.0 / m) * (-Y @ np.log(AL).T - (1 - Y) @ np.log(1 - AL).T)
    return np.squeeze(cost)


def compute_cost_with_l2(AL, Y, parameters, architecture, weight_decay):
    """
    Binary cross-entropy + L2 weight decay penalty.

    $$ J = -1/m Σ [y log(ŷ) + (1-y)log(1-ŷ)] + λ/(2m) Σ ||W||² $$

    The sum runs over all trainable weight matrices (biases excluded).
    """
    m = Y.shape[1]
    cross_entropy = (1.0 / m) * (
        -Y @ np.log(AL + 1e-12).T - (1 - Y) @ np.log(1 - AL + 1e-12).T
    )
    cross_entropy = np.squeeze(cross_entropy)

    if weight_decay <= 0:
        return cross_entropy

    l2_penalty = 0.0
    for layer in architecture:
        if layer["type"] in ("conv", "dense"):
            W = parameters[f"{layer['name']}_W"]
            l2_penalty += np.sum(W ** 2)
    l2_penalty *= weight_decay / (2.0 * m)

    return cross_entropy + l2_penalty


# =============================================================================
# 10. Parameter initialisation
# =============================================================================

def initialize_parameters_cnn(architecture, input_shape=(64, 64, 3)):
    """
    He-initialise parameters for a CNN defined by an architecture list.

    Each entry in *architecture* is a dict, e.g.:
        {"type": "conv",  "name": "conv1", "filters": 16, "kernel": 3,
         "stride": 1, "padding": "same"}
        {"type": "relu",  "name": "relu1"}
        {"type": "pool",  "name": "pool1", "pool_size": 2, "stride": 2}
        {"type": "flatten","name": "flat"}
        {"type": "dense", "name": "fc1", "units": 256, "activation": "relu"}
        {"type": "dense", "name": "fc2", "units": 1,   "activation": "sigmoid"}
    """
    np.random.seed(1)
    parameters = {}
    prev_shape = input_shape                         # (H, W, C) → (feats,) after flatten

    for layer in architecture:
        lt = layer["type"]
        name = layer["name"]

        if lt == "conv":
            filters = layer["filters"]
            k = layer["kernel"]
            stride = layer.get("stride", 1)
            padding = layer.get("padding", "same")

            C = prev_shape[2]
            scale = np.sqrt(2.0 / (k * k * C))
            parameters[f"{name}_W"] = np.random.randn(filters, k, k, C) * scale
            parameters[f"{name}_b"] = np.zeros(filters)

            if padding == "same":
                out_h, out_w = prev_shape[0], prev_shape[1]
            else:
                out_h = (prev_shape[0] - k) // stride + 1
                out_w = (prev_shape[1] - k) // stride + 1
            prev_shape = (out_h, out_w, filters)

        elif lt == "pool":
            ps = layer["pool_size"]
            stride = layer.get("stride", ps)
            out_h = (prev_shape[0] - ps) // stride + 1
            out_w = (prev_shape[1] - ps) // stride + 1
            prev_shape = (out_h, out_w, prev_shape[2])

        elif lt == "flatten":
            prev_shape = (prev_shape[0] * prev_shape[1] * prev_shape[2],)

        elif lt == "dense":
            units = layer["units"]
            n_in = prev_shape[0]
            scale = np.sqrt(2.0 / n_in)
            parameters[f"{name}_W"] = np.random.randn(units, n_in) * scale
            parameters[f"{name}_b"] = np.zeros((units, 1))
            prev_shape = (units,)

        elif lt == "dropout":
            continue  # no parameters, no shape change

        # relu: no parameters, shape unchanged

    return parameters


# =============================================================================
# 11. Forward / backward orchestration
# =============================================================================

def cnn_forward(X, parameters, architecture, training=True):
    """
    Full forward pass through a CNN defined by *architecture*.

    Parameters
    ----------
    X : ndarray  shape (m, H, W, C)  — batch-first
    parameters : dict
    architecture : list of layer-spec dicts
    training : bool  if True, dropout is active; if False, it's a no-op

    Returns
    -------
    AL : ndarray  final output (1, m)
    caches : list of (layer_type, cache) tuples
    """
    caches = []
    A = X

    for layer in architecture:
        lt = layer["type"]
        name = layer["name"]

        if lt == "conv":
            W = parameters[f"{name}_W"]
            b = parameters[f"{name}_b"]
            stride = layer.get("stride", 1)
            padding = layer.get("padding", "same")
            A, cache = conv2d_forward(A, W, b, stride, padding)
            caches.append(("conv", cache))

        elif lt == "relu":
            A, cache = relu(A)
            caches.append(("relu", cache))

        elif lt == "pool":
            ps = layer["pool_size"]
            stride = layer.get("stride", ps)
            A, cache = maxpool_forward(A, ps, stride)
            caches.append(("pool", cache))

        elif lt == "flatten":
            A, cache = flatten_forward(A)
            caches.append(("flatten", cache))

        elif lt == "dense":
            W = parameters[f"{name}_W"]
            b = parameters[f"{name}_b"]
            A, cache = linear_activation_forward(A, W, b, layer["activation"])
            caches.append(("dense", cache))

        elif lt == "dropout":
            keep_prob = layer.get("keep_prob", 0.5) if training else 1.0
            A, cache = dropout_forward(A, keep_prob)
            caches.append(("dropout", cache))

    return A, caches


def cnn_backward(AL, Y, caches, architecture):
    """
    Full backward pass through a CNN.

    Parameters
    ----------
    AL : ndarray  final output (1, m)
    Y  : ndarray  ground-truth labels (1, m)
    caches : list  from cnn_forward
    architecture : list of layer-spec dicts

    Returns
    -------
    grads : dict  {name_W: dW, name_b: db, ...}
    """
    grads = {}
    Y = Y.reshape(AL.shape)

    # Initial gradient of cross-entropy w.r.t. sigmoid output
    dA = -(np.divide(Y, AL + 1e-12) - np.divide(1 - Y, 1 - AL + 1e-12))

    for l in reversed(range(len(architecture))):
        layer = architecture[l]
        lt = layer["type"]
        name = layer["name"]
        ctype, cache = caches[l]

        if lt == "dense":
            dA, dW, db = linear_activation_backward(
                dA, cache, layer["activation"])
            grads[f"{name}_W"] = dW
            grads[f"{name}_b"] = db

        elif lt == "flatten":
            dA = flatten_backward(dA, cache)

        elif lt == "pool":
            dA = maxpool_backward(dA, cache)

        elif lt == "relu":
            dA = relu_backward(dA, cache)

        elif lt == "conv":
            dA, dW, db = conv2d_backward(dA, cache)
            grads[f"{name}_W"] = dW
            grads[f"{name}_b"] = db

        elif lt == "dropout":
            dA = dropout_backward(dA, cache)

    return grads


# =============================================================================
# 11. Optimisers
# =============================================================================

def update_parameters_gd(parameters, grads, architecture, learning_rate,
                         weight_decay=0.0):
    """Vanilla gradient-descent update with optional decoupled weight decay."""
    for layer in architecture:
        if layer["type"] in ("conv", "dense"):
            name = layer["name"]
            dW = grads[f"{name}_W"] + weight_decay * parameters[f"{name}_W"]
            parameters[f"{name}_W"] -= learning_rate * dW
            parameters[f"{name}_b"] -= learning_rate * grads[f"{name}_b"]
    return parameters


def initialize_adam(parameters, architecture):
    """Initialise Adam moment estimates."""
    v, s = {}, {}
    for layer in architecture:
        if layer["type"] in ("conv", "dense"):
            name = layer["name"]
            v[f"{name}_W"] = np.zeros_like(parameters[f"{name}_W"])
            v[f"{name}_b"] = np.zeros_like(parameters[f"{name}_b"])
            s[f"{name}_W"] = np.zeros_like(parameters[f"{name}_W"])
            s[f"{name}_b"] = np.zeros_like(parameters[f"{name}_b"])
    return v, s


def update_parameters_adam(parameters, grads, v, s, t, architecture,
                           learning_rate=0.001, beta1=0.9, beta2=0.999,
                           epsilon=1e-8, weight_decay=0.0):
    """Adam update with optional decoupled weight decay (AdamW-style)."""
    for layer in architecture:
        if layer["type"] in ("conv", "dense"):
            name = layer["name"]
            # Decoupled weight decay: add λ·W to the gradient
            dW = grads[f"{name}_W"] + weight_decay * parameters[f"{name}_W"]
            db = grads[f"{name}_b"]

            # Moving averages
            v[f"{name}_W"] = beta1 * v[f"{name}_W"] + (1 - beta1) * dW
            v[f"{name}_b"] = beta1 * v[f"{name}_b"] + (1 - beta1) * db
            s[f"{name}_W"] = beta2 * s[f"{name}_W"] + (1 - beta2) * (dW ** 2)
            s[f"{name}_b"] = beta2 * s[f"{name}_b"] + (1 - beta2) * (db ** 2)

            # Bias-corrected estimates
            vw = v[f"{name}_W"] / (1 - beta1 ** t)
            vb = v[f"{name}_b"] / (1 - beta1 ** t)
            sw = s[f"{name}_W"] / (1 - beta2 ** t)
            sb = s[f"{name}_b"] / (1 - beta2 ** t)

            parameters[f"{name}_W"] -= learning_rate * vw / (np.sqrt(sw) + epsilon)
            parameters[f"{name}_b"] -= learning_rate * vb / (np.sqrt(sb) + epsilon)

    return parameters, v, s


# =============================================================================
# 12. Training loop
# =============================================================================

def cnn_model(X, Y, architecture,
              learning_rate=0.001, num_epochs=100, batch_size=64,
              optimizer="adam", beta1=0.9, beta2=0.999,
              weight_decay=0.0, augment=True, print_cost=True,
              print_every=10):
    """
    Train a CNN defined by *architecture*.

    Parameters
    ----------
    X : ndarray  shape (m, H, W, C)
    Y : ndarray  shape (1, m)
    architecture : list of layer-spec dicts
    optimizer : 'adam' or 'gd'
    batch_size : int   mini-batch size
    num_epochs : int
    learning_rate : float
    print_cost : bool
    print_every : int   print cost every N epochs
    weight_decay : float  L2 penalty coefficient (0 = no regularisation)
    augment : bool  if True, apply random horizontal flip to each batch

    Returns
    -------
    parameters : dict
    costs : list of per-epoch costs
    """
    np.random.seed(1)
    m = X.shape[0]
    costs = []

    parameters = initialize_parameters_cnn(
        architecture, input_shape=X.shape[1:])

    if optimizer == "adam":
        v, s = initialize_adam(parameters, architecture)
        t = 0

    for epoch in range(num_epochs):
        # Shuffle
        perm = np.random.permutation(m)
        X_s = X[perm]
        Y_s = Y[:, perm]

        epoch_cost = 0.0
        num_batches = 0

        for i in range(0, m, batch_size):
            Xb = X_s[i:i + batch_size]
            Yb = Y_s[:, i:i + batch_size]

            # Random horizontal flip augmentation (50% chance per image)
            if augment:
                flip_mask = np.random.rand(Xb.shape[0]) < 0.5
                if np.any(flip_mask):
                    Xb = Xb.copy()
                    Xb[flip_mask] = Xb[flip_mask, :, ::-1, :]

            AL, caches = cnn_forward(Xb, parameters, architecture, training=True)
            cost = compute_cost_with_l2(AL, Yb, parameters, architecture, weight_decay)
            epoch_cost += cost
            num_batches += 1

            grads = cnn_backward(AL, Yb, caches, architecture)

            if optimizer == "adam":
                t += 1
                parameters, v, s = update_parameters_adam(
                    parameters, grads, v, s, t, architecture, learning_rate,
                    beta1, beta2, weight_decay=weight_decay)
            else:
                parameters = update_parameters_gd(
                    parameters, grads, architecture, learning_rate,
                    weight_decay=weight_decay)

        avg_cost = epoch_cost / num_batches
        costs.append(avg_cost)

        if print_cost and (epoch % print_every == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:4d}  cost = {avg_cost:.6f}")

    return parameters, costs


# =============================================================================
# 13. Prediction helpers
# =============================================================================

def cnn_predict(X, y, parameters, architecture):
    """
    Evaluate the CNN on dataset (X, y).  Dropout is OFF during evaluation.

    Returns
    -------
    p : ndarray  shape (1, m)  binary predictions
    """
    AL, _ = cnn_forward(X, parameters, architecture, training=False)
    p = (AL > 0.5).astype(int)
    acc = np.mean(p == y)
    print(f"Accuracy: {acc:.4f}")
    return p


def cnn_predict_single(X, parameters, architecture):
    """
    Predict for a single example  (1, H, W, C).  Dropout is OFF.

    Returns (prediction, confidence).
    """
    AL, _ = cnn_forward(X, parameters, architecture, training=False)
    pred = int(AL[0, 0] > 0.5)
    conf = float(AL[0, 0]) if pred == 1 else float(1 - AL[0, 0])
    return pred, conf


# =============================================================================
# 14. Visualisation
# =============================================================================

def print_mislabeled_images(classes, X, y, p):
    """
    Plot images where the prediction differs from the true label.

    X : (m, 64, 64, 3)  or  (12288, m)  (tries to handle both)
    y, p : (1, m)
    """
    # Support both (m,H,W,C) and (features,m) shapes
    if X.ndim == 4:
        X_disp = X
    else:
        X_disp = X.T.reshape(-1, 64, 64, 3)

    a = p + y
    mis_idx = np.asarray(np.where(a == 1))
    num = len(mis_idx[0])
    if num == 0:
        print("No mislabeled images!")
        return

    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    for i in range(num):
        idx = mis_idx[1][i]
        plt.subplot(2, num, i + 1)
        plt.imshow(X_disp[idx])
        plt.axis('off')
        lbl_pred = classes[int(p[0, idx])].decode("utf-8")
        lbl_true = classes[int(y[0, idx])].decode("utf-8")
        plt.title(f"Pred: {lbl_pred}\nTrue: {lbl_true}")
