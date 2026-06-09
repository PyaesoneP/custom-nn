# Convolutional Neural Network from Scratch

A deep CNN for binary image classification (cat vs non-cat). Built with NumPy: no deep learning frameworks. Deployed as a web app with live monitoring.

**Live Demo:** [cat-classifier-899084497532.asia-southeast1.run.app](https://cat-classifier-899084497532.asia-southeast1.run.app)

**Monitoring Dashboard:** [cat-classifier-899084497532.asia-southeast1.run.app/dashboard](https://cat-classifier-899084497532.asia-southeast1.run.app/dashboard)

## Media

![CNN Classifier Demo](assets/cnn%20classifier.gif)

## Overview

A deep neural network implemented from scratch using only NumPy. The model is served as a REST API with a browser interface for uploading images and a dashboard for tracking predictions in real time.

### Architecture

```

Input (64×64×3)
→ Conv2D(16, 3×3) → ReLU
→ Conv2D(16, 3×3) → ReLU
→ MaxPool(2×2)      → (32×32×16)
→ Conv2D(32, 3×3) → ReLU
→ Conv2D(32, 3×3) → ReLU
→ MaxPool(2×2)      → (16×16×32)
→ Flatten           → (8192)
→ Dense(128, ReLU)
→ Dense(64, ReLU)
→ Dense(1, Sigmoid)

```

**~1.07M parameters** | 4 convolutional layers | 3 fully-connected layers

> **Sweet-spot configuration:** arrived at after systematic regularisation experiments
> (see [§5 Challenges](#5-challenges-and-lessons-learned-the-overfitting-problem)).
> Regularised with on-the-fly horizontal flip augmentation and light L2 weight decay
> ($\lambda = 5\times10^{-5}$).  No dropout: counterproductive on small dense layers.
> Training: Adam ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$), 120 epochs, batch size 32.

### How It Works: Forward and Backward Propagation in Detail

This section walks through every operation in the CNN, from input image to gradient update, with exact shapes at each step.  All variables use the following conventions:

| Symbol | Meaning | Example shape |
|--------|---------|---------------|
| $m$ | batch size | 32 |
| $H, W, C$ | height, width, channels of a feature map | 64, 64, 3 |
| $f_h, f_w$ | kernel spatial size | 3, 3 |
| $F$ | number of filters (output channels) | 16 |
| $S$ | stride | 1 |
| $P$ | zero-padding per side | 1 (for "same") |
| $n^{[l]}$ | number of units in dense layer $l$ | 256 |

---

#### 4.1  Conv2D Layer

##### Forward pass: $Z = A_{\text{prev}} \ast W + b$

Direct convolution with nested loops is too slow in pure Python.  Instead I use the **im2col** trick:

1. **Pad** the input with $P$ zeros on each spatial side:

   $$A_{\text{pad}} \in \mathbb{R}^{m \times (H+2P) \times (W+2P) \times C}$$

2. **Extract patches:** for every output position $(i, j)$, grab the $f_h \times f_w \times C$ receptive field:

   $$X_{\text{col}} \in \mathbb{R}^{(m \cdot H_{\text{out}} \cdot W_{\text{out}}) \;\times\; (f_h \cdot f_w \cdot C)}$$

   Each row of $X_{\text{col}}$ is one flattened image patch.  The output spatial dimensions are:

   $$
   H_{\text{out}} = \left\lfloor\frac{H + 2P - f_h}{S}\right\rfloor + 1, \qquad
   W_{\text{out}} = \left\lfloor\frac{W + 2P - f_w}{S}\right\rfloor + 1
   $$

3. **Matrix multiply:** reshape the filter bank $W \in \mathbb{R}^{F \times f_h \times f_w \times C}$ into a matrix $W_{\text{col}} \in \mathbb{R}^{(f_h f_w C) \times F}$, then:

   $$Z_{\text{col}} = X_{\text{col}} \, W_{\text{col}} + b \qquad b \in \mathbb{R}^{F}$$

4. **Reshape** back to 4-D:

   $$Z \in \mathbb{R}^{m \times H_{\text{out}} \times W_{\text{out}} \times F}$$

For `padding = "same"` with $S = 1$, $P = \lfloor f_h/2 \rfloor$ so that $H_{\text{out}} = H$ and $W_{\text{out}} = W$.

##### Backward pass: gradients w.r.t. $W$, $b$, and $A_{\text{prev}}$

Given the upstream gradient $\frac{\partial \mathcal{L}}{\partial Z}$ of shape $(m, H_{\text{out}}, W_{\text{out}}, F)$, flatten it to $\mathrm{d}Z_{\text{col}} \in \mathbb{R}^{(m H_{\text{out}} W_{\text{out}}) \times F}$.

| Gradient | Formula | Shape |
|----------|---------|-------|
| $\displaystyle\frac{\partial \mathcal{L}}{\partial W}$ | $\mathrm{d}W_{\text{col}} = X_{\text{col}}^\top \; \mathrm{d}Z_{\text{col}}$, then reshape to $(F, f_h, f_w, C)$ | same as $W$ |
| $\displaystyle\frac{\partial \mathcal{L}}{\partial b}$ | $\mathrm{d}b = \displaystyle\sum_{m,\,H_{\text{out}},\,W_{\text{out}}} \mathrm{d}Z$ | $(F,)$ |
| $\displaystyle\frac{\partial \mathcal{L}}{\partial A_{\text{prev}}}$ | $\mathrm{d}X_{\text{col}} = \mathrm{d}Z_{\text{col}} \; W_{\text{col}}^\top$, then **col2im** scatters the rows back to 4-D | $(m, H, W, C)$ |

**col2im** is the exact inverse of im2col. It takes each row of $\mathrm{d}X_{\text{col}}$, reshapes it to $(f_h, f_w, C)$, and *adds* it to the correct spatial location in the (padded) gradient map.  Regions visited multiple times (when $S < f_h$) accumulate gradients.  After filling the padded array, the padding strip is sliced off to recover the original $(H, W)$ dimensions.

---

#### 4.2  ReLU Activation

$$A = \max(0, Z)$$

**Backward**:  $\displaystyle\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial A} \odot \mathbf{1}[Z > 0]$

The gradient passes through unchanged where $Z > 0$, and is zeroed out everywhere else.  This is an element-wise operation, so shapes are preserved.

---

#### 4.3  MaxPool Layer

##### Forward pass

For a pool window of size $p \times p$ and stride $S$:

$$A_{i,j} = \max_{(u,v) \in \text{window}} A_{\text{prev}}[\,u, v\,]$$

While computing the max, I record the **argmax position** inside each window in a binary mask $M$ (same shape as $A_{	ext{prev}}$) where $M = 1$ at every winning pixel and $0$ elsewhere.

Output shape:

$$
H_{\text{out}} = \left\lfloor\frac{H - p}{S}\right\rfloor + 1, \qquad
W_{\text{out}} = \left\lfloor\frac{W - p}{S}\right\rfloor + 1, \qquad C_{\text{out}} = C
$$

##### Backward pass

The upstream gradient $\frac{\partial \mathcal{L}}{\partial A}$ lives on the down-sampled grid.  To propagate it back:

1. **Upsample:** for each output position, broadcast its scalar gradient across the entire $p \times p$ window.
2. **Mask:** multiply element-wise by $M$, so the gradient only reaches the pixel that actually won the max.

$$
\frac{\partial \mathcal{L}}{\partial A_{\text{prev}}} =
\text{upsample}\!\left(\frac{\partial \mathcal{L}}{\partial A}\right) \;\odot\; M
$$

This is the standard "max-pooling switch" technique. Only the neuron that fired in the forward pass receives a gradient in the backward pass.

---

#### 4.4  Flatten Layer

##### Forward

$$
A_{\text{flat}} = \text{reshape}\!\left(A_{\text{prev}},\; (m,\; HWC)\right)^\top
\;\in\; \mathbb{R}^{HWC \;\times\; m}
$$

The transpose puts features along rows and examples along columns, which matches the convention used by the Dense layers.

##### Backward

$$
\frac{\partial \mathcal{L}}{\partial A_{\text{prev}}} =
\text{reshape}\!\left(
  \left(\frac{\partial \mathcal{L}}{\partial A_{\text{flat}}}\right)^\top,\;
  (m, H, W, C)
\right)
$$

The original 4-D shape is cached during the forward pass so the backward pass can restore it exactly.

---

#### 4.5  Dense (Fully-Connected) Layer

A dense layer computes `LINEAR → ACTIVATION`.

##### Linear forward

$$Z = W A_{\text{prev}} + b, \qquad W \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}},\; b \in \mathbb{R}^{n^{[l]} \times 1}$$

##### Activation forward

$$
A = g(Z) \quad\text{where}\quad
g(z) = \begin{cases}
  \max(0, z)            & \text{ReLU (hidden layers)} \\[4pt]
  \dfrac{1}{1 + e^{-z}} & \text{Sigmoid (output layer)}
\end{cases}
$$

##### Linear backward

Given $\mathrm{d}Z = \frac{\partial \mathcal{L}}{\partial Z}$:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{m}\; \mathrm{d}Z \; A_{\text{prev}}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{\text{cols}} \mathrm{d}Z, \qquad
\frac{\partial \mathcal{L}}{\partial A_{\text{prev}}} = W^\top \mathrm{d}Z
$$

The $\frac{1}{m}$ factor averages the gradient over the mini-batch so the weight update scale is independent of batch size.

##### Activation backward

- **Sigmoid**:  $\displaystyle\mathrm{d}Z = \mathrm{d}A \;\odot\; \sigma(Z) \odot (1 - \sigma(Z))$
- **ReLU**:     $\displaystyle\mathrm{d}Z = \mathrm{d}A \;\odot\; \mathbf{1}[Z > 0]$

---

#### 4.6  Loss Function and Initial Gradient

I use **binary cross-entropy** for this two-class problem:

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \Big[
  y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})
\Big]
$$

where $\hat{y} = A^{[L]}$ is the Sigmoid output of the final layer.

The backward pass is seeded with the gradient of $\mathcal{L}$ w.r.t. the output activation:

$$
\frac{\partial \mathcal{L}}{\partial A^{[L]}} =
-\left(\frac{Y}{A^{[L]}} - \frac{1 - Y}{1 - A^{[L]}}\right)
\;\in\; \mathbb{R}^{1 \times m}
$$

From here, the chain rule propagates this gradient backward through every layer in **reverse architecture order**: Dense → Flatten → MaxPool → ReLU → Conv2D. Each layer computes $\frac{\partial \mathcal{L}}{\partial A_{\text{prev}}}$ for the layer before it and $\frac{\partial \mathcal{L}}{\partial W}, \frac{\partial \mathcal{L}}{\partial b}$ for its own parameters.

---

#### 4.7  Parameter Update (Adam)

Once all gradients are collected, Adam updates each trainable parameter $\theta$:

$$v_t = \beta_1 v_{t-1} + (1 - \beta_1)\,\nabla_\theta\mathcal{L}$$

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2)\,(\nabla_\theta\mathcal{L})^2$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}, \qquad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

$$\theta \leftarrow \theta - \alpha\;\frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

Default hyper-parameters: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.  Vanilla gradient descent is also available as a simpler alternative.

---

### 5.  Challenges and Lessons Learned: The Overfitting Problem

The dataset from the Deep Learning Specialization contains only **209 training images** and **50 test images** at 64×64 resolution.  This section documents every attempt to close the gap between training and test accuracy, what worked, what backfired, and why.

#### 5.1  Baseline: Original Dense Network

| Model | Train Acc | Test Acc | Params |
|-------|-----------|----------|--------|
| 4-layer DNN  `(12288→20→7→5→1)` | ~98% | **~80%** | ~250K |

The dense network flattens the image immediately, discarding all spatial structure.  It cannot learn translation-invariant features, which limits its ceiling on this task.

#### 5.2  Attempt 1: Deep CNN (No Regularisation)

| Model | Train Acc | Test Acc | Params |
|-------|-----------|----------|--------|
| 4×Conv(16/16/32/32) + Dense(256/128/1) | **100%** | 84% | 2.1M |

**What worked**: The CNN learned spatial hierarchies. The conv filters detected edges and textures that the dense network missed.  Test accuracy improved from 80% to 84%.

**What didn't**:  2.1 million parameters on 209 images is ~10,000× more parameters than examples.  The model **memorised the training set perfectly** (100% accuracy) while the test accuracy stalled.  The $256 \times 8192$ weight matrix in the first dense layer alone contributed 2M of the 2.1M parameters. Nearly all the overfitting originated there.

#### 5.3  Attempt 2: Dropout + L2 + Smaller Architecture Backfired

| Model | Train Acc | Test Acc | Params |
|-------|-----------|----------|--------|
| 4×Conv(8/8/16/16) + Dropout(0.5) + Dense(64/32/1) + L2(1e-4) | 100% | **66%** | 269K |

**What I tried**: Cut conv filters by 50%, shrunk dense layers from 256/128 to 64/32, added inverted dropout (keep_prob=0.5) after each dense layer, and applied AdamW-style decoupled weight decay at $\lambda = 10^{-4}$.

**Why it failed**: The regularisation was **too aggressive for the model capacity that remained**.

- **Dropout at 0.5 on 64-unit and 32-unit layers** left only 32 and 16 active neurons per forward pass, which is too few to form useful representations.  Dropout is designed for large fully-connected layers (≥256 units). On small layers it destroys signal faster than it prevents co-adaptation.
- **8 filters in the first conv layer** could only detect 8 distinct low-level features (edges, colour blobs, gradients).  At 64×64×3 input, this is insufficient to build a rich enough feature hierarchy for the later layers to exploit.
- **Combined effect**: the model was both under-parameterised (cannot learn enough) *and* over-regularised (cannot use what little it learned).  It converged to a degenerate solution that memorised the training set through a narrow set of brittle features that didn't generalise at all.

**Lesson**: Regularisation must be proportional to model capacity.  Never apply aggressive dropout to layers with fewer than ~128 units.

#### 5.4  Attempt 3: Restored Conv Filters + Augmentation + Light L2

| Model | Train Acc | Test Acc | Params |
|-------|-----------|----------|--------|
| 4×Conv(16/16/32/32) + Dense(128/64/1) + L2(5e-5) + horizontal flip | 100% | **86%** | 1.07M |

**What I tried**:
- **Restored conv filters** to 16/16/32/32. The model needs capacity in the convolutional stem to learn useful spatial features.
- **Reduced dense layers** from 256/128 to 128/64. This focused parameter reduction where overfitting is worst (the FC layers hold the majority of parameters: $128 \times 4096 = 524\text{K}$ vs the original $256 \times 8192 = 2.1\text{M}$).
- **Removed dropout entirely.** With 128 and 64 units, dropout is more harmful than helpful.
- **Light L2 weight decay at $5\times10^{-5}$.** Gentle regularisation that nudges weights toward zero without crippling learning.
- **On-the-fly horizontal flip.** Randomly flips 50% of images in each batch, giving the model left/right invariance without inflating the dataset on disk.

**Result**: Test accuracy improved to 86%, the best so far.  The conv layers had enough capacity, the dense layers were restrained but not starved, and the augmentation provided meaningful regularisation without destroying signal.

#### 5.5  Attempt 4: Pre-Generated 4× Augmented Dataset

| Model | Train Acc | Test Acc | Params |
|-------|-----------|----------|--------|
| Same as Attempt 3 + 4× pre-augmented data (flip, rotate ±12°, brightness ±20%, zoom crop) | 100% | **84%** | 1.07M |

**What I tried**: Instead of on-the-fly flips, I pre-generated 4 variants of every training image (original, flipped, rotated + brightness-jittered, zoom-cropped + flipped), expanding the training set from 209 to 836 images.

**Why it didn't help more**: The augmented images, while diverse, are still derived from the same 209 source images taken under the same lighting conditions, camera angle, and background.  The model had already learned to handle flips (from attempt 3's on-the-fly augmentation).  Rotations and crops introduced new variations, but the **50-image test set is so small** that accuracy measurements have high variance. A single misclassified image shifts accuracy by 2%.  The true generalisation gain from the extra augmentations is likely real but drowned out by test-set noise.

#### 5.6  Why 100% Train Accuracy Persists

After four iterations of regularisation, the training accuracy remains pinned at 100% while test accuracy hovers around 84–86%.  This is not a failure of the regularisation techniques. It is a **fundamental data problem**:

1. **209 images is extremely small** for any CNN.  State-of-the-art image classifiers train on millions of images (ImageNet has 1.2M).  At this scale, the model *will* memorise the training set because there simply aren't enough examples to force it to learn general features rather than instance-specific patterns.

2. **The 50-image test set is too small for reliable measurement**.  With only 50 examples, the 95% confidence interval for accuracy is roughly $\pm 10\%$.  An 84% measurement could correspond to a true accuracy anywhere from 74% to 94%.

3. **The images are homogeneous.** They share the same resolution (64×64), lighting, framing, and background.  There's limited diversity for the model to learn invariance from.

4. **Binary classification on a balanced set** means random guessing gives 50%.  An 84–86% accuracy represents a real but modest improvement over chance, consistent with a model that has learned some general features but still relies partially on dataset-specific shortcuts.

#### 5.7  What Would Actually Help (Beyond This Project's Scope)

| Approach | Expected Impact | Why Not Done |
|----------|----------------|-------------|
| **Larger dataset** (e.g., Kaggle Dogs vs Cats — 25K images) | +10–15% accuracy | Different data source; would make this a different project |
| **Transfer learning** (pretrained ResNet/VGG features) | +10–20% accuracy | Violates the "no frameworks" constraint — those weights come from PyTorch/TensorFlow models |
| **Higher resolution** (128×128 or 224×224) | +3–5% accuracy | Requires larger conv filters and more memory; training time grows quadratically |
| **Cross-validation** (k-fold on the combined 259 images) | Better accuracy estimate | The 50-image test set is too small to split further; k-fold would give unstable per-fold estimates |
| **Test-time augmentation** (average predictions over 8 augmented versions of each test image) | +1–3% accuracy | Simple to add; predicts on flipped/rotated versions and averages the probabilities |

#### 5.8  Summary: What I Learned

| Principle | Details |
|-----------|---------|
| **Data > Architecture** | No amount of architectural tuning can compensate for a 209-image dataset.  Augmentation helps but cannot create truly new data. |
| **Regularise proportionally** | Dropout at 0.5 on 64-unit layers destroys signal.  L2 at $10^{-4}$ on a 2M-parameter model is negligible; the same L2 on a 250K model is crippling.  Match regularisation strength to parameter count. |
| **Convolutional capacity matters** | 8 filters in conv1 is too few for 64×64 RGB input.  16 is a reasonable minimum.  Spatial feature learning needs headroom. |
| **FC layers are the overfitting bottleneck** | The $HWC \times n^{[l]}$ weight matrix in the first dense layer dominates the parameter count.  Reducing FC units and adding pooling (to shrink the spatial dimensions before flattening) are the most impactful levers. |
| **Test-set size limits conclusions** | With 50 test images, you cannot reliably distinguish between 84% and 88% accuracy.  Don't over-optimise for the last 2–3%. |

## Features

### Neural Network (from scratch)
- Conv2D with im2col + matrix multiply (no loops)
- MaxPool with argmax routing in backward pass
- Forward/backward orchestration via architecture spec (list of layer dicts)
- He initialization for weights
- Adam optimiser with bias correction (vanilla GD also available)
- Binary cross-entropy loss

### Web Interface
- Editorial Brutalism design: dark theme, high contrast, sharp typography
- Two-column layout on desktop (upload left, result right)
- Drag and drop upload, paste to upload (Ctrl+V), and click to browse
- Real-time prediction with confidence score and latency display
- Title character reveal animation on load
- Magnetic button and custom cursor
- SVG noise texture overlay
- Preloader with progress bar
- Respects `prefers-reduced-motion`; full keyboard navigation

### Monitoring Dashboard
- Prediction counts and cat/non-cat ratio
- Latency percentiles (avg, P50, P95, P99) with color-coded bars
- Confidence distribution (bar chart) and prediction breakdown (doughnut chart)
- Data drift detection with baseline comparison
- Recent predictions log, auto-refreshes every 5 seconds
- Chart.js with brutalist dark theme

### Security
- **Rate limiting**: 30 POST requests per 60 seconds per IP
- **File size cap**: 5 MB maximum
- **Magic byte validation**: type checked by file headers, not MIME
- **No stack traces**: generic error messages only
- **XSS protection**: `textContent` rendering for user-provided data
- **Non-root container**: `USER app`, HEALTHCHECK, `.dockerignore`
- **Opt-in auth**: set `MONITORING_TOKEN` to protect dashboard endpoints

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | NumPy (CNN from scratch) |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Container | Docker |
| Deployment | Google Cloud Run |
| Registry | Docker Hub |

## Project Structure

```

├── custom-nn.ipynb     # Training notebook
├── utils.py            # Core neural network functions
├── main.py             # FastAPI application (rate-limited, secured)
├── inference.py        # Model inference functions
├── monitoring.py       # Performance tracking and drift detection
├── save_model.py       # Model serialization
├── index.html          # Classifier UI (Editorial Brutalism design)
├── dashboard.html      # Monitoring dashboard
├── Dockerfile          # Non-root container with HEALTHCHECK
├── .dockerignore
├── requirements.txt
├── datasets/           # Training/test data (HDF5)
├── logs/               # Prediction log (JSONL)
└── model/
└── parameters.pkl  # Trained model weights

```

## Local Development

### Prerequisites
- Python 3.11+
- Docker (optional)

### Run Locally

```bash
git clone https://github.com/pyaesonep/custom-nn.git
cd custom-nn
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8080`

### Run with Docker

```bash
docker build -t cat-classifier .
docker run -p 8080:8080 cat-classifier
```

## Deployment

### Deploy to Cloud Run via Docker Hub

1. Build and push:

```bash
docker build -t yourusername/cat-classifier:latest .
docker push yourusername/cat-classifier:latest
```

2. Deploy on Cloud Run:
* Go to [Cloud Run Console](https://console.cloud.google.com/run)
* Click "Create Service", select "Deploy from existing container image"
* Enter `docker.io/yourusername/cat-classifier:latest`
* Set region, allow unauthenticated access, deploy



## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Classifier web interface |
| GET | `/dashboard` | Monitoring dashboard |
| GET | `/health` | Health check |
| GET | `/predict` | Redirects to classifier |
| POST | `/predict` | Classify an image (rate-limited, 5 MB max) |
| GET | `/docs` | Swagger API docs |
| GET | `/api/monitor/report` | Full monitoring report |
| GET | `/api/monitor/summary` | Prediction summary |
| GET | `/api/monitor/performance` | Latency and throughput |
| GET | `/api/monitor/confidence` | Confidence distribution |
| GET | `/api/monitor/drift` | Drift metrics |
| GET | `/api/monitor/recent` | Recent predictions |

> Rate limit: 30 POST requests per 60 seconds per IP (returns `429`). File limit: 5 MB (returns `413`).

## Results

| Metric | Value |
| --- | --- |
| Parameters | ~1.07M |
| Architecture | 4×Conv2D (16/16/32/32) + 3×Dense (128/64/1) |
| Training images | 209 → 836 (4× augmentation) |
| Regularisation | L2 weight decay (5×10⁻⁵) + horizontal flip augmentation |
| Training Accuracy | 100% |
| Test Accuracy | ~84–86% |
| Avg Latency | <100ms |

> **Note on test accuracy**: The test set contains only 50 images.  With a sample this small, accuracy measurements have a ~±10% confidence interval.  The 84–86% range represents the model's true generalisation ceiling given the dataset size. See [§5 Challenges and Lessons Learned](#5-challenges-and-lessons-learned-the-overfitting-problem) for the full story.  The model reliably distinguishes cats from non-cats in practice, as verified by manual testing with diverse real-world images.

### Training Progression

| Iteration | Train Acc | Test Acc | Key Change |
| --- | --- | --- | --- |
| DNN baseline | 98% | 80% | 4-layer dense network |
| CNN v1 | 100% | 84% | 4×Conv + 3×Dense (2.1M params) |
| CNN v2 | 100% | 66% | Dropout 0.5 + tiny arch (269K): over-regularised |
| CNN v3 | 100% | 86% | Restored convs + light L2 + flip aug (1.07M) |
| CNN v4 | 100% | 84% | 4× pre-augmented dataset: marginal gain |

## Acknowledgments

Built as part of the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI on Coursera. The neural network implementation follows the course assignments.

## License

MIT License