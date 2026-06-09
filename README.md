# Deep Neural Network from Scratch

A 4-layer neural network for binary image classification (cat vs non-cat). Built with NumPy. No deep learning frameworks. Deployed as a web app with live monitoring.

**Live Demo:** [cat-classifier-899084497532.asia-southeast1.run.app](https://cat-classifier-899084497532.asia-southeast1.run.app)

**Monitoring Dashboard:** [cat-classifier-899084497532.asia-southeast1.run.app/dashboard](https://cat-classifier-899084497532.asia-southeast1.run.app/dashboard)

## Media

<!-- Screenshots need updating to reflect current Editorial Brutalism redesign -->

<img width="2560" height="1600" alt="Screenshot 2025-12-24 174143" src="https://github.com/user-attachments/assets/6833873b-e6d4-4081-a839-7fcf37ebcaad" />

<img width="2560" height="1600" alt="Screenshot 2025-12-24 174131" src="https://github.com/user-attachments/assets/8f375a97-0516-40bb-b5f1-926010a6cec2" />

## Overview

A deep neural network implemented from scratch using only NumPy. The model is served as a REST API with a browser interface for uploading images and a dashboard for tracking predictions in real time.

### Architecture

```
Input (12288) -> Dense (20, ReLU) -> Dense (7, ReLU) -> Dense (5, ReLU) -> Output (1, Sigmoid)
```

### How It Works

**Forward propagation** for each layer l:
- Z[l] = W[l] * A[l-1] + b[l]
- A[l] = g(Z[l])

**Backpropagation** via chain rule:
- dW[l] = (1/m) * dZ[l] * A[l-1]^T
- db[l] = (1/m) * sum(dZ[l])
- dA[l-1] = W[l]^T * dZ[l]

## Features

### Neural Network (from scratch)
- Forward propagation with vectorized operations
- Backpropagation with gradient computation
- He initialization for weights
- Gradient descent optimization
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
| ML Model | NumPy |
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
   - Go to [Cloud Run Console](https://console.cloud.google.com/run)
   - Click "Create Service", select "Deploy from existing container image"
   - Enter `docker.io/yourusername/cat-classifier:latest`
   - Set region, allow unauthenticated access, deploy

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
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
|--------|-------|
| Training Accuracy | ~98% |
| Test Accuracy | ~80% |
| Avg Latency | <50ms |

## Acknowledgments

Built as part of the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI on Coursera. The neural network implementation follows the course assignments.

## License

MIT License
