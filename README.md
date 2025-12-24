# Deep Neural Network from Scratch

A 4-layer neural network built entirely from scratch using NumPy for binary image classification (cat vs non-cat). Deployed as a full-stack web application with real-time monitoring.

🔗 **Live Demo:** [cat-classifier-899084497532.asia-southeast1.run.app](https://cat-classifier-899084497532.asia-southeast1.run.app)

📊 **Monitoring Dashboard:** [cat-classifier-899084497532.asia-southeast1.run.app/dashboard](https://cat-classifier-899084497532.asia-southeast1.run.app/dashboard)

## Media

<img width="2560" height="1600" alt="Screenshot 2025-12-24 174143" src="https://github.com/user-attachments/assets/6833873b-e6d4-4081-a839-7fcf37ebcaad" />

<img width="2560" height="1600" alt="Screenshot 2025-12-24 174131" src="https://github.com/user-attachments/assets/8f375a97-0516-40bb-b5f1-926010a6cec2" />

## Overview

This project implements a deep neural network without using any deep learning frameworks. The model is deployed as a REST API with a web interface for image upload and a monitoring dashboard for tracking model performance.

### Architecture

```
Input (12288) → Dense (20, ReLU) → Dense (7, ReLU) → Dense (5, ReLU) → Output (1, Sigmoid)
```

## Features

### Neural Network (from scratch)
- Forward propagation with vectorized operations
- Backpropagation with gradient computation  
- He initialization for weights
- Gradient descent optimization
- Binary cross-entropy loss

### Web Application
- Drag & drop image upload
- Real-time prediction with confidence scores
- Responsive design

### Monitoring Dashboard
- Prediction counts and cat/non-cat ratio
- Latency metrics (avg, P50, P95, P99)
- Confidence score distribution
- Data drift detection
- Recent predictions log

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
├── main.py             # FastAPI application
├── inference.py        # Model inference functions
├── monitoring.py       # Performance tracking
├── index.html          # Frontend UI
├── dashboard.html      # Monitoring dashboard
├── Dockerfile
├── requirements.txt
└── model/
    └── parameters.pkl  # Trained model weights
```

## Local Development

### Prerequisites
- Python 3.11+
- Docker (optional)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/pyaesonep/custom-nn.git
cd custom-nn

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Visit `http://localhost:8080`

### Run with Docker

```bash
# Build
docker build -t cat-classifier .

# Run
docker run -p 8080:8080 cat-classifier
```

## Deployment

### Deploy to Cloud Run via Docker Hub

1. **Build and push to Docker Hub:**
```bash
docker build -t yourusername/cat-classifier:latest .
docker push yourusername/cat-classifier:latest
```

2. **Deploy on Cloud Run Console:**
   - Go to [Cloud Run Console](https://console.cloud.google.com/run)
   - Click "Create Service"
   - Select "Deploy from existing container image"
   - Enter: `docker.io/yourusername/cat-classifier:latest`
   - Set region and allow unauthenticated access
   - Deploy

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/dashboard` | Monitoring dashboard |
| GET | `/health` | Health check |
| POST | `/predict` | Upload image for classification |
| GET | `/api/monitor/report` | Full monitoring metrics |

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Test Accuracy | ~80% |
| Avg Latency | <50ms |

## How It Works

### Forward Propagation

For each layer l:
- Z[l] = W[l] · A[l-1] + b[l]
- A[l] = g(Z[l])

### Backpropagation

Gradients computed via chain rule:
- dW[l] = (1/m) · dZ[l] · A[l-1]ᵀ
- db[l] = (1/m) · Σ dZ[l]
- dA[l-1] = W[l]ᵀ · dZ[l]

## Acknowledgments

This project was completed as part of the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) by DeepLearning.AI on Coursera. The neural network implementation is based on the course assignments.

## License

MIT License
