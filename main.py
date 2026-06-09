"""FastAPI application for cat/non-cat image classification with monitoring."""

import io
import os
import time
import pickle
import secrets
from pathlib import Path
from collections import deque

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from inference import predict
from monitoring import monitor

# Load model parameters on startup
MODEL_PATH = Path("model/parameters.pkl")

app = FastAPI(
    title="Cat Classifier API",
    description="A CNN that classifies images as cat or non-cat. Built from scratch with NumPy.",
    version="2.0.0"
)

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global parameters and architecture loaded at startup
parameters = None
architecture = None

# --- Security: Rate Limiting (in-memory, per-IP) ---
RATE_WINDOW = 60       # seconds
RATE_MAX = 30           # max requests per window per IP
_rate_store = {}        # IP → deque of timestamps

def _check_rate(client_ip: str) -> bool:
    """Return True if request is within rate limit, False if exceeded."""
    now = time.time()
    if client_ip not in _rate_store:
        _rate_store[client_ip] = deque()
    window = _rate_store[client_ip]
    # Purge old entries
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_MAX:
        return False
    window.append(now)
    return True

# --- Security: File Size Limit ---
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# --- Security: Optional Monitoring Auth ---
MONITORING_TOKEN = os.getenv("MONITORING_TOKEN", None)

def _check_monitoring_auth(request: Request):
    """If MONITORING_TOKEN is set, require it in X-Monitoring-Token header."""
    if MONITORING_TOKEN:
        token = request.headers.get("X-Monitoring-Token", "")
        if not secrets.compare_digest(token, MONITORING_TOKEN):
            raise HTTPException(status_code=403, detail="Forbidden")

# --- Security: Magic Bytes for Image Validation ---
JPEG_MAGIC = {b'\xff\xd8\xff'}
PNG_MAGIC  = {b'\x89PNG\r\n\x1a\n'}

def _validate_image_magic(header: bytes) -> bool:
    """Check file header against known image magic bytes."""
    return header[:3] in JPEG_MAGIC or header[:8] == b'\x89PNG\r\n\x1a\n'


@app.on_event("startup")
def load_model():
    global parameters, architecture
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run save_model.py first.")
    
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple) and len(data) == 2:
        parameters, architecture = data
    else:
        raise RuntimeError(
            "Old DNN model detected. Please retrain with the CNN notebook."
        )

    # Count trainable layers for logging
    n_layers = sum(1 for l in architecture if l["type"] in ("conv", "dense"))
    total_params = sum(v.size for v in parameters.values())
    print(f"CNN loaded: {n_layers} trainable layers, {total_params:,} parameters")


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html>
        <body>
            <h1>Cat Classifier API</h1>
            <p>Endpoints:</p>
            <ul>
                <li><code>POST /predict</code> - Upload an image to classify</li>
                <li><code>GET /health</code> - Check API status</li>
                <li><code>GET /docs</code> - Interactive API documentation</li>
            </ul>
        </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the monitoring dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return dashboard_path.read_text()
    return "<html><body><h1>Dashboard not found</h1></body></html>"


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": parameters is not None}


@app.get("/predict", response_class=HTMLResponse)
def predict_page():
    """Redirect GET requests to /predict back to the classifier UI."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html><body><h1>Cat Classifier</h1>
    <p>Use the <a href="/">web interface</a> or send a POST request with an image file.</p>
    </body></html>"""


@app.get("/favicon.ico")
def favicon():
    """Silence favicon 404s."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/predict")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    Upload an image to classify as cat or non-cat.
    
    Accepts: JPEG, PNG images (validated by magic bytes, not Content-Type)
    Returns: prediction (cat/non-cat), confidence score
    """
    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")

    # Read file (with size cap)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5 MB.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Validate by magic bytes (not Content-Type, which is client-controlled)
    if not _validate_image_magic(contents):
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a JPEG or PNG image.")

    start_time = time.time()

    try:
        # Preprocess image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((64, 64))

        # Convert to numpy array and normalize — keep 4-D shape for CNN
        img_array = np.array(image) / 255.0           # (64, 64, 3)
        img_batch = img_array[np.newaxis, ...]         # (1, 64, 64, 3)

        # Predict
        prediction, confidence = predict(img_batch, parameters, architecture)
        label = "cat" if prediction == 1 else "non-cat"

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log to monitor
        monitor.log_prediction(
            filename=file.filename or "unknown",
            prediction=label,
            confidence=confidence,
            latency_ms=latency_ms,
            input_data=img_batch
        )

        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "filename": file.filename,
            "latency_ms": round(latency_ms, 2)
        }

    except HTTPException:
        raise  # re-raise known HTTP exceptions as-is
    except Exception:
        monitor.log_error()
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again with a different image.")


# ============== Monitoring Endpoints ==============

@app.get("/api/monitor/summary")
def get_summary(request: Request):
    """Get prediction summary statistics."""
    _check_monitoring_auth(request)
    return monitor.get_summary()


@app.get("/api/monitor/performance")
def get_performance(request: Request):
    """Get latency and throughput metrics."""
    _check_monitoring_auth(request)
    return monitor.get_performance_metrics()


@app.get("/api/monitor/confidence")
def get_confidence(request: Request):
    """Get confidence score distribution."""
    _check_monitoring_auth(request)
    return monitor.get_confidence_distribution()


@app.get("/api/monitor/drift")
def get_drift(request: Request):
    """Get data drift metrics."""
    _check_monitoring_auth(request)
    return monitor.get_drift_metrics()


@app.get("/api/monitor/recent")
def get_recent(request: Request, n: int = 20):
    """Get recent predictions."""
    _check_monitoring_auth(request)
    return monitor.get_recent_predictions(n)


@app.get("/api/monitor/report")
def get_report(request: Request):
    """Get full monitoring report."""
    _check_monitoring_auth(request)
    return monitor.get_full_report()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
