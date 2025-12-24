"""FastAPI application for cat/non-cat image classification with monitoring."""

import io
import time
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

from inference import predict
from monitoring import monitor

# Load model parameters on startup
MODEL_PATH = Path("model/parameters.pkl")

app = FastAPI(
    title="Cat Classifier API",
    description="A 4-layer neural network that classifies images as cat or non-cat. Built from scratch with NumPy.",
    version="1.0.0"
)

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global parameters loaded at startup
parameters = None


@app.on_event("startup")
def load_model():
    global parameters
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run save_model.py first.")
    
    with open(MODEL_PATH, "rb") as f:
        parameters = pickle.load(f)
    
    print(f"Model loaded with {len(parameters) // 2} layers")


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


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image to classify as cat or non-cat.
    
    Accepts: JPEG, PNG images
    Returns: prediction (cat/non-cat), confidence score
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a JPEG or PNG image."
        )
    
    start_time = time.time()
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((64, 64))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_flatten = img_array.reshape(-1, 1) / 255.0  # Shape: (12288, 1)
        
        # Predict
        prediction, confidence = predict(img_flatten, parameters)
        label = "cat" if prediction == 1 else "non-cat"
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to monitor
        monitor.log_prediction(
            filename=file.filename or "unknown",
            prediction=label,
            confidence=confidence,
            latency_ms=latency_ms,
            input_data=img_flatten
        )
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "filename": file.filename,
            "latency_ms": round(latency_ms, 2)
        }
    
    except Exception as e:
        monitor.log_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============== Monitoring Endpoints ==============

@app.get("/api/monitor/summary")
def get_summary():
    """Get prediction summary statistics."""
    return monitor.get_summary()


@app.get("/api/monitor/performance")
def get_performance():
    """Get latency and throughput metrics."""
    return monitor.get_performance_metrics()


@app.get("/api/monitor/confidence")
def get_confidence():
    """Get confidence score distribution."""
    return monitor.get_confidence_distribution()


@app.get("/api/monitor/drift")
def get_drift():
    """Get data drift metrics."""
    return monitor.get_drift_metrics()


@app.get("/api/monitor/recent")
def get_recent(n: int = 20):
    """Get recent predictions."""
    return monitor.get_recent_predictions(n)


@app.get("/api/monitor/report")
def get_report():
    """Get full monitoring report."""
    return monitor.get_full_report()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
