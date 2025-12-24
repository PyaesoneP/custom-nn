"""Model monitoring module for tracking predictions and performance."""

import time
import json
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


@dataclass
class PredictionLog:
    timestamp: str
    filename: str
    prediction: str
    confidence: float
    latency_ms: float
    input_mean: float
    input_std: float


class ModelMonitor:
    """Tracks model predictions and performance metrics."""
    
    def __init__(self, max_history: int = 1000, log_file: Optional[str] = "logs/predictions.jsonl"):
        self.predictions = deque(maxlen=max_history)
        self.log_file = Path(log_file) if log_file else None
        
        # Counters
        self.total_predictions = 0
        self.cat_predictions = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
        # Performance tracking
        self.latencies = deque(maxlen=max_history)
        
        # Confidence buckets for distribution
        self.confidence_buckets = {
            "0.5-0.6": 0,
            "0.6-0.7": 0,
            "0.7-0.8": 0,
            "0.8-0.9": 0,
            "0.9-1.0": 0
        }
        
        # Input data statistics (for drift detection)
        self.input_means = deque(maxlen=max_history)
        self.input_stds = deque(maxlen=max_history)
        
        # Baseline stats (set after first N predictions)
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_set = False
        
        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(exist_ok=True)
    
    def log_prediction(
        self, 
        filename: str, 
        prediction: str, 
        confidence: float, 
        latency_ms: float,
        input_data: np.ndarray
    ):
        """Log a single prediction with metadata."""
        
        # Calculate input statistics
        input_mean = float(np.mean(input_data))
        input_std = float(np.std(input_data))
        
        log_entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            latency_ms=round(latency_ms, 2),
            input_mean=round(input_mean, 4),
            input_std=round(input_std, 4)
        )
        
        # Store in memory
        self.predictions.append(log_entry)
        self.latencies.append(latency_ms)
        self.input_means.append(input_mean)
        self.input_stds.append(input_std)
        
        # Update counters
        self.total_predictions += 1
        if prediction == "cat":
            self.cat_predictions += 1
        
        # Update confidence distribution
        self._update_confidence_bucket(confidence)
        
        # Set baseline after 50 predictions
        if not self.baseline_set and len(self.input_means) >= 50:
            self._set_baseline()
        
        # Persist to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(log_entry)) + "\n")
    
    def log_error(self):
        """Log a prediction error."""
        self.error_count += 1
    
    def _update_confidence_bucket(self, confidence: float):
        """Update confidence distribution buckets."""
        if confidence < 0.6:
            self.confidence_buckets["0.5-0.6"] += 1
        elif confidence < 0.7:
            self.confidence_buckets["0.6-0.7"] += 1
        elif confidence < 0.8:
            self.confidence_buckets["0.7-0.8"] += 1
        elif confidence < 0.9:
            self.confidence_buckets["0.8-0.9"] += 1
        else:
            self.confidence_buckets["0.9-1.0"] += 1
    
    def _set_baseline(self):
        """Set baseline statistics for drift detection."""
        self.baseline_mean = np.mean(list(self.input_means))
        self.baseline_std = np.mean(list(self.input_stds))
        self.baseline_set = True
    
    def get_drift_score(self) -> Optional[float]:
        """
        Calculate drift score comparing recent inputs to baseline.
        Returns z-score of recent mean vs baseline.
        """
        if not self.baseline_set or len(self.input_means) < 10:
            return None
        
        recent_mean = np.mean(list(self.input_means)[-50:])
        drift = abs(recent_mean - self.baseline_mean) / (self.baseline_std + 1e-7)
        return round(float(drift), 3)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": round(uptime, 1),
            "total_predictions": self.total_predictions,
            "cat_predictions": self.cat_predictions,
            "non_cat_predictions": self.total_predictions - self.cat_predictions,
            "cat_ratio": round(self.cat_predictions / max(1, self.total_predictions), 3),
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(1, self.total_predictions + self.error_count), 4)
        }
    
    def get_performance_metrics(self) -> dict:
        """Get latency and throughput metrics."""
        if not self.latencies:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "throughput_per_min": 0
            }
        
        latency_list = list(self.latencies)
        uptime_min = (datetime.now() - self.start_time).total_seconds() / 60
        
        return {
            "avg_latency_ms": round(np.mean(latency_list), 2),
            "p50_latency_ms": round(np.percentile(latency_list, 50), 2),
            "p95_latency_ms": round(np.percentile(latency_list, 95), 2),
            "p99_latency_ms": round(np.percentile(latency_list, 99), 2),
            "throughput_per_min": round(self.total_predictions / max(0.1, uptime_min), 2)
        }
    
    def get_confidence_distribution(self) -> dict:
        """Get confidence score distribution."""
        return self.confidence_buckets.copy()
    
    def get_drift_metrics(self) -> dict:
        """Get data drift metrics."""
        drift_score = self.get_drift_score()
        
        return {
            "baseline_set": self.baseline_set,
            "baseline_mean": round(self.baseline_mean, 4) if self.baseline_mean else None,
            "baseline_std": round(self.baseline_std, 4) if self.baseline_std else None,
            "current_mean": round(np.mean(list(self.input_means)[-50:]), 4) if self.input_means else None,
            "drift_score": drift_score,
            "drift_alert": drift_score is not None and drift_score > 2.0
        }
    
    def get_recent_predictions(self, n: int = 20) -> list:
        """Get N most recent predictions."""
        recent = list(self.predictions)[-n:]
        return [asdict(p) for p in reversed(recent)]
    
    def get_full_report(self) -> dict:
        """Get complete monitoring report."""
        return {
            "summary": self.get_summary(),
            "performance": self.get_performance_metrics(),
            "confidence_distribution": self.get_confidence_distribution(),
            "drift": self.get_drift_metrics(),
            "recent_predictions": self.get_recent_predictions(10)
        }


# Global monitor instance
monitor = ModelMonitor()
