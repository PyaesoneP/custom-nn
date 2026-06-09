FROM python:3.11-slim

WORKDIR /app

# Create non-root user early
RUN addgroup --system app && adduser --system --no-create-home --group app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY inference.py .
COPY main.py .
COPY monitoring.py .
COPY index.html .
COPY dashboard.html .
COPY model/ ./model/

# Create logs directory and set ownership
RUN mkdir -p logs && chown -R app:app /app

# Switch to non-root user
USER app

# Cloud Run uses PORT environment variable (default 8080)
ENV PORT=8080
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:' + ('${PORT}' or '8080') + '/health')"

# Run the application - use shell form to expand $PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
