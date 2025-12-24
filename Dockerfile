FROM python:3.11-slim

WORKDIR /app

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

# Create logs directory
RUN mkdir -p logs

# Cloud Run uses PORT environment variable (default 8080)
ENV PORT=8080
EXPOSE ${PORT}

# Run the application - use shell form to expand $PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
