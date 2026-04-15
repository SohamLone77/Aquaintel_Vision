FROM python:3.11-slim

# Set environment variables to ensure output is not buffered and to define a known working path
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies (e.g. for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to cache the layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace
COPY . .

# Expose the standard Streamlit port
EXPOSE 8501

# Run the unified Streamlit entrypoint
CMD ["python", "-m", "streamlit", "run", "streamlit_app_optimized.py", "--server.port=8501", "--server.address=0.0.0.0"]
