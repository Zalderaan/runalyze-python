# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system-level graphics and video processing dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Headless graphics / EGL dependencies for MediaPipe / OpenCV
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    # Video processing dependencies
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy requirements file first to optimize layer caching
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Make start script executable
RUN chmod +x start.sh

# Start the application using start.sh script
CMD ["./start.sh"]
