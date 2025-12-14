# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p Model

# Expose port
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app.py"]


# # ===== 1. Lightweight Python Image =====
# FROM python:3.10-slim

# # ===== 2. Set Working Directory =====
# WORKDIR /app

# # ===== 3. Install OS dependencies (needed by numpy/pandas/lightgbm) =====
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# # ===== 4. Copy requirements first (Docker caching) =====
# COPY requirements.txt .

# # ===== 5. Install Python dependencies =====
# RUN pip install --no-cache-dir -r requirements.txt

# # ===== 6. Copy project files =====
# COPY . .

# # ===== 7. Expose Flask port =====
# EXPOSE 5001

# # ===== 8. Run app =====
# CMD ["python", "app.py"]
