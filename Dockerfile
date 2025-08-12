# Use slim Python base
FROM python:3.10-slim

# avoid Debian prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps (ffmpeg, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy requirements and app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# Install python deps
# Note: we install a CPU-only torch wheel via the official index link below.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# Expose port (Render sets PORT env)
EXPOSE 7860

# Run the app
CMD ["python", "speechtotext.py"]
