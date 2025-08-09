# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for faiss, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 8080

# Command to run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
