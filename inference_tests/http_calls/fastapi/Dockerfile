FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python 3 and pip
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install required libraries
RUN pip3 install fastapi uvicorn torch torchvision pillow matplotlib seaborn scikit-learn

# Set working directory
WORKDIR /app

# Copy everything (including model)
COPY app /app

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]