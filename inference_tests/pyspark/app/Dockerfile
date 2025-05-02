# Usa una imagen base con soporte CUDA (por ejemplo, CUDA 11.2 con Ubuntu 20.04)
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar Python y las dependencias de PyTorch y otras bibliotecas
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip

# Instalar PyTorch y otras dependencias necesarias
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    pyspark pandas matplotlib seaborn scikit-learn

# Crear directorio de trabajo
WORKDIR /app

# Copiar el contenido de tu aplicación al contenedor
COPY . .

# Comando por defecto
CMD ["sleep", "infinity"]
