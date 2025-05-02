# Usar una imagen base de NVIDIA con CUDA y soporte para PyTorch
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Establecer la zona horaria y evitar interacciones
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    openjdk-17-jdk wget curl tzdata \
    && rm -rf /var/lib/apt/lists/*

# Configurar la zona horaria (por ejemplo, América/New_York)
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Establecer JAVA_HOME (opcional si usas Spark o alguna otra herramienta que lo requiera)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Instalar PyTorch y otras dependencias desde el repositorio oficial
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    dask \
    distributed \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Crear directorio de trabajo
WORKDIR /app

# Copiar el contenido de la carpeta actual al directorio de trabajo en el contenedor
COPY . .

# Comando por defecto (modificar según el caso)
CMD ["sleep", "infinity"]
