# Dockerfile for Main App
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies
RUN pip install --no-cache-dir \
    --upgrade pip \
    huggingface_hub \
    git+https://github.com/huggingface/diffusers.git \
    kernels \
    "smolagents[transformers]" \
    transformers \
    accelerate \
    "gradio>=5.0.0" \
    gradio_client \
    ddgs \
    httpx \
    sentencepiece \
    safetensors

# Install document processing dependencies
RUN pip install --no-cache-dir \
    pdfplumber \
    docx2txt \
    odfpy \
    pandas \
    openpyxl \
    xlrd \
    pillow

# Install spacy (pinned version from requirements)
RUN pip install --no-cache-dir spacy==3.8.7

# Copy application code
COPY app.py .
COPY audio_client_wrapper.py .
COPY video_client_wrapper.py .
COPY system_prompt.txt .

# Expose port
EXPOSE 8080

# Run the main app (bind to 0.0.0.0 for Docker)
CMD ["python", "-c", "import app; app.demo.launch(server_name='0.0.0.0', server_port=8080)"]
