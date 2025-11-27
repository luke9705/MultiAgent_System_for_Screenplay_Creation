# Dockerfile for Main App
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version (main app doesn't need GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip install --no-cache-dir \
    huggingface_hub \
    smolagents \
    "smolagents[audio]" \
    openai \
    gradio \
    gradio_client \
    duckduckgo-search \
    anthropic \
    httpx

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
