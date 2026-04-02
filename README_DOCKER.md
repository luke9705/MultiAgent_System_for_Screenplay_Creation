# Docker Setup - Internal Documentation

## Architecture Overview

The application consists of 3 services that need to run together:

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   audio     │    │   video     │    │    app      │     │
│  │  (MusicGen) │    │ (LTX Video) │    │  (Gradio)   │     │
│  │  Port 7860  │    │  Port 7861  │    │  Port 8080  │     │
│  │   GPU 1     │    │   GPU 1     │    │   GPU 0     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ▲                  ▲                  │             │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                    HTTP requests                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      localhost:8080 (User access)
```

## Why This Setup?

### Problem
Originally, the apps required 3 separate conda environments:
- `conda activate audio` + `python audio_app.py`
- `conda activate video` + `python video_app.py`
- `python app.py` (base environment)

This was tedious to start manually every time.

### Solution
Docker Compose runs all 3 services with a single command, handling:
- Dependency installation (no conda needed)
- GPU allocation
- Service startup order
- Inter-service networking
- Automatic restarts

## Files Explained

### `docker-compose.yml`
Main orchestration file. Defines:
- **audio service**: Runs MusicGen on host GPU 1, exposes port 7860
- **video service**: Runs LTX Video on host GPU 1, exposes port 7861
- **app service**: Runs GPT-OSS, Gemma 3 vision, Whisper, and FLUX 2 Klein on host GPU 0, exposes port 8080

Key configurations:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["0"]
          capabilities: [gpu]
```
This enables deterministic host GPU pinning per container.

```yaml
depends_on:
  audio:
    condition: service_started
  video:
    condition: service_started
```
Ensures app starts after audio and video are running.

### `Dockerfile.audio`
- Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (has CUDA)
- Installs: audiocraft (MusicGen), gradio, ffmpeg
- Runs: `audio_app.py` bound to `0.0.0.0:7860`

### `Dockerfile.video`
- Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (has CUDA)
- Installs: diffusers, transformers, accelerate (for LTX Video)
- Runs: `video_app.py` bound to `0.0.0.0:7861`

### `Dockerfile.app`
- Base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- Installs: `smolagents[transformers]`, `transformers`, `diffusers`, `kernels`, document processing libs
- Preloads local GPT-OSS, Gemma 3 4B, Whisper, and FLUX 2 Klein at startup
- Runs: `app.py` bound to `0.0.0.0:8080`

### Client Wrapper Changes
Modified `audio_client_wrapper.py` and `video_client_wrapper.py` to read server URLs from environment variables:
- `AUDIO_SERVER_URL` (default: `http://127.0.0.1:7860`)
- `VIDEO_SERVER_URL` (default: `http://127.0.0.1:7861`)

In Docker, these are set to `http://audio:7860` and `http://video:7861` (Docker service names).

## First Time Setup (Linux + Docker Engine + NVIDIA GPUs)

### Prerequisites
1. **Docker Engine** installed
2. **NVIDIA GPU drivers** installed on the host
3. **NVIDIA Container Toolkit** installed and working
4. **Hugging Face token** with access to gated model repos (`Gemma` and `FLUX.2-klein-*`)

### Step-by-Step

```bash
# 1. Navigate to project directory
cd /path/to/MultiAgent_System_for_Screenplay_Creation

# 2. Create .env file from template
cp .env.example .env

# 3. Edit .env with your actual model configuration
$EDITOR .env
```

Your `.env` should look like:
```
HF_TOKEN=hf_xxxxxxxxxxxxx
MAIN_LLM_MODEL_ID=openai/gpt-oss-20b
VISION_MODEL_ID=google/gemma-3-4b-it
TRANSCRIBE_MODEL_ID=openai/whisper-small
IMAGE_MODEL_ID=black-forest-labs/FLUX.2-klein-base-9B
```

```bash
# 4. Build and start all services (first time takes time to download models)
docker compose up --build

# Or run in background (detached mode)
docker compose up -d --build
```

### First Run Notes
- **Model downloads**: First startup downloads large local model weights for GPT-OSS, Gemma, Whisper, FLUX, MusicGen, and LTX Video
- **GPU split**: App uses host GPU 0. Audio and video share host GPU 1
- **Patience**: App startup is now heavier because it preloads the local app-side models
- **Check logs**: `docker compose logs -f` to see progress
- **Healthchecks**: Services have healthchecks with long start periods to allow model loading

## Common Commands

```bash
# Start all services
docker compose up

# Start in background
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f

# View logs for specific service
docker compose logs -f audio

# Rebuild after code changes
docker compose up --build

# Full cleanup (removes volumes with cached models)
docker compose down -v

# Check service status
docker compose ps
```

## Troubleshooting

### GPU not detected
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```
If this fails, fix the NVIDIA Container Toolkit / Docker GPU integration first.

### Out of GPU memory
If you get OOM errors:
- Confirm GPU 0 has enough VRAM for GPT-OSS + Gemma + Whisper + FLUX
- Reduce video resolution in `video_app.py`
- Generate shorter audio/video clips
- Or revise the app-side model preload strategy

### Service won't start
```bash
# Check specific service logs
docker compose logs audio
docker compose logs video
docker compose logs app
```

### Models re-downloading
Models are cached in a Docker volume `huggingface_cache`. If you run `docker compose down -v`, it deletes this cache. Use `docker compose down` (without `-v`) to preserve cache.

## Running Locally (Without Docker)

If you need to run without Docker (e.g., for debugging on a GPU host):

```bash
# Terminal 1
conda activate audio
python audio_app.py

# Terminal 2
conda activate video
python video_app.py

# Terminal 3
export HF_TOKEN=hf_xxxxxxxxxxxxx
python app.py
```

The client wrappers default to `localhost` URLs when environment variables aren't set.
