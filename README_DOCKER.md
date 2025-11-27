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
│  │    GPU      │    │    GPU      │    │    CPU      │     │
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
- **audio service**: Runs MusicGen on GPU, exposes port 7860
- **video service**: Runs LTX Video on GPU, exposes port 7861
- **app service**: Main Gradio interface, connects to audio/video via Docker network

Key configurations:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```
This enables NVIDIA GPU passthrough to containers.

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
- Base: `python:3.11-slim` (no GPU needed)
- Installs: smolagents, gradio, openai, document processing libs
- Uses CPU-only PyTorch (lighter)
- Runs: `app.py` bound to `0.0.0.0:8080`

### Client Wrapper Changes
Modified `audio_client_wrapper.py` and `video_client_wrapper.py` to read server URLs from environment variables:
- `AUDIO_SERVER_URL` (default: `http://127.0.0.1:7860`)
- `VIDEO_SERVER_URL` (default: `http://127.0.0.1:7861`)

In Docker, these are set to `http://audio:7860` and `http://video:7861` (Docker service names).

## First Time Setup (Windows 11 + NVIDIA GPU)

### Prerequisites
1. **Docker Desktop** installed with WSL2 backend
2. **NVIDIA GPU drivers** installed on Windows
3. **Enable GPU in Docker Desktop**:
   - Open Docker Desktop → Settings → Resources → WSL Integration
   - Also check: Settings → Docker Engine, ensure no GPU-blocking configs

### Step-by-Step

```powershell
# 1. Navigate to project directory
cd C:\path\to\MultiAgent_System_for_Screenplay_Creation

# 2. Create .env file from template
copy .env.example .env

# 3. Edit .env with your actual API keys
notepad .env
```

Your `.env` should look like:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
NEBIUS_API_KEY=xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

```powershell
# 4. Build and start all services (first time takes 10-20 min to download models)
docker compose up --build

# Or run in background (detached mode)
docker compose up -d --build
```

### First Run Notes
- **Model downloads**: First startup downloads ~10GB+ of models (MusicGen, LTX Video)
- **Patience**: Audio service takes ~2 min to load, video takes ~3-5 min
- **Check logs**: `docker compose logs -f` to see progress
- **Healthchecks**: Services have healthchecks with long start periods to allow model loading

## Common Commands

```powershell
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
```powershell
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```
If this fails, check Docker Desktop GPU settings.

### Out of GPU memory
Both audio and video services share the GPU. If you get OOM errors:
- Reduce video resolution in `video_app.py`
- Generate shorter audio/video clips
- Or run services sequentially instead of parallel

### Service won't start
```powershell
# Check specific service logs
docker compose logs audio
docker compose logs video
docker compose logs app
```

### Models re-downloading
Models are cached in a Docker volume `huggingface_cache`. If you run `docker compose down -v`, it deletes this cache. Use `docker compose down` (without `-v`) to preserve cache.

## Running Locally (Without Docker)

If you need to run without Docker (e.g., for debugging):

```bash
# Terminal 1
conda activate audio
python audio_app.py

# Terminal 2
conda activate video
python video_app.py

# Terminal 3 (base environment)
python app.py
```

The client wrappers default to `localhost` URLs when environment variables aren't set.
