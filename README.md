---
title: MultiAgent_for_MCP2526
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---

# Scriptura: A MultiAgent System for Screenplay Creation and Editing

The explanation video is available [here](https://www.youtube.com/watch?v=I0201ruB1Uo)

The screenplay used in the video as sample is available [here](https://www.studiobinder.com/blog/best-free-movie-scripts-online/)

## Introduction

**Scriptura** is a multi-agent AI framework based on HF-SmolAgents that streamlines the creation of screenplays, storyboards, and soundtracks by automating the stages of analysis, summarization, and multimodal enrichment—freeing authors to focus on pure creativity.

At its heart:

* GPT-OSS 20B serves as the primary orchestrating agent, coordinating workflows and managing high-level reasoning across the system through a fully local `transformers` runtime.
* Gemma-3-4B-IT acts as the local multimodal assistant for image-aware tasks such as captioning and visual analysis.

For media generation, Scriptura integrates:

* MusicGen models (per the AudioCraft MusicGen specification), deployed as a local service, enabling the agent to produce original soundtracks and sound effects from text prompts or combined text + audio samples.
* FLUX 2 Klein for on-the-fly local image creation, ideal for storyboards, concept art, and visual references that seamlessly tie into the narrative flow.

Optionally, Scriptura can query external sources (e.g., via a DuckDuckGo API integration) to pull in reference scripts, sound samples, or research materials, ensuring that every draft is not only creatively rich but also contextually informed.

---

## Agent Capabilities

Scriptura provides a rich set of agents and tools to cover the full screenplay production and enrichment pipeline:

- **Text Analysis & Summarization**  
  - Automatically extracts key themes, character arcs, and plot points  
  - Segments and summarizes scenes for rapid iteration  

- **Multimodal Ingestion**  
  - Supports PDF, DOCX, ODT, TXT and image uploads  
  - Transcribes audio files using a local Whisper model  

- **Image Generation**  
  - On-the-fly storyboard and concept art creation via local FLUX 2 Klein  

- **Audio Generation**  
  - Produces original soundtracks and SFX with MusicGen (AudioCraft spec)  
  - Allows sample-conditioned audio generation  

- **Captioning & Metadata**  
  - Auto-generates captions and descriptions for images using local Gemma-3-4B-IT  

- **Optional Web Research**  
  - Queries DuckDuckGo to fetch example scripts, sound samples, or contextual references  

---

## Agent Flow

Here’s an example flow demonstrating how you could use the agent.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/683eca9c72e8702dc425b51f/FFhfD2gCL-BjRC1eT-ELB.png)

---

## Code Overview

```bash
.
├── app.py               # Entry point: defines Gradio interface and routing logic
├── system_prompt.txt    # System-level prompt template for the CodeAgent
├── requirements.txt     # Python dependencies (Gradio, SmolAgents, transformers, diffusers, etc.)
└── README.md            # Project documentation
```

* **app.py**

  * **Agent** class: loads GPT-OSS 20B plus local image/caption/transcription runtimes, then registers all tools
  * **respond()**: orchestrates between Gradio inputs and CodeAgent
  * Decorated `@tool` functions for image download, media generation, transcription, captioning
  * Gradio `ChatInterface` setup with text/file support and “Enable web search” toggle

* **system\_prompt.txt**

  * Injects the agent’s “way of thinking,” including reasoning structure and error handling

* **requirements.txt**

  * Lists all required libraries (Gradio, SmolAgents, transformers, diffusers, Hugging Face, PDFPlumber, etc.)

---

## Deployment & Access

### Local GPU Deployment

Scriptura now targets a local Linux Docker deployment with NVIDIA GPUs:

1. Set `HF_TOKEN` and the desired model IDs in `.env`  
2. Use Docker Compose to launch the app, audio, and video services together  
3. Keep host GPU 0 for the main app and host GPU 1 for audio/video services  
4. Follow the full setup instructions in `README_DOCKER.md`

> **Note:** The current runtime is GPU-only and is not intended for Hugging Face Spaces or CPU-only hosts.

---
## Use Cases

**Independent Writer**  
* Upload a screenplay and quickly get a summary, a list of characters, and locations.  
* Create visual storyboards of key narrative moments via FLUX (PNG/JPEG outputs).  
* Generate brief soundtracks or sound effects to accompany script presentations (MP3/WAV).

**Film Production Company**  
* Import multiple screenplays (PDF, DOCX) and automatically receive reports on characters, locations, and potential copyright issues.  
* Use the web search feature to find reference scripts or specific sound effects from free/paid sources.  
* Develop visual storyboards and audio prototypes to share with directors, artists, and investors.

**Translation and Adaptation Agency**  
* Upload foreign-language scripts and obtain a structured text version with extracted entities (JSON/CSV).  
* Generate contextual images for cultural adaptation (e.g., images matching the original setting via FLUX).  
* Produce reference audio via MusicGen to test culturally appropriate music for the target audience.

**Digital Humanities Course**  
* Demonstrate how to build a text-mining tool applied to performing arts, combining NLP, image, and audio pipelines.  
* Allow students to analyze real scripts, generate abstracts, scene maps, and visual/audio prototypes in a hands-on environment.  
* Explore local transformer models, OCR, speech-to-text, and AI-driven media generation as part of the curriculum.

---

## Contributors:

* Code development and implementation made by **luke9705**;
* Ideas creation, testing and videomaking conducted by **OrianIce**;
* Research and testing by **Loren1214**;
* Code revisions by **DDPM**.

---
## Sources
The following libraries, models, and tools power Scriptura’s agents and multimodal capabilities:

- **GPT-OSS 20B** – primary orchestrating LLM for high-level reasoning and workflow management  
- **Gradio** – interactive web UI framework  
- **smolagents** – lightweight multi-agent orchestrator from Hugging Face  
- **huggingface_hub** – model & dataset management  
- **duckduckgo-search** – optional web research integration  
- **transformers** – local LLM, multimodal, and Whisper inference  
- **diffusers** – local FLUX 2 Klein image generation  
- **pdfplumber** – PDF text extraction  
- **docx2txt** – DOCX parsing  
- **odfpy** – ODT parsing  
- **pandas** – data handling  
- **Pillow (PIL)** – image processing  
- **requests** – HTTP client for external APIs  
- **numpy** – numerical operations  
- **MusicGen (AudioCraft)** – soundtrack and SFX generation  
- **FLUX 2 Klein** – on-the-fly image generation  
- **Gemma-3-4B-IT** – multimodal captioning and metadata  
