import asyncio
import logging
import os
import sys
import tempfile
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional

import docx2txt
import gradio as gr
import httpx
import numpy as np
import pandas as pd
import pdfplumber
import requests
import torch
from diffusers import Flux2KleinPipeline
from odf.opendocument import load as load_odt
from PIL import Image
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel, VisitWebpageTool, tool
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline

from audio_client_wrapper import generate_audio_gradio
from video_client_wrapper import generate_image_to_video, generate_text_to_video

# Configure logging to save to file and show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

MAIN_LLM_MODEL_ID = os.getenv("MAIN_LLM_MODEL_ID", "openai/gpt-oss-20b")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "google/gemma-3-4b-it")
TRANSCRIBE_MODEL_ID = os.getenv("TRANSCRIBE_MODEL_ID", "openai/whisper-small")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-base-9B")
MODEL_DTYPE = torch.bfloat16
ASR_DTYPE = torch.float16
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda")

# Tee stdout/stderr to output.txt while keeping console output
class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._primary.flush()
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except:
            pass  # Don't crash if file write fails

    def flush(self):
        self._primary.flush()
        try:
            self._secondary.flush()
        except:
            pass

    def isatty(self):
        return self._primary.isatty()

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return self._primary.encoding

    @property
    def name(self):
        return getattr(self._primary, 'name', '<tee>')

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False

os.makedirs("output", exist_ok=True)
_log_file = open('output/output.txt', 'a', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)


def require_cuda() -> None:
    """Fail fast when the local-only runtime is started without CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for the local model runtime. "
            "Start this app on the target GPU host."
        )


class LocalModelRuntime:
    """Owns all local inference models used by the main app."""

    def __init__(self) -> None:
        require_cuda()
        logger.info("Loading local app models on %s", CUDA_DEVICE)

        self._image_lock = threading.Lock()
        self._vision_lock = threading.Lock()
        self._asr_lock = threading.Lock()

        self.llm = TransformersModel(
            model_id=MAIN_LLM_MODEL_ID,
            device_map="auto",
            torch_dtype="auto",
        )

        self.image_pipe = None
        self.vision_model = None
        self.vision_processor = None
        self.asr_pipe = None

        logger.info("Main LLM loaded successfully; auxiliary models will load on demand")

    def _get_image_pipe(self) -> Flux2KleinPipeline:
        if self.image_pipe is None:
            with self._image_lock:
                if self.image_pipe is None:
                    logger.info("Loading FLUX 2 Klein image pipeline")
                    pipe = Flux2KleinPipeline.from_pretrained(
                        IMAGE_MODEL_ID,
                        torch_dtype=MODEL_DTYPE,
                    )
                    pipe.to(CUDA_DEVICE)
                    self.image_pipe = pipe
        return self.image_pipe

    def _get_vision_stack(self) -> tuple[Gemma3ForConditionalGeneration, AutoProcessor]:
        if self.vision_model is None or self.vision_processor is None:
            with self._vision_lock:
                if self.vision_model is None or self.vision_processor is None:
                    logger.info("Loading Gemma vision model")
                    self.vision_model = Gemma3ForConditionalGeneration.from_pretrained(
                        VISION_MODEL_ID,
                        torch_dtype=MODEL_DTYPE,
                        device_map="auto",
                    )
                    self.vision_processor = AutoProcessor.from_pretrained(
                        VISION_MODEL_ID,
                        padding_side="left",
                    )
        return self.vision_model, self.vision_processor

    def _get_asr_pipe(self):
        if self.asr_pipe is None:
            with self._asr_lock:
                if self.asr_pipe is None:
                    logger.info("Loading Whisper transcription pipeline")
                    self.asr_pipe = pipeline(
                        task="automatic-speech-recognition",
                        model=TRANSCRIBE_MODEL_ID,
                        torch_dtype=ASR_DTYPE,
                        device=CUDA_DEVICE,
                    )
        return self.asr_pipe

    def transcribe_audio(self, audio_path: str) -> str:
        asr_pipe = self._get_asr_pipe()
        result = asr_pipe(audio_path, return_timestamps=False)
        return str(result.get("text", "")).strip()

    def generate_image(self, prompt: str, neg_prompt: str) -> Image.Image:
        image_pipe = self._get_image_pipe()
        generator = torch.Generator(device=CUDA_DEVICE).manual_seed(torch.randint(0, 2**31 - 1, ()).item())
        if neg_prompt:
            logger.info("Ignoring neg_prompt for FLUX 2 Klein local generation; embeddings are not implemented")
        output = image_pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=4.0,
            height=1024,
            width=1024,
            generator=generator,
        )
        return output.images[0].convert("RGB")

    def caption_image(self, img_path: str, prompt: str) -> str:
        vision_model, vision_processor = self._get_vision_stack()
        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You describe images precisely and concisely."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = vision_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(vision_model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = vision_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        return vision_processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


RUNTIME = LocalModelRuntime()


## utilties and class definition
def is_image_extension(filename: str) -> bool:
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    ext = os.path.splitext(filename)[1].lower() # os.path.splitext(path) returns (root, ext)
    return ext in IMAGE_EXTS

def load_file(path: str) -> dict:
    """Based on the file extension, load the file into a suitable object."""

    text = None
    ext = Path(path).suffix.lower()  # same as os.path.splitext(filename)[1].lower()

    match ext:
        case '.jpg'| '.jpeg'| '.png'| '.gif'| '.bmp'| '.tiff'| '.webp'| '.svg':
            return {"image path": path}
        case '.docx':
            text = docx2txt.process(path)
        case ".xlsx" | ".xls" :
            text = pd.read_excel(path)  # DataFrame
            text = str(text).strip()
        case '.odt':
            text = load_odt(path)
            text = str(text.body).strip()
            pass
        case ".csv":
            text = pd.read_csv(path)  # DataFrame
            text = str(text).strip()
        case ".pdf":
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        case '.py' | '.txt':
            with open(path, 'r') as f:
                text = f.read()  # plain text str
        case '.mp3' | '.wav':
            return {"audio path": path}
        case '.mp4' | '.avi' | '.mov' | '.mkv' | '.webm':
            return {"video path": path}
        case _: # default case
            text = None

    return {"raw document text": text, "file path": path}
    
def check_format(answer: str | list, *args, **kwargs) -> list:
    """Check if the answer is a list and not a nested list."""
    # other args are ignored on purpose, they are there just for compatibility
    print("Checking format of the answer:", answer)
    if isinstance(answer, list):
        for item in answer:
            if isinstance(item, list):
                print("Nested list detected")
                raise TypeError("Nested lists are not allowed in the final answer.")
        print("Final answer is a list:")
        return answer
    elif isinstance(answer, str):
        return [answer]
    elif isinstance(answer, dict):
        raise TypeError("Final answer must be a list, not a dict. Please check the answer format.")
    else:
        raise TypeError("Answer format not recognized. The answer must be either a list or a string.")


## tools definition

# Async helper functions for improved concurrency (used internally)
async def _download_image_async(url: str, session: httpx.AsyncClient) -> Optional[Image.Image]:
    """Helper function to download a single image asynchronously."""
    try:
        resp = await session.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to download from url ({url}): {e}")
        return None

async def _download_images_async(image_urls: str) -> list:
    """Async version of download_images for better performance."""
    urls = [u.strip() for u in image_urls.split(",") if u.strip()]
    async with httpx.AsyncClient() as session:
        tasks = [_download_image_async(url, session) for url in urls]
        images = await asyncio.gather(*tasks)

    wrapped = []
    for img in images:
        if img is not None:
            wrapped.append(gr.Image(value=img))
    return wrapped

@tool
def download_images(image_urls: str) -> list:
    """
    Download web images from the given comma‐separated URLs and return them in a list of PIL Images.
    Args:
        image_urls: comma‐separated list of URLs to download
    Returns:
        List of PIL.Image.Image objects wrapped by gr.Image
    """
    urls = [u.strip() for u in image_urls.split(",") if u.strip()]  # strip() removes whitespaces
    images = []
    for n_url, url in enumerate(urls, start=1):  # enumerate seems not needed... keeping it for now
        try:
            # Fetch the image bytes
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            # Load into a PIL image
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            images.append(img)

        except Exception as e:
            print(f"Failed to download from url {n_url} ({url}): {e}")

    wrapped = []
    for img in images:
        wrapped.append(gr.Image(value=img))
    return wrapped

@tool
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using a local Whisper model.
    Args:
        audio_path: path to the audio file to be transcribed.
    Returns:
        str : Transcription of the audio.
    """
    try:
        transcript = RUNTIME.transcribe_audio(audio_path)
        print(transcript)
        return transcript
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

@tool
def generate_image(prompt: str, neg_prompt: str) -> Image.Image:
    """
    Generate an image based on a text prompt using a local FLUX 2 Klein pipeline.
    Args:
        prompt: The text prompt to generate the image from.
        neg_prompt: The negative prompt to avoid certain elements in the image.
    Returns:
        Image.Image: The generated image as a PIL Image object.
    """
    image = RUNTIME.generate_image(prompt, neg_prompt)
    return gr.Image(value=image, label="Generated Image")

@tool
def generate_audio(prompt: str, duration: int) -> gr.Component:
    """
    Generate audio from a text prompt using MusicGen.
    Args:
        prompt: The text prompt to generate the audio from.
        duration: Duration of the generated audio in seconds. Max 30 seconds.
    Returns:
        gr.Component: The generated audio as a Gradio Audio component.
    """

    DURATION_LIMIT = 30
    duration = duration if duration < DURATION_LIMIT else DURATION_LIMIT

    try:
        # Use the wrapper to call local Gradio server
        return generate_audio_gradio(prompt, duration, None)
    except Exception as e:
        print(f"Error generating audio: {e}")
        raise


@tool
def generate_audio_from_sample(prompt: str, duration: int, sample_path: str = None) -> gr.Component:
    """
    Generate audio from a text prompt + audio sample using MusicGen.
    Args:
        prompt: The text prompt to generate the audio from.
        duration: Duration of the generated audio in seconds. Max 30 seconds.
        sample_path: audio sample path to guide generation.
    Returns:
        gr.Component: The generated audio as a Gradio Audio component.
    """

    DURATION_LIMIT = 30
    duration = duration if duration < DURATION_LIMIT else DURATION_LIMIT

    try:
        # Use the wrapper to call local Gradio server with sample
        return generate_audio_gradio(prompt, duration, sample_path)
    except Exception as e:
        print(f"Error generating audio with sample: {e}")
        raise

@tool
def generate_video(prompt: str, duration: float = 2.0, height: int = 384, width: int = 512) -> gr.Component:
    """
    Generate a video from a text prompt using LTX Video model.
    Args:
        prompt: The text prompt describing the desired video content.
        duration: Duration of the generated video in seconds. Range: 0.3 to 8.5 seconds. Default is 2.0.
        height: Height of the output video in pixels (must be divisible by 32). Default is 512.
        width: Width of the output video in pixels (must be divisible by 32). Default is 704.
    Returns:
        gr.Component: The generated video as a Gradio Video component.
    """
    DURATION_LIMIT = 8.5
    duration = min(duration, DURATION_LIMIT)

    # Ensure dimensions are divisible by 32
    height = (height // 32) * 32
    width = (width // 32)  * 32

    try:
        # Use the wrapper to call local Gradio server
        video_path, seed = generate_text_to_video(
            prompt=prompt,
            duration=duration,
            height=height,
            width=width,
            randomize_seed=True
        )
        return gr.Video(value=video_path, label=f"Generated Video (seed: {seed})")
    except Exception as e:
        print(f"Error generating video: {e}")
        raise

@tool
def generate_video_from_image(prompt: str, image_path: str, duration: float = 2.0, height: int = 384, width: int = 512) -> gr.Component:
    """
    Generate a video by animating an input image based on a text prompt using LTX Video model.
    Args:
        prompt: The text prompt describing how the image should be animated.
        image_path: Path to the input image file to be animated, or a PIL Image object.
        duration: Duration of the generated video in seconds. Range: 0.3 to 8.5 seconds. Default is 2.0.
        height: Height of the output video in pixels (must be divisible by 32). Default is 512.
        width: Width of the output video in pixels (must be divisible by 32). Default is 704.
    Returns:
        gr.Component: The generated video as a Gradio Video component.
    """
    DURATION_LIMIT = 8.5
    duration = min(duration, DURATION_LIMIT)

    # Ensure dimensions are divisible by 32
    height = (height // 32) * 32
    width = (width // 32) * 32

    # Handle both PIL Image objects and file paths
    if isinstance(image_path, Image.Image):
        # Save PIL Image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image_path.save(tmp_file.name)
            actual_image_path = tmp_file.name
    else:
        actual_image_path = image_path

    try:
        # Use the wrapper to call local Gradio server
        video_path, seed = generate_image_to_video(
            prompt=prompt,
            input_image_filepath=actual_image_path,
            duration=duration,
            height=height,
            width=width,
            randomize_seed=True
        )
        return gr.Video(value=video_path, label=f"Generated Video from Image (seed: {seed})")
    except Exception as e:
        print(f"Error generating video from image: {e}")
        raise

@tool
def caption_image(img_path: str, prompt: str) -> str:
    """
    Generate a caption for an image at the given path using a local Gemma 3 vision-language model.
    Args:
        img_path: The file path to the image to be captioned.
        prompt: A text prompt describing what you want the model to focus on or ask about the image.
    Returns:
        str: A description of the image.
    """
    return RUNTIME.caption_image(img_path, prompt)
    

## agent definition
class Agent:
    def __init__(self, ):
        self.agent = CodeAgent(
            model=RUNTIME.llm,
            tools=[DuckDuckGoSearchTool(max_results=5),
                   VisitWebpageTool(max_output_length=20000),
                   generate_image,
                   generate_audio_from_sample,
                   generate_audio,
                   generate_video,
                   generate_video_from_image,
                   caption_image,
                   download_images,
                   transcribe_audio],
            additional_authorized_imports=["pandas", "PIL", "io"],
            #planning_interval=5,
            max_steps=5,
            stream_outputs=False,
            final_answer_checks=[check_format]
        )
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
            self.agent.prompt_templates["system_prompt"] = system_prompt

        # Thread pool executor for running blocking agent.run() calls
        self.executor = ThreadPoolExecutor(max_workers=10)

    def __call__(self, message: str,
                 images: Optional[list[Image.Image]] = None,
                 files: Optional[dict] = None,
                 conversation_history: Optional[dict] = None) -> str:
        answer = self.agent.run(message, images = images, additional_args={"files": files, "conversation_history": conversation_history}
                                )
        return answer

    async def async_call(self, message: str,
                         images: Optional[list[Image.Image]] = None,
                         files: Optional[dict] = None,
                         conversation_history: Optional[dict] = None) -> str:
        """
        Async wrapper for agent.run() to handle concurrent requests from multiple users.
        Runs the blocking agent.run() call in a thread pool executor.
        """
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            self.executor,
            lambda: self.agent.run(message, images=images, additional_args={"files": files, "conversation_history": conversation_history})
        )
        return answer


## gradio functions
async def respond(message: str, history : dict, web_search: bool = False):
    """
    Async respond function that handles multiple concurrent user requests.
    Each user's request runs in a separate thread via the agent's thread pool executor.
    """
    global agent
    # input
    print("history:", history)
    text = message.get("text", "")
    if not message.get("files") and not web_search: # no files uploaded
        print("No files received.")
        message = await agent.async_call(text + "\nADDITIONAL CONTRAINT: Don't use web search", conversation_history=history) # conversation_history is a dict with the history of the conversation
    elif not message.get("files") and web_search: # no files uploaded
        print("No files received + web search enabled.")
        message = await agent.async_call(text, conversation_history=history)
    else:
        files = message.get("files", [])
        if not web_search:
            file = load_file(files[0])
            message = await agent.async_call(text + "\nADDITIONAL CONTRAINT: Don't use web search", files=file, conversation_history=history)
        else:
            file = load_file(files[0])
            message = await agent.async_call(text, files=file, conversation_history=history)

    # output
    print("Agent response:", message)

    return message

def initialize_agent():
    agent = Agent()
    print("Agent initialized.")
    return agent

## gradio interface
description = textwrap.dedent("""**Scriptura** is a multi-agent AI framework based on HF-SmolAgents that streamlines the creation of screenplays, storyboards, 
and soundtracks by automating the stages of analysis, summarization, and multimodal enrichment, freeing authors to focus on pure creativity.
    
To view the presentation **video**, click [here](https://www.youtube.com/watch?v=I0201ruB1Uo&ab_channel=3DLabFactory) 🤓
""")
                    
# global agent 
agent = initialize_agent()
demo = gr.ChatInterface(
                    fn=respond,
                    type='messages',
                    multimodal=True,
                    title='Scriptura: A MultiAgent System for Screenplay Creation and Editing 🎞️',
                    description=description,
                    show_progress='full',
                    fill_height=True,
                    fill_width=True,
                    save_history=True,
                    autoscroll=True,
                    additional_inputs=[
                        gr.Checkbox(value=False, label="Web Search", 
                                info="Enable web search to find information online. If disabled, the agent will only use the provided files and images.",
                                render=False),
                            ],   
                    additional_inputs_accordion=gr.Accordion(label="Tools available: ", open=True, render=False)
                        ).queue(
                            max_size=100,            # Maximum queue size (pending requests)
                            default_concurrency_limit=10  # Match ThreadPoolExecutor max_workers
                        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080
    )
