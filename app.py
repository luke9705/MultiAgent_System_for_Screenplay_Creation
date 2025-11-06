import gradio as gr
from gradio import ChatMessage
import os
import base64
import pandas as pd
from PIL import Image
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel, InferenceClientModel,tool, Tool
from typing import Optional
import requests
from io import BytesIO
import re
from pathlib import Path
import openai
from openai import OpenAI
import pdfplumber
import numpy as np
import textwrap
import docx2txt
from odf.opendocument import load as load_odt
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

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

@tool # since they gave us OpenAI API credits, we can keep using it
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper API.
    Args:
        audio_path: path to the audio file to be transcribed.
    Returns:
        str : Transcription of the audio.
    """
    try:
        client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        with open(audio_path, "rb") as audio:  # to modify path because it is arriving from gradio
            transcript = client.audio.transcriptions.create(
                file=audio,
                model="whisper-1",
                response_format="text",
            )
        print(transcript)
        return transcript
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

@tool
def generate_image(prompt: str, neg_prompt: str) -> Image.Image:
    """
    Generate an image based on a text prompt using Flux Dev.
    Args:
        prompt: The text prompt to generate the image from.
        neg_prompt: The negative prompt to avoid certain elements in the image.
    Returns:
        Image.Image: The generated image as a PIL Image object.
    """
    client = OpenAI(base_url="https://api.studio.nebius.com/v1",
                    api_key=os.environ.get("NEBIUS_API_KEY"),
                    )

    completion = client.images.generate(
        model="black-forest-labs/flux-dev",
        prompt=prompt,
        response_format="b64_json",
        extra_body={
            "response_extension": "png",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "seed": -1,
            "negative_prompt": neg_prompt,
        }
    )
    
    image_data = base64.b64decode(completion.to_dict()['data'][0]['b64_json'])
    image = BytesIO(image_data)
    image = Image.open(image).convert("RGB") 

    return gr.Image(value=image, label="Generated Image")

@tool # not ready yet
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

    client = Tool.from_gradio(
        "luke9705/MusicGen_custom",
        name="Sound_Generator",
        description="Generate music or sound effects from a text prompt using MusicGen."
    )

    sound = client(prompt, duration)

    return gr.Audio(value=sound)


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
    
    client = Tool.from_space(
        space_id="luke9705/MusicGen_custom",
        token=os.environ.get('HF_TOKEN'),
        name="Sound_Generator",
        description="Generate music or sound effects from a text prompt using MusicGen."
    )
    
    sound = client(prompt, duration, sample_path)

    return gr.Audio(value=sound)

@tool   
def caption_image(img_path: str, prompt: str) -> str:
    """
    Generate a caption for an image at the given path using Gemma3.
    Args:
        img_path: The file path to the image to be captioned.
        prompt: A text prompt describing what you want the model to focus on or ask about the image.
    Returns:
        str: A description of the image.
    """
    client_2 = InferenceClientModel("google/gemma-3-27b-it", 
                          provider="nebius", 
                          api_key=os.getenv("NEBIUS_API_KEY"))
    
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{encoded}"
    messages = [{"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]}]
    resp = client_2(messages)
    return resp.content 
    

## agent definition
class Agent:
    def __init__(self, ):

        client = InferenceClientModel("Qwen/Qwen3-30B-A3B-Instruct-2507",
                                      provider="nebius",
                                      api_key=os.getenv("NEBIUS_API_KEY"))

        """client = OpenAIServerModel(
            model_id="claude-opus-4-20250514",
            api_base="https://api.anthropic.com/v1/",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )"""

        """client = OpenAIServerModel(
        model_id= "gpt-4.1-2025-04-14", #"gpt-5-nano-2025-08-07", #gpt-5-mini-2025-08-07"
        api_base="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        )"""
        """from smolagents import TransformersModel

        client = TransformersModel(model_id="google/gemma-3-1b-it",
                                   device_map="cuda",)"""

        self.agent = CodeAgent(
            model=client,
            tools=[DuckDuckGoSearchTool(max_results=5),
                   VisitWebpageTool(max_output_length=20000),
                   generate_image,
                   generate_audio_from_sample,
                   generate_audio,
                   caption_image,
                   download_images,
                   transcribe_audio],
            additional_authorized_imports=["pandas", "PIL", "io"],
            #planning_interval=5,
            max_steps=5,
            stream_outputs=True,
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
async def respond(message: dict, history : list, web_search: bool = False):
    """
    Async respond function - shows agent progress in real-time in the chat UI.
    Streams the actual content of each step (thoughts, code, tool outputs).
    """
    global agent
    text = message.get("text", "")

    # Prepare the agent call parameters
    file = None
    if not message.get("files") and not web_search:
        prompt = text + "\nADDITIONAL CONTRAINT: Don't use web search"
    elif not message.get("files") and web_search:
        prompt = text
    else:
        files = message.get("files", [])
        prompt = text if web_search else text + "\nADDITIONAL CONTRAINT: Don't use web search"
        file = load_file(files[0])

    # Run agent and track steps with their actual content
    loop = asyncio.get_event_loop()
    step_contents = []  # Store actual step objects/content
    last_update = [0]

    def run_agent():
        """Run agent and capture all steps"""
        result = None
        for step in agent.agent.run(prompt, images=None, additional_args={"files": file, "conversation_history": history}):
            step_contents.append(step)
            print(f"[{len(step_contents)}] {type(step).__name__}")
            result = step
        return result

    try:
        # Start agent in background
        future = loop.run_in_executor(agent.executor, run_agent)

        # Update with real-time streaming of actual step content
        while not future.done():
            await asyncio.sleep(0.2)

            if len(step_contents) > last_update[0]:
                last_update[0] = len(step_contents)

                # Build the full output showing actual step content
                output_text = "🤖 **Agent Working...**\n\n"

                for i, step in enumerate(step_contents, 1):
                    step_type = type(step).__name__

                    if "Thought" in step_type:
                        # Extract the actual thought text
                        thought_text = getattr(step, 'thought', None) or getattr(step, 'text', str(step))
                        output_text += f"💭 **Thought {i}:**\n{thought_text}\n\n"

                    elif "Code" in step_type:
                        # Extract the actual code being executed
                        code_text = getattr(step, 'code', None) or getattr(step, 'action', str(step))
                        output_text += f"⚙️ **Code {i}:**\n```python\n{code_text}\n```\n\n"

                    elif "Tool" in step_type:
                        # Extract tool call details
                        tool_name = getattr(step, 'tool_name', 'Unknown Tool')
                        tool_args = getattr(step, 'tool_args', {})
                        output_text += f"🔧 **Tool {i}:** {tool_name}\n"
                        if tool_args:
                            output_text += f"_Args:_ {tool_args}\n\n"

                    elif "Action" in step_type:
                        # Handle action steps
                        action_text = getattr(step, 'action', None) or str(step)
                        output_text += f"⚡ **Action {i}:**\n{action_text}\n\n"

                    elif "Observation" in step_type or "Output" in step_type:
                        # Handle observation/output steps
                        obs_text = getattr(step, 'observation', None) or getattr(step, 'output', str(step))
                        # Truncate long observations
                        if len(str(obs_text)) > 500:
                            obs_text = str(obs_text)[:500] + "..."
                        output_text += f"📊 **Output {i}:**\n{obs_text}\n\n"

                    else:
                        # Fallback for unknown step types
                        output_text += f"⏳ **Step {i} ({step_type}):**\n{str(step)[:300]}\n\n"

                output_text += "---\n_Processing..._"
                yield output_text

        # Get final answer
        final_answer = await future

        # Build final output with complete log
        output_text = "🤖 **Agent Complete!**\n\n"

        # Show abbreviated log
        for i, step in enumerate(step_contents, 1):
            step_type = type(step).__name__
            if "Thought" in step_type:
                output_text += f"💭 Step {i}: Thought\n"
            elif "Code" in step_type:
                output_text += f"⚙️ Step {i}: Code execution\n"
            elif "Tool" in step_type:
                tool_name = getattr(step, 'tool_name', 'Tool')
                output_text += f"🔧 Step {i}: {tool_name}\n"
            else:
                output_text += f"📝 Step {i}: {step_type}\n"

        output_text += f"\n✅ Total steps: {len(step_contents)}\n\n---\n\n"
        output_text += "📋 **Final Answer:**\n\n"

        # Append final answer
        if isinstance(final_answer, str):
            output_text += final_answer
        elif isinstance(final_answer, list):
            output_text += "\n\n".join(str(item) for item in final_answer)
        else:
            output_text += str(final_answer)

        yield output_text

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {str(e)}"

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
                    show_progress='minimal',  # Minimal progress bar since we show status in chat
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
                            max_size=100,
                            default_concurrency_limit=10
                        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
