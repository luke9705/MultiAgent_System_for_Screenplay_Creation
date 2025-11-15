import gradio as gr
import torch
from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
from typing import Optional, Tuple
import numpy as np
import tempfile
import os

# Load LTX-Video models (0.9.8 distilled variant)
print("Loading LTX-Video text-to-video model...")
pipe_t2v = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe_t2v.enable_model_cpu_offload()
print("Text-to-video model loaded!")

print("Loading LTX-Video image-to-video model...")
pipe_i2v = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.bfloat16
)
pipe_i2v.enable_model_cpu_offload()
print("Image-to-video model loaded!")

def duration_to_frames(duration: float, fps: int = 24) -> int:
    """
    Convert duration in seconds to number of frames.
    LTX-Video requires frames to be divisible by 8 + 1 (e.g., 9, 17, 25, 33, 41, 49, etc.)
    """
    frames = int(duration * fps)
    # Round to nearest valid frame count (divisible by 8 + 1)
    frames = ((frames - 1) // 8) * 8 + 1
    # Ensure minimum of 9 frames
    frames = max(9, frames)
    return frames

def generate_video_fn(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    input_image: Optional[Image.Image] = None,
    input_video: Optional[str] = None,  # Not used, kept for compatibility
    height: int = 512,
    width: int = 704,
    mode: str = "text-to-video",
    duration: float = 2.0,
    frames_to_use: int = 9,  # Not used, kept for compatibility
    seed: int = 42,
    randomize_seed: bool = True,
    guidance_scale: float = 3.0,
    improve_texture: bool = False
) -> Tuple[str, int]:
    """
    Generate video using LTX-Video model.

    Args:
        prompt: Text description of the desired video content
        negative_prompt: Text describing what to avoid in the generated video
        input_image: Input image for image-to-video mode
        input_video: Not used, kept for compatibility
        height: Height of the output video (must be divisible by 32)
        width: Width of the output video (must be divisible by 32)
        mode: Generation mode ("text-to-video" or "image-to-video")
        duration: Duration in seconds (0.3 to 8.5)
        frames_to_use: Not used, kept for compatibility
        seed: Random seed for reproducible generation
        randomize_seed: Whether to use a random seed
        guidance_scale: CFG scale (1.0 to 10.0)
        improve_texture: Not used for now (CPU offload compatibility)

    Returns:
        Tuple of (output_video_path, used_seed)
    """

    # Ensure dimensions are divisible by 32
    height = (height // 32) * 32
    width = (width // 32) * 32

    # Clamp duration
    duration = max(0.3, min(8.5, duration))

    # Convert duration to frames
    num_frames = duration_to_frames(duration)

    # Handle seed
    if randomize_seed:
        seed = np.random.randint(0, 2**31 - 1)  # int32 max limit

    generator = torch.Generator(device="cpu").manual_seed(seed)

    try:
        # Select the appropriate pipeline and prepare inputs
        if mode == "image-to-video" and input_image is not None:
            # Use image-to-video pipeline
            pipe = pipe_i2v
            # Resize input image to match output dimensions
            input_image = input_image.resize((width, height))
            pipeline_kwargs = {
                "image": input_image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": "pil"
            }
        else:
            # Use text-to-video pipeline
            pipe = pipe_t2v
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": "pil"
            }

        # Generate video
        print(f"Generating {mode} with prompt: '{prompt[:50]}...'")
        print(f"Parameters: {height}x{width}, {num_frames} frames, seed={seed}")

        output = pipe(**pipeline_kwargs)

        # Extract frames
        frames = output.frames[0]  # type: ignore

        # Create temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            output_path = tmp_file.name

        # Export to video
        fps = 24
        export_to_video(frames, output_path, fps=fps)  # type: ignore

        print(f"Video generated successfully: {output_path}")
        return output_path, seed

    except Exception as e:
        print(f"Error generating video: {e}")
        raise

def text_to_video(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    duration: float,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    improve_texture: bool
) -> Tuple[str, int]:
    """Wrapper for text-to-video generation."""
    return generate_video_fn(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=None,
        input_video=None,
        height=height,
        width=width,
        mode="text-to-video",
        duration=duration,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        improve_texture=improve_texture
    )

def image_to_video(
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    height: int,
    width: int,
    duration: float,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    improve_texture: bool
) -> Tuple[str, int]:
    """Wrapper for image-to-video generation."""
    return generate_video_fn(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        input_video=None,
        height=height,
        width=width,
        mode="image-to-video",
        duration=duration,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        improve_texture=improve_texture
    )

# Create Gradio interface with tabs for different modes
with gr.Blocks(title="LTX-Video Generation") as demo:
    gr.Markdown("# LTX-Video Generation Server")
    gr.Markdown("Generate videos from text prompts or animate images using the LTX-Video model.")

    with gr.Tab("Text-to-Video"):
        with gr.Row():
            with gr.Column():
                t2v_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3
                )
                t2v_negative = gr.Textbox(
                    label="Negative Prompt",
                    value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    lines=2
                )
                with gr.Row():
                    t2v_height = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=32,
                        label="Height (must be divisible by 32)"
                    )
                    t2v_width = gr.Slider(
                        minimum=256, maximum=1280, value=704, step=32,
                        label="Width (must be divisible by 32)"
                    )
                t2v_duration = gr.Slider(
                    minimum=0.3, maximum=8.5, value=2.0, step=0.1,
                    label="Duration (seconds)"
                )
                with gr.Row():
                    t2v_seed = gr.Number(value=42, label="Seed")
                    t2v_randomize = gr.Checkbox(value=True, label="Randomize Seed")
                t2v_guidance = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                    label="Guidance Scale"
                )
                t2v_improve = gr.Checkbox(value=False, label="Improve Texture (may require more VRAM)")
                t2v_generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                t2v_output = gr.Video(label="Generated Video")
                t2v_seed_output = gr.Number(label="Used Seed")

        t2v_generate_btn.click(
            fn=text_to_video,
            inputs=[
                t2v_prompt, t2v_negative, t2v_height, t2v_width,
                t2v_duration, t2v_seed, t2v_randomize, t2v_guidance, t2v_improve
            ],
            outputs=[t2v_output, t2v_seed_output],
            api_name="text_to_video"
        )

    with gr.Tab("Image-to-Video"):
        with gr.Row():
            with gr.Column():
                i2v_image = gr.Image(label="Input Image", type="pil")
                i2v_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe how the image should be animated...",
                    lines=3
                )
                i2v_negative = gr.Textbox(
                    label="Negative Prompt",
                    value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    lines=2
                )
                with gr.Row():
                    i2v_height = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=32,
                        label="Height (must be divisible by 32)"
                    )
                    i2v_width = gr.Slider(
                        minimum=256, maximum=1280, value=704, step=32,
                        label="Width (must be divisible by 32)"
                    )
                i2v_duration = gr.Slider(
                    minimum=0.3, maximum=8.5, value=2.0, step=0.1,
                    label="Duration (seconds)"
                )
                with gr.Row():
                    i2v_seed = gr.Number(value=42, label="Seed")
                    i2v_randomize = gr.Checkbox(value=True, label="Randomize Seed")
                i2v_guidance = gr.Slider(
                    minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                    label="Guidance Scale"
                )
                i2v_improve = gr.Checkbox(value=False, label="Improve Texture (may require more VRAM)")
                i2v_generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                i2v_output = gr.Video(label="Generated Video")
                i2v_seed_output = gr.Number(label="Used Seed")

        i2v_generate_btn.click(
            fn=image_to_video,
            inputs=[
                i2v_prompt, i2v_negative, i2v_image, i2v_height, i2v_width,
                i2v_duration, i2v_seed, i2v_randomize, i2v_guidance, i2v_improve
            ],
            outputs=[i2v_output, i2v_seed_output],
            api_name="image_to_video"
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
        share=False
    )
