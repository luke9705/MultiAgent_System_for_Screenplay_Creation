import gradio as gr
import torch
import numpy as np
from audiocraft.models import musicgen
from typing import Optional
from io import BytesIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = musicgen.MusicGen.get_pretrained("facebook/musicgen-small", device=DEVICE)

# generate function
def generate_music(prompt: str, duration: int, sample: Optional[tuple[int, np.ndarray]] = None):
    """Generate an 8-second music clip from a text prompt.
    Args:
        prompt: Natural-language description of the desired music.
    Returns:
        A tuple ``(sample_rate, audio)`` that Gradio's ``Audio`` component
        understands, where ``audio`` is a NumPy array shaped ``(samples,)``
        or ``(samples, channels)``.
    """
    if not prompt:
        return None  # Gradio will display empty output gracefully
    
    with torch.no_grad():
        # MusicGen expects a list of prompts; we generate one sample
        MODEL.set_generation_params(duration=duration)  # seconds per sample
        
        if sample == None:
            output = MODEL.generate([prompt])
        else:
            print("generating with sample!")
            print(type(sample[1]))
            wav = sample[1]
            if wav.ndim == 2:
                wav = wav.T     # now (C, T)
            melody = torch.from_numpy(wav.astype(np.float32))
            melody = melody.unsqueeze(0)  # now (1, C, T)
            print(melody.shape)
            output = MODEL.generate_with_chroma([prompt], melody, sample[0]) # str, np.array, sr
        
    # ``output`` is a list with a single tensor of shape (channels, samples)
    waveform = output[0].cpu().numpy()

    # Transpose to (samples, channels) if multi-channel
    if waveform.ndim == 2 and waveform.shape[0] <= 4:  # typical channel count
        waveform = waveform.T

    sample_rate = 32000  # MusicGen models are trained at 32 kHz
    return sample_rate, waveform

# gradio interface using Blocks for better API compatibility
with gr.Blocks(title="MusicGen text-to-music demo") as demo:
    gr.Markdown("# MusicGen text-to-music demo")
    gr.Markdown(
        "Enter a text prompt and click **Generate** to receive a "
        "music clip created by Meta's MusicGen-medium model."
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="Describe the music you'd like to hear…",
                label="Text prompt"
            )
            duration_input = gr.Number(value=10, label="Duration (seconds)")
            sample_input = gr.Audio(label="Sample audio (optional)")
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(type="numpy", label="Generated track (32 kHz)")

    # UI interaction
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt_input, duration_input, sample_input],
        outputs=audio_output
    )

    # Explicit API endpoint for gradio_client
    gr.api(
        fn=generate_music,
        api_name="generate_audio"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False
    )