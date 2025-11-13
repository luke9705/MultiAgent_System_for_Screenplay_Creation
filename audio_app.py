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

# gradio interface
demo = gr.Interface(
            fn=generate_music,
            inputs=[gr.Textbox(
                        lines=2,
                        placeholder="Describe the music you'd like to hear…",
                        label="Text prompt",
                        ),
                    gr.Number(value = 10, label="Duration"),
                    gr.Audio(),
                    ],
                
            outputs=gr.Audio(type="numpy", label="Generated track (32 kHz)"),
            title="MusicGen text-to-music demo",
            description=(
                "Enter a text prompt and click **Generate** to receive an 8-second "
                "music clip created by Meta's MusicGen-medium model."
            ),
            allow_flagging="never",
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False
    )