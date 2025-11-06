"""
Wrapper functions for calling the local Gradio audio generation server.
Handles different approaches to ensure compatibility.
"""

import requests
import json
import numpy as np
from typing import Optional, Tuple
import gradio as gr


class LocalAudioClient:
    """Client for calling the local MusicGen Gradio server."""

    def __init__(self, server_url: str = "http://127.0.0.2:7860"):
        self.server_url = server_url.rstrip("/")
        self.api_url = f"{self.server_url}/api/predict"

    def generate_audio(
        self,
        prompt: str,
        duration: int = 8,
        sample_audio: Optional[str] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio using the local MusicGen server.

        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds (max 30)
            sample_audio: Optional path to sample audio file for melody conditioning

        Returns:
            Tuple of (sample_rate, audio_data) where audio_data is a numpy array
        """

        # Prepare the request payload
        payload = {
            "data": [
                prompt,
                duration,
                sample_audio  # Can be None or a file path
            ]
        }

        try:
            # Make the HTTP request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300  # 5 minutes timeout for generation
            )

            if response.status_code == 200:
                result = response.json()

                # Extract the audio data from the response
                if "data" in result:
                    audio_data = result["data"][0]

                    # The response should be [sample_rate, audio_array]
                    if isinstance(audio_data, list) and len(audio_data) == 2:
                        sample_rate = audio_data[0]
                        audio_array = np.array(audio_data[1])
                        return sample_rate, audio_array
                    else:
                        # If it's just a file path
                        return audio_data

                else:
                    raise ValueError(f"Unexpected response format: {result}")

            else:
                error_text = response.text
                raise RuntimeError(
                    f"Server returned error {response.status_code}: {error_text}"
                )

        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Audio generation timed out after 300 seconds. "
                f"Try reducing the duration or check server status."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to audio server at {self.server_url}. "
                f"Make sure audio.py is running."
            )
        except Exception as e:
            raise RuntimeError(f"Error generating audio: {e}")

    def generate_audio_for_gradio(
        self,
        prompt: str,
        duration: int = 8,
        sample_audio: Optional[str] = None
    ) -> gr.Audio:
        """
        Generate audio and return as a Gradio Audio component.

        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds (max 30)
            sample_audio: Optional path to sample audio file

        Returns:
            Gradio Audio component with the generated audio
        """
        audio_data = self.generate_audio(prompt, duration, sample_audio)
        return gr.Audio(value=audio_data)


# Standalone function versions for easy import
_client = None

def get_audio_client() -> LocalAudioClient:
    """Get or create a singleton audio client instance."""
    global _client
    if _client is None:
        _client = LocalAudioClient()
    return _client


def generate_audio_local(
    prompt: str,
    duration: int = 8,
    sample_audio: Optional[str] = None
) -> Tuple[int, np.ndarray]:
    """
    Convenience function to generate audio using the local server.

    Args:
        prompt: Text description of the music to generate
        duration: Duration in seconds (max 30)
        sample_audio: Optional path to sample audio file

    Returns:
        Tuple of (sample_rate, audio_data)
    """
    client = get_audio_client()
    return client.generate_audio(prompt, duration, sample_audio)


def generate_audio_gradio(
    prompt: str,
    duration: int = 8,
    sample_audio: Optional[str] = None
) -> gr.Audio:
    """
    Convenience function to generate audio as a Gradio component.

    Args:
        prompt: Text description of the music to generate
        duration: Duration in seconds (max 30)
        sample_audio: Optional path to sample audio file

    Returns:
        Gradio Audio component
    """
    client = get_audio_client()
    return client.generate_audio_for_gradio(prompt, duration, sample_audio)


if __name__ == "__main__":
    # Test the client
    print("Testing local audio client...")

    try:
        client = LocalAudioClient()
        print("Generating audio: 'A warm, lo-fi piano loop' for 8 seconds...")

        result = client.generate_audio("A warm, lo-fi piano loop", 8, None)

        print(f"✓ Success!")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. Your audio.py server is running at http://127.0.0.2:7860")
        print("  2. The server has finished loading the model")
