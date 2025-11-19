"""
Wrapper functions for calling the local Gradio audio generation server.
Uses Gradio Client for reliable API communication with MusicGen model.
"""

from gradio_client import Client
from typing import Optional, Tuple, Union, Any
import numpy as np
import gradio as gr


class LocalAudioClient:
    """Client for calling the local MusicGen Gradio server."""

    def __init__(self, server_url: str = "http://127.0.0.1:7860"):
        self.server_url = server_url.rstrip("/")
        self.client = None

    def _get_client(self) -> Client:
        """Get or create Gradio client instance."""
        if self.client is None:
            self.client = Client(self.server_url)
        return self.client

    def generate_audio(
        self,
        prompt: str,
        duration: int = 8,
        sample_audio: Optional[str] = None
    ) -> Union[Tuple[int, np.ndarray], dict, Any]:
        """
        Generate audio using the local MusicGen server.

        Args:
            prompt: Text description of the music to generate
            duration: Duration in seconds (max 30)
            sample_audio: Optional path to sample audio file for melody conditioning

        Returns:
            Tuple of (sample_rate, audio_data) where audio_data is a numpy array
        """
        try:
            client = self._get_client()

            # Call the audio generation endpoint with explicit API name
            result = client.predict(
                prompt,
                duration,
                sample_audio,  # Can be None or a file path
                api_name="/generate_audio"
            )

            # Handle different result formats
            if isinstance(result, dict):
                # If it's a file path dictionary, we need to handle it
                if "name" in result:
                    # This is a file reference, return it as-is for Gradio to handle
                    return result

            # If it's a tuple (sample_rate, audio_array)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                sample_rate = result[0]
                audio_array = np.array(result[1]) if not isinstance(result[1], np.ndarray) else result[1]
                return sample_rate, audio_array

            # Otherwise return as-is
            return result

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

        # Handle case where server returns a file path dictionary
        if isinstance(audio_data, dict) and "name" in audio_data:
            # If it's a file path dictionary, use the file path directly
            file_path = audio_data.get("name")
            if file_path:
                return gr.Audio(value=file_path)
            else:
                raise ValueError(f"Audio file path is missing in response: {audio_data}")

        # Otherwise, it should be a tuple (sample_rate, audio_array)
        return gr.Audio(value=audio_data)  # type: ignore


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
) -> Union[Tuple[int, np.ndarray], dict, Any]:
    """
    Convenience function to generate audio using the local server.

    Args:
        prompt: Text description of the music to generate
        duration: Duration in seconds (max 30)
        sample_audio: Optional path to sample audio file

    Returns:
        Tuple of (sample_rate, audio_data) or dict with file path
    """
    client = get_audio_client()
    return client.generate_audio(prompt, duration, sample_audio)  # type: ignore


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
        if isinstance(result, tuple):
            print(f"  Sample rate: {result[0]}")
            print(f"  Audio shape: {result[1].shape}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. Your audio_app.py server is running at http://127.0.0.1:7860")
        print("  2. The server has finished loading the MusicGen model")
