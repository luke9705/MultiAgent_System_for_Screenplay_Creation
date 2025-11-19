"""
Wrapper functions for calling the local Gradio audio generation server.
Uses HTTP requests for API communication with MusicGen model.
"""

import requests
import json
from typing import Optional, Tuple, Union, Any
import numpy as np
import gradio as gr
import time
import uuid


class LocalAudioClient:
    """Client for calling the local MusicGen Gradio server via HTTP."""

    def __init__(self, server_url: str = "http://127.0.0.1:7860"):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        self.session_hash = str(uuid.uuid4())

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
            # Prepare the request payload with required Gradio fields
            payload = {
                "data": [prompt, duration, sample_audio],
                "fn_index": 0,
                "session_hash": self.session_hash
            }

            # Send request to the Gradio API
            response = self.session.post(
                f"{self.server_url}/api/generate_audio",
                json=payload,
                timeout=300  # 5 minute timeout for generation
            )

            if response.status_code != 200:
                raise RuntimeError(f"Server returned status {response.status_code}: {response.text}")

            result = response.json()

            # Extract data from response
            if "data" not in result:
                raise RuntimeError(f"Could not find 'data' key in response. Response received: {result}")

            data = result["data"]

            # Handle different result formats
            if isinstance(data, list) and len(data) >= 1:
                audio_result = data[0]

                # If it's a file path dictionary
                if isinstance(audio_result, dict) and "name" in audio_result:
                    return audio_result

                # If it's a tuple (sample_rate, audio_array)
                if isinstance(audio_result, (list, tuple)) and len(audio_result) == 2:
                    sample_rate = audio_result[0]
                    audio_array = np.array(audio_result[1]) if not isinstance(audio_result[1], np.ndarray) else audio_result[1]
                    return sample_rate, audio_array

            # Otherwise return as-is
            return data

        except requests.exceptions.RequestException as e:
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
