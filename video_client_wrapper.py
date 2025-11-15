
"""
Wrapper functions for calling the local Gradio video generation server.
Handles different approaches to ensure compatibility with LTX Video model.
"""

import requests
import json
import uuid
from typing import Optional, Tuple


class LocalVideoClient:
    """Client for calling the local LTX Video Gradio server."""

    def __init__(self, server_url: str = "http://127.0.0.1:7861"):
        self.server_url = server_url.rstrip("/")
        self.session_hash = str(uuid.uuid4())  # Generate a unique session ID

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        input_image_filepath: Optional[str] = None,
        input_video_filepath: Optional[str] = None,
        height: int = 512,
        width: int = 704,
        mode: str = "text-to-video",
        duration: float = 2.0,
        frames_to_use: int = 9,
        seed: int = 42,
        randomize_seed: bool = True,
        guidance_scale: float = 3.0,
        improve_texture: bool = False  # Disabled for CPU offload compatibility
    ) -> Tuple[str, int]:
        """
        Generate video using the local LTX Video server.

        Args:
            prompt: Text description of the desired video content
            negative_prompt: Text describing what to avoid in the generated video
            input_image_filepath: Path to input image file (for image-to-video mode)
            input_video_filepath: Path to input video file (not used, kept for compatibility)
            height: Height of the output video in pixels (must be divisible by 32)
            width: Width of the output video in pixels (must be divisible by 32)
            mode: Generation mode ("text-to-video" or "image-to-video")
            duration: Duration of the output video in seconds (0.3 to 8.5)
            frames_to_use: Number of frames to use from input video (not used, kept for compatibility)
            seed: Random seed for reproducible generation
            randomize_seed: Whether to use a random seed instead of the specified seed
            guidance_scale: CFG scale controlling prompt influence (1.0 to 10.0)
            improve_texture: Whether to use multi-scale generation (disabled by default for CPU offload)

        Returns:
            Tuple of (output_video_path, used_seed)
        """

        # Determine the correct API endpoint based on mode
        api_endpoints = {
            "text-to-video": f"{self.server_url}/api/text_to_video",
            "image-to-video": f"{self.server_url}/api/image_to_video"
        }

        api_url = api_endpoints.get(mode, api_endpoints["text-to-video"])

        # Prepare the request payload based on mode
        if mode == "text-to-video":
            payload = {
                "data": [
                    prompt,
                    negative_prompt,
                    height,
                    width,
                    duration,
                    seed,
                    randomize_seed,
                    guidance_scale,
                    improve_texture
                ]
            }
        else:  # image-to-video
            payload = {
                "data": [
                    prompt,
                    negative_prompt,
                    input_image_filepath,
                    height,
                    width,
                    duration,
                    seed,
                    randomize_seed,
                    guidance_scale,
                    improve_texture
                ]
            }

        try:
            # Make the HTTP request
            response = requests.post(
                api_url,
                json=payload,
                timeout=600  # 10 minutes timeout for video generation
            )

            if response.status_code == 200:
                result = response.json()

                # Extract the video data from the response
                if "data" in result:
                    video_data = result["data"]

                    # The response should be [video_path, seed]
                    if isinstance(video_data, list) and len(video_data) >= 2:
                        video_path = video_data[0]
                        used_seed = video_data[1]

                        # Handle case where video_path is a dict with 'name' or 'path' key
                        if isinstance(video_path, dict):
                            video_path = video_path.get("name") or video_path.get("path")

                        return video_path, used_seed
                    else:
                        raise ValueError(f"Unexpected response format: {video_data}")

                else:
                    raise ValueError(f"Unexpected response format: {result}")

            else:
                error_text = response.text
                raise RuntimeError(
                    f"Server returned error {response.status_code}: {error_text}"
                )

        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Video generation timed out after 600 seconds. "
                f"Try reducing the duration or check server status."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to video server at {self.server_url}. "
                f"Make sure video.py is running."
            )
        except Exception as e:
            raise RuntimeError(f"Error generating video: {e}")

    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        height: int = 512,
        width: int = 704,
        duration: float = 2.0,
        seed: int = 42,
        randomize_seed: bool = True,
        guidance_scale: float = 3.0,
        improve_texture: bool = True
    ) -> Tuple[str, int]:
        """
        Convenience method for text-to-video generation.

        Args:
            prompt: Text description of the desired video content
            negative_prompt: Text describing what to avoid
            height: Height of the output video in pixels
            width: Width of the output video in pixels
            duration: Duration in seconds (0.3 to 8.5)
            seed: Random seed
            randomize_seed: Whether to randomize the seed
            guidance_scale: CFG scale (1.0 to 10.0)
            improve_texture: Whether to use multi-scale generation

        Returns:
            Tuple of (output_video_path, used_seed)
        """
        return self.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode="text-to-video",
            height=height,
            width=width,
            duration=duration,
            seed=seed,
            randomize_seed=randomize_seed,
            guidance_scale=guidance_scale,
            improve_texture=improve_texture
        )

    def generate_image_to_video(
        self,
        prompt: str,
        input_image_filepath: str,
        negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
        height: int = 512,
        width: int = 704,
        duration: float = 2.0,
        seed: int = 42,
        randomize_seed: bool = True,
        guidance_scale: float = 3.0,
        improve_texture: bool = True
    ) -> Tuple[str, int]:
        """
        Convenience method for image-to-video generation.

        Args:
            prompt: Text description of the desired video content
            input_image_filepath: Path to input image file
            negative_prompt: Text describing what to avoid
            height: Height of the output video in pixels
            width: Width of the output video in pixels
            duration: Duration in seconds (0.3 to 8.5)
            seed: Random seed
            randomize_seed: Whether to randomize the seed
            guidance_scale: CFG scale (1.0 to 10.0)
            improve_texture: Whether to use multi-scale generation

        Returns:
            Tuple of (output_video_path, used_seed)
        """
        return self.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image_filepath=input_image_filepath,
            mode="image-to-video",
            height=height,
            width=width,
            duration=duration,
            seed=seed,
            randomize_seed=randomize_seed,
            guidance_scale=guidance_scale,
            improve_texture=improve_texture
        )

# Standalone function versions for easy import
_client = None


def get_video_client(server_url: str = "http://127.0.0.1:7861") -> LocalVideoClient:
    """Get or create a singleton video client instance."""
    global _client
    if _client is None:
        _client = LocalVideoClient(server_url=server_url)
    return _client


def generate_video_local(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    input_image_filepath: Optional[str] = None,
    input_video_filepath: Optional[str] = None,
    height: int = 512,
    width: int = 704,
    mode: str = "text-to-video",
    duration: float = 2.0,
    frames_to_use: int = 9,
    seed: int = 42,
    randomize_seed: bool = True,
    guidance_scale: float = 3.0,
    improve_texture: bool = True
) -> Tuple[str, int]:
    """
    Convenience function to generate video using the local server.

    Args:
        prompt: Text description of the desired video content
        negative_prompt: Text describing what to avoid
        input_image_filepath: Path to input image (for image-to-video)
        input_video_filepath: Path to input video (not used, kept for compatibility)
        height: Height in pixels (must be divisible by 32)
        width: Width in pixels (must be divisible by 32)
        mode: Generation mode ("text-to-video" or "image-to-video")
        duration: Duration in seconds (0.3 to 8.5)
        frames_to_use: Frames to use from input video (not used, kept for compatibility)
        seed: Random seed
        randomize_seed: Whether to randomize seed
        guidance_scale: CFG scale (1.0 to 10.0)
        improve_texture: Use multi-scale generation

    Returns:
        Tuple of (output_video_path, used_seed)
    """
    client = get_video_client()
    return client.generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image_filepath=input_image_filepath,
        input_video_filepath=input_video_filepath,
        height=height,
        width=width,
        mode=mode,
        duration=duration,
        frames_to_use=frames_to_use,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        improve_texture=improve_texture
    )


def generate_text_to_video(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 512,
    width: int = 704,
    duration: float = 2.0,
    seed: int = 42,
    randomize_seed: bool = True,
    guidance_scale: float = 3.0,
    improve_texture: bool = True
) -> Tuple[str, int]:
    """Convenience function for text-to-video generation."""
    client = get_video_client()
    return client.generate_text_to_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        duration=duration,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        improve_texture=improve_texture
    )


def generate_image_to_video(
    prompt: str,
    input_image_filepath: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 512,
    width: int = 704,
    duration: float = 2.0,
    seed: int = 42,
    randomize_seed: bool = True,
    guidance_scale: float = 3.0,
    improve_texture: bool = True
) -> Tuple[str, int]:
    """Convenience function for image-to-video generation."""
    client = get_video_client()
    return client.generate_image_to_video(
        prompt=prompt,
        input_image_filepath=input_image_filepath,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        duration=duration,
        seed=seed,
        randomize_seed=randomize_seed,
        guidance_scale=guidance_scale,
        improve_texture=improve_texture
    )


if __name__ == "__main__":
    # Test the client
    print("Testing local video client...")

    try:
        client = LocalVideoClient()
        print("Generating video: 'A majestic dragon flying over a medieval castle' for 2 seconds...")

        video_path, used_seed = client.generate_text_to_video(
            prompt="A majestic dragon flying over a medieval castle",
            duration=2.0,
            randomize_seed=False,
            seed=42
        )

        print(f"✓ Success!")
        print(f"  Video path: {video_path}")
        print(f"  Used seed: {used_seed}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. Your video.py server is running at http://127.0.0.1:7861")
        print("  2. The server has finished loading the LTX Video model")
        print("  3. You have enough GPU memory for video generation")
