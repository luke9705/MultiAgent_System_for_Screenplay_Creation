
"""
Wrapper functions for calling the local Gradio video generation server.
Uses Gradio Client for reliable API communication with LTX Video model.
"""

from gradio_client import Client
from typing import Optional, Tuple


class LocalVideoClient:
    """Client for calling the local LTX Video Gradio server."""

    def __init__(self, server_url: str = "http://127.0.0.1:7861"):
        self.server_url = server_url.rstrip("/")
        self.client = None

    def _get_client(self) -> Client:
        """Get or create Gradio client instance."""
        if self.client is None:
            self.client = Client(self.server_url)
        return self.client

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
        improve_texture: bool = False
    ) -> Tuple[str, int]:
        """
        Generate video from text prompt using the local LTX Video server.

        Args:
            prompt: Text description of the desired video content
            negative_prompt: Text describing what to avoid
            height: Height of the output video in pixels (must be divisible by 32)
            width: Width of the output video in pixels (must be divisible by 32)
            duration: Duration in seconds (0.3 to 8.5)
            seed: Random seed for reproducible generation
            randomize_seed: Whether to use a random seed
            guidance_scale: CFG scale (1.0 to 10.0)
            improve_texture: Whether to use multi-scale generation

        Returns:
            Tuple of (output_video_path, used_seed)
        """
        try:
            client = self._get_client()

            # Call the text_to_video API endpoint
            result = client.predict(
                prompt,
                negative_prompt,
                height,
                width,
                duration,
                seed,
                randomize_seed,
                guidance_scale,
                improve_texture,
                api_name="/text_to_video"
            )

            # Result should be a tuple of (video_path, seed)
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                video_path = result[0]
                used_seed = result[1]

                # Handle case where video_path is a dict
                if isinstance(video_path, dict):
                    video_path = video_path.get("video") or video_path.get("name") or video_path.get("path")

                return video_path, used_seed
            else:
                raise ValueError(f"Unexpected result format: {result}")

        except Exception as e:
            raise RuntimeError(f"Error generating video: {e}")

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
        improve_texture: bool = False
    ) -> Tuple[str, int]:
        """
        Generate video from image using the local LTX Video server.

        Args:
            prompt: Text description of how to animate the image
            input_image_filepath: Path to input image file
            negative_prompt: Text describing what to avoid
            height: Height of the output video in pixels (must be divisible by 32)
            width: Width of the output video in pixels (must be divisible by 32)
            duration: Duration in seconds (0.3 to 8.5)
            seed: Random seed for reproducible generation
            randomize_seed: Whether to use a random seed
            guidance_scale: CFG scale (1.0 to 10.0)
            improve_texture: Whether to use multi-scale generation

        Returns:
            Tuple of (output_video_path, used_seed)
        """
        try:
            client = self._get_client()

            # Call the image_to_video API endpoint
            result = client.predict(
                prompt,
                negative_prompt,
                input_image_filepath,
                height,
                width,
                duration,
                seed,
                randomize_seed,
                guidance_scale,
                improve_texture,
                api_name="/image_to_video"
            )

            # Result should be a tuple of (video_path, seed)
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                video_path = result[0]
                used_seed = result[1]

                # Handle case where video_path is a dict
                if isinstance(video_path, dict):
                    video_path = video_path.get("video") or video_path.get("name") or video_path.get("path")

                return video_path, used_seed
            else:
                raise ValueError(f"Unexpected result format: {result}")

        except Exception as e:
            raise RuntimeError(f"Error generating video: {e}")


# Standalone function versions for easy import
_client = None


def get_video_client(server_url: str = "http://127.0.0.1:7861") -> LocalVideoClient:
    """Get or create a singleton video client instance."""
    global _client
    if _client is None:
        _client = LocalVideoClient(server_url=server_url)
    return _client


def generate_text_to_video(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 512,
    width: int = 704,
    duration: float = 2.0,
    seed: int = 42,
    randomize_seed: bool = True,
    guidance_scale: float = 3.0,
    improve_texture: bool = False
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
    improve_texture: bool = False
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
        print("  1. Your video_app.py server is running at http://127.0.0.1:7861")
        print("  2. The server has finished loading the LTX Video model")
        print("  3. You have enough GPU memory for video generation")
