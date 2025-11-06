"""
Test script to verify local audio generation integration.
Make sure your audio.py server is running at http://127.0.0.2:7860 before running this test.
"""

from gradio_client import Client
import os

def test_local_audio_generation():
    """Test basic audio generation without sample."""
    print("Testing local audio generation...")
    print("Connecting to local server at http://127.0.0.2:7860")

    try:
        client = Client("http://127.0.0.2:7860")
        print("✓ Connected to local server")

        # Test basic generation
        print("\nGenerating audio: 'A warm, lo-fi piano loop' for 8 seconds...")
        result = client.predict(
            "A warm, lo-fi piano loop",  # prompt
            8,  # duration
            None,  # no sample
            api_name="/predict"
        )

        print(f"✓ Audio generated successfully!")
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. Your audio.py server is running")
        print("  2. It's accessible at http://127.0.0.2:7860")
        print("  3. The server has finished loading the model")
        raise

def test_audio_with_sample():
    """Test audio generation with a sample file."""
    print("\n" + "="*50)
    print("Testing audio generation with sample...")

    # You'll need to provide a sample audio file path
    sample_path = "/path/to/your/sample.wav"  # Update this path

    if not os.path.exists(sample_path):
        print(f"⚠ Sample file not found at {sample_path}")
        print("  Skipping sample-based generation test")
        return

    try:
        client = Client("http://127.0.0.2:7860")

        print(f"\nGenerating audio with sample: {sample_path}")
        result = client.predict(
            "A warm, lo-fi piano loop",  # prompt
            8,  # duration
            sample_path,  # sample audio path
            api_name="/predict"
        )

        print(f"✓ Audio with sample generated successfully!")
        print(f"  Result: {result}")

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("Audio Integration Test")
    print("=" * 50)

    # Test 1: Basic generation
    result = test_local_audio_generation()

    # Test 2: Generation with sample (optional)
    # Uncomment if you have a sample file:
    # test_audio_with_sample()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
