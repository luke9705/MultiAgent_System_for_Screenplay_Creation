"""
Test script using direct HTTP requests to call the local Gradio server.
This approach often works better than gradio_client for local servers.
"""

import requests
import json

def test_http_gradio_call():
    """Test calling the local Gradio server using direct HTTP requests."""

    url = "http://127.0.0.2:7860/api/predict"

    # Prepare the request payload
    payload = {
        "data": [
            "A warm, lo-fi piano loop",  # prompt
            8,  # duration
            None  # sample (no audio file)
        ]
    }

    print("Testing HTTP request to local Gradio server...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        response = requests.post(url, json=payload)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print(f"Result keys: {result.keys()}")
            print(f"Full result: {result}")
            return result
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"Response text: {response.text}")

    except Exception as e:
        print(f"✗ Exception: {e}")
        raise

def test_gradio_info():
    """Get info about the Gradio server."""
    url = "http://127.0.0.2:7860/info"

    print("\nFetching Gradio server info...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            info = response.json()
            print("Server info:")
            print(json.dumps(info, indent=2))
        else:
            print(f"Could not get server info: {response.status_code}")
    except Exception as e:
        print(f"Error getting info: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("HTTP-based Gradio Test")
    print("=" * 60)

    # Get server info first
    test_gradio_info()

    print("\n" + "=" * 60)

    # Test the actual call
    result = test_http_gradio_call()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
