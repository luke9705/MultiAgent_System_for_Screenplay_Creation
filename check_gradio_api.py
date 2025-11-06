"""
Script to inspect the local Gradio API and understand its structure.
"""

from gradio_client import Client

def inspect_api():
    """Inspect the local Gradio server API."""
    print("Connecting to local server...")

    try:
        client = Client("http://127.0.0.2:7860")
        print("✓ Connected successfully\n")

        # View the API structure
        print("API Structure:")
        print("=" * 60)
        print(client.view_api(return_format="dict"))
        print("=" * 60)

        # Get info about endpoints
        print("\nAvailable endpoints:")
        try:
            endpoints = client.endpoints
            for endpoint in endpoints:
                print(f"  - {endpoint}")
        except Exception as e:
            print(f"Could not retrieve endpoints: {e}")

    except Exception as e:
        print(f"✗ Error: {e}")
        raise

if __name__ == "__main__":
    inspect_api()
