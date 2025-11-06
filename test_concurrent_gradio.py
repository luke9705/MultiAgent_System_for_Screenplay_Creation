#!/usr/bin/env python3
"""
Quick test to verify Gradio is processing requests concurrently.
Run the app first (python app.py), then run this script in another terminal.
"""

import requests
import time
import threading
from typing import List, Tuple

GRADIO_URL = "http://127.0.0.1:7860"

def send_request(user_id: int) -> Tuple[int, float, bool]:
    """Send a single request to the Gradio app."""
    start_time = time.time()

    try:
        # Simple query that should take a few seconds
        payload = {
            "data": [
                {"text": f"What is 2+2? (User {user_id})", "files": []},
                [],  # history
                False  # web_search
            ]
        }

        print(f"[User {user_id}] Sending request at {time.time():.2f}")

        response = requests.post(
            f"{GRADIO_URL}/call/respond",
            json=payload,
            timeout=120
        )

        elapsed = time.time() - start_time
        success = response.status_code == 200

        print(f"[User {user_id}] Completed in {elapsed:.2f}s (status: {response.status_code})")

        return user_id, elapsed, success

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[User {user_id}] FAILED after {elapsed:.2f}s: {e}")
        return user_id, elapsed, False

def test_concurrent_requests(num_users: int = 3):
    """Test multiple concurrent requests."""
    print(f"\n{'='*60}")
    print(f"Testing {num_users} concurrent requests to Gradio")
    print(f"{'='*60}\n")

    # Check if server is running
    try:
        requests.get(GRADIO_URL, timeout=2)
    except:
        print(f"❌ ERROR: Gradio server not running at {GRADIO_URL}")
        print("   Please start the app first: python app.py")
        return

    print("✓ Gradio server is running\n")

    # Send concurrent requests using threads
    threads = []
    results = []

    overall_start = time.time()

    for i in range(1, num_users + 1):
        thread = threading.Thread(target=lambda uid=i: results.append(send_request(uid)))
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Stagger slightly

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    overall_elapsed = time.time() - overall_start

    # Analyze results
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Total time for {num_users} concurrent requests: {overall_elapsed:.2f}s")

    if results:
        successful = sum(1 for _, _, success in results if success)
        print(f"Successful requests: {successful}/{num_users}")

        if successful > 0:
            avg_time = sum(elapsed for _, elapsed, _ in results) / len(results)
            print(f"Average individual request time: {avg_time:.2f}s")

            # Determine if concurrent
            if overall_elapsed < avg_time * 0.75 * num_users:
                print(f"\n✅ SUCCESS: Requests processed CONCURRENTLY!")
                print(f"   Expected sequential time: ~{avg_time * num_users:.2f}s")
                print(f"   Actual time: {overall_elapsed:.2f}s")
                speedup = (avg_time * num_users) / overall_elapsed
                print(f"   Speedup: {speedup:.2f}x")
            else:
                print(f"\n⚠️  WARNING: Requests may be SEQUENTIAL!")
                print(f"   Expected concurrent time: ~{avg_time:.2f}s")
                print(f"   Actual time: {overall_elapsed:.2f}s")
                print(f"\n   Check Gradio queue configuration:")
                print(f"   - default_concurrency_limit should be > 1")
                print(f"   - Should match ThreadPoolExecutor max_workers")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONCURRENT REQUEST TEST FOR GRADIO")
    print("="*60)
    print("\nThis test will:")
    print("1. Verify the Gradio server is running")
    print("2. Send 3 concurrent requests")
    print("3. Measure if they execute concurrently or sequentially")
    print("\nMake sure you started the app first: python app.py")
    print("="*60)

    input("\nPress Enter to start the test...")

    try:
        test_concurrent_requests(num_users=3)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
