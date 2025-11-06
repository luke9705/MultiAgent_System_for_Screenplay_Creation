#!/usr/bin/env python3
"""
Test script to verify async implementation for concurrent user requests.
This simulates multiple users querying the agent simultaneously.
"""

import asyncio
import time
from app import Agent

async def simulate_user_request(user_id: int, agent: Agent, delay: float = 0):
    """Simulate a single user request."""
    await asyncio.sleep(delay)  # Stagger requests slightly

    print(f"[User {user_id}] Starting request at {time.time():.2f}")
    start_time = time.time()

    try:
        # Simple test query
        response = await agent.async_call(
            message=f"What is 2+2? (User {user_id})",
            conversation_history={}
        )

        elapsed = time.time() - start_time
        print(f"[User {user_id}] Completed in {elapsed:.2f}s")
        print(f"[User {user_id}] Response: {response[:100]}...")  # First 100 chars
        return elapsed, True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[User {user_id}] FAILED after {elapsed:.2f}s: {e}")
        return elapsed, False

async def test_concurrent_users(num_users: int = 3):
    """Test multiple concurrent users."""
    print(f"\n{'='*60}")
    print(f"Testing {num_users} concurrent user requests")
    print(f"{'='*60}\n")

    # Initialize agent once (shared across requests)
    print("Initializing agent...")
    agent = Agent()
    print("Agent initialized!\n")

    # Create tasks for multiple users
    tasks = [
        simulate_user_request(i, agent, delay=i*0.1)
        for i in range(1, num_users + 1)
    ]

    overall_start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    overall_elapsed = time.time() - overall_start

    # Analyze results
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    print(f"Total time for {num_users} concurrent requests: {overall_elapsed:.2f}s")

    successful = sum(1 for _, success in results if success)
    print(f"Successful requests: {successful}/{num_users}")

    if successful > 0:
        avg_time = sum(elapsed for elapsed, _ in results) / len(results)
        print(f"Average individual request time: {avg_time:.2f}s")

        # Calculate concurrency benefit
        if overall_elapsed > 0:
            speedup = (avg_time * num_users) / overall_elapsed
            print(f"Concurrency speedup: {speedup:.2f}x")

    print(f"{'='*60}\n")

async def test_sequential_baseline(num_users: int = 3):
    """Test sequential (non-concurrent) requests for comparison."""
    print(f"\n{'='*60}")
    print(f"Baseline: {num_users} SEQUENTIAL user requests")
    print(f"{'='*60}\n")

    print("Initializing agent...")
    agent = Agent()
    print("Agent initialized!\n")

    overall_start = time.time()
    results = []

    for i in range(1, num_users + 1):
        result = await simulate_user_request(i, agent)
        results.append(result)

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print("Baseline Results:")
    print(f"{'='*60}")
    print(f"Total time for {num_users} sequential requests: {overall_elapsed:.2f}s")
    print(f"{'='*60}\n")

async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ASYNC IMPLEMENTATION TEST SUITE")
    print("="*60)

    # Test with 3 concurrent users
    await test_concurrent_users(num_users=3)

    # Optional: Test with more users
    # await test_concurrent_users(num_users=5)

    # Optional: Run baseline test for comparison
    # print("\n\nRunning baseline test for comparison...")
    # await test_sequential_baseline(num_users=3)

    print("\nAll tests completed!")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NOTE: This test requires valid API keys in environment:")
    print("  - NEBIUS_API_KEY")
    print("  - OPENAI_API_KEY (for transcription)")
    print("="*60)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
