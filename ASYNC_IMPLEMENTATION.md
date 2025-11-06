# Async Implementation for Multi-User Concurrent Access

## Overview
The application has been restructured to handle asynchronous calls, allowing multiple users to query the agent concurrently without blocking each other.

## Key Changes

### 1. Added Async Dependencies
**File: [app.py](app.py:19-21)**
```python
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
```

### 2. Enhanced Agent Class with ThreadPoolExecutor
**File: [app.py](app.py:297-298)**

The `Agent` class now includes a `ThreadPoolExecutor` with 10 worker threads to handle concurrent blocking operations:

```python
# Thread pool executor for running blocking agent.run() calls
self.executor = ThreadPoolExecutor(max_workers=10)
```

**Configuration:**
- `max_workers=10`: Up to 10 users can have their agent.run() calls executing simultaneously
- Adjust this number based on your server resources and expected concurrent load

### 3. New Async Method: `async_call()`
**File: [app.py](app.py:307-320)**

A new asynchronous wrapper method that runs the blocking `agent.run()` call in a thread pool:

```python
async def async_call(self, message: str,
                     images: Optional[list[Image.Image]] = None,
                     files: Optional[dict] = None,
                     conversation_history: Optional[dict] = None) -> str:
    """
    Async wrapper for agent.run() to handle concurrent requests from multiple users.
    Runs the blocking agent.run() call in a thread pool executor.
    """
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        self.executor,
        lambda: self.agent.run(message, images=images, additional_args={"files": files, "conversation_history": conversation_history})
    )
    return answer
```

**Why this approach:**
- `smolagents.CodeAgent.run()` is a blocking synchronous method
- By running it in a thread pool executor, we prevent blocking the async event loop
- Multiple users' requests can now execute concurrently in separate threads

### 4. Async `respond()` Function
**File: [app.py](app.py:324-351)**

The main Gradio interface function is now async:

```python
async def respond(message: str, history : dict, web_search: bool = False):
    """
    Async respond function that handles multiple concurrent user requests.
    Each user's request runs in a separate thread via the agent's thread pool executor.
    """
    global agent
    # ... (handles different input cases)
    message = await agent.async_call(text, conversation_history=history)
    # ...
    return message
```

**Benefits:**
- Gradio's native async support means multiple users won't block each other
- When combined with `.queue()`, requests are processed concurrently
- Each user gets their own thread from the pool

### 5. Async Helper Functions (Optional Enhancement)
**File: [app.py](app.py:85-107)**

Added async helper functions for HTTP operations using `httpx`:

```python
async def _download_image_async(url: str, session: httpx.AsyncClient) -> Optional[Image.Image]:
    """Helper function to download a single image asynchronously."""
    # Downloads images concurrently using httpx

async def _download_images_async(image_urls: str) -> list:
    """Async version of download_images for better performance."""
    # Downloads multiple images in parallel
```

**Note:** These are currently helper functions. The `@tool` decorated versions remain synchronous for smolagents compatibility, but you can integrate these async helpers in the future if needed.

## How It Works

### Architecture Flow

```
User 1 Request → Gradio Queue → async respond() → agent.async_call() → Thread Pool (Thread 1) → agent.run()
User 2 Request → Gradio Queue → async respond() → agent.async_call() → Thread Pool (Thread 2) → agent.run()
User 3 Request → Gradio Queue → async respond() → agent.async_call() → Thread Pool (Thread 3) → agent.run()
...
```

### Concurrency Model

1. **Gradio Queue**: Manages incoming requests (already present via `.queue()`)
2. **Async Event Loop**: Handles async/await operations without blocking
3. **Thread Pool Executor**: Runs blocking `agent.run()` calls in separate threads
4. **Result Return**: Each thread returns its result back through the async chain

### Performance Characteristics

**Before (Synchronous):**
- User 1 requests → blocks → returns (5s)
- User 2 requests → blocks → returns (5s)
- User 3 requests → blocks → returns (5s)
- **Total time: 15 seconds** (sequential)

**After (Asynchronous):**
- User 1 requests → Thread 1 (5s)
- User 2 requests → Thread 2 (5s)
- User 3 requests → Thread 3 (5s)
- **Total time: ~5 seconds** (concurrent)

## Configuration and Tuning

### Adjusting Concurrent Capacity

To change the number of concurrent users, modify the `max_workers` parameter:

```python
# In Agent.__init__()
self.executor = ThreadPoolExecutor(max_workers=20)  # Support 20 concurrent users
```

**Guidelines:**
- **Low traffic (1-5 users)**: `max_workers=5`
- **Medium traffic (5-20 users)**: `max_workers=10-15`
- **High traffic (20+ users)**: `max_workers=20-30`

**Resource considerations:**
- Each worker thread consumes memory and CPU
- Monitor your server's resources under load
- Balance between concurrency and resource usage

### Gradio Queue Configuration ⚠️ CRITICAL

**IMPORTANT:** The `.queue()` configuration in [app.py](app.py:419-422) is essential for concurrency:

```python
demo = gr.ChatInterface(
    fn=respond,  # Async function
    # ... other config ...
).queue(
    max_size=100,                     # Maximum queue size (pending requests)
    default_concurrency_limit=10      # CRITICAL: Enables 10 concurrent executions
)
```

**Without `default_concurrency_limit`, requests are processed SEQUENTIALLY even with async functions!**

This parameter must match your ThreadPoolExecutor's `max_workers`:
- ThreadPoolExecutor: `max_workers=10` (line 295)
- Queue: `default_concurrency_limit=10` (line 421)

To adjust concurrency capacity:
1. Change `max_workers` in Agent.__init__()
2. Change `default_concurrency_limit` in .queue()
3. Both values should be identical

## Testing Concurrency

### Simple Load Test

Create a test script to simulate multiple concurrent users:

```python
import asyncio
import httpx

async def test_concurrent_requests():
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for i in range(5):
            task = client.post(
                "http://127.0.0.1:7860/api/predict",
                json={"data": [{"text": f"Test query {i}", "files": []}, [], False]}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses):
            print(f"Response {i}: {resp.status_code}")

asyncio.run(test_concurrent_requests())
```

### Monitor Performance

Watch for these indicators:
1. **Response times**: Should remain relatively constant under concurrent load
2. **Thread pool usage**: Check that threads aren't all occupied (queue buildup)
3. **Memory usage**: Monitor for memory leaks with long-running sessions

## Compatibility

### Smolagents Compatibility
- The synchronous `__call__()` method is preserved for backward compatibility
- The async `async_call()` method is a new addition
- All existing `@tool` decorated functions remain synchronous (smolagents requirement)

### Gradio Compatibility
- Gradio natively supports async functions in `ChatInterface`
- The `.queue()` method works seamlessly with async functions
- No changes needed to the Gradio interface configuration

## Future Enhancements

### 1. Full Async Tool Chain
If smolagents adds native async support, you can convert all tools:

```python
@tool
async def download_images_async(image_urls: str) -> list:
    # Use the existing _download_images_async helper
    return await _download_images_async(image_urls)
```

### 2. Connection Pooling
For production deployments, add connection pooling:

```python
# Global httpx client with connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

### 3. Rate Limiting
Add per-user rate limiting to prevent abuse:

```python
from asyncio import Semaphore

user_semaphores = {}

async def rate_limited_respond(message, history, web_search, user_id):
    if user_id not in user_semaphores:
        user_semaphores[user_id] = Semaphore(2)  # 2 concurrent requests per user

    async with user_semaphores[user_id]:
        return await respond(message, history, web_search)
```

## Troubleshooting

### Issue: "RuntimeError: Event loop is closed"
**Solution:** Ensure you're not mixing sync and async calls incorrectly. Always use `await` with async functions.

### Issue: Thread pool exhaustion (requests timing out)
**Solution:** Increase `max_workers` or implement request queuing with timeouts:
```python
self.executor = ThreadPoolExecutor(max_workers=20)
```

### Issue: Memory leaks with long sessions
**Solution:** Ensure proper cleanup in the Agent class:
```python
def __del__(self):
    if hasattr(self, 'executor'):
        self.executor.shutdown(wait=False)
```

## Summary

The restructuring enables:
- ✅ **Concurrent multi-user support** via thread pool executor
- ✅ **Non-blocking async operations** with async/await pattern
- ✅ **Backward compatibility** with existing synchronous code
- ✅ **Scalable architecture** with configurable concurrency limits
- ✅ **Gradio native async support** for smooth integration

Your application is now ready to handle multiple users querying the agent simultaneously without blocking each other!
