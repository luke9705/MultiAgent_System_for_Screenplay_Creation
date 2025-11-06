import gradio as gr
import time

def simple_stream_test(message, history):
    """Simple test to verify if streaming works at all"""
    # Test 1: Simple incremental text
    response = ""
    for i in range(5):
        response += f"Step {i+1}... "
        print(f"Yielding: {response}")
        yield response
        time.sleep(0.5)

    yield "✅ Streaming test complete!"

# Test with basic ChatInterface
demo = gr.ChatInterface(
    fn=simple_stream_test,
    type='messages'
).queue()

if __name__ == "__main__":
    print("=" * 50)
    print("STREAMING TEST")
    print("If you see text appear incrementally in the UI,")
    print("streaming is working. If you only see the final")
    print("message, streaming is NOT working.")
    print("=" * 50)
    demo.launch(server_name="127.0.0.1", server_port=7861)
