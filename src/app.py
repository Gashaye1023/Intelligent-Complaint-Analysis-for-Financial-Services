# app.py

import gradio as gr
from src.rag import RAGSystem
import time

# Initialize RAG system
rag_system = RAGSystem()

def respond(message, history):
    """Generate response to user message using RAG system"""
    # Get response from RAG system
    result = rag_system.query(message)
    answer = result['result']

    # Prepare sources for display
    sources = []
    for doc in result['source_documents']:
        source_info = (
            f"Product: {doc.metadata['product']}\n"
            f"Complaint ID: {doc.metadata['complaint_id']}\n"
            f"Excerpt: {doc.page_content[:200]}..."
        )
        sources.append(source_info)

    # Combine answer and sources
    full_response = f"{answer}\n\n---\n\n**Sources:**\n\n" + "\n\n".join(sources[:3])  # Show top 3 sources

    # Simulate streaming
    for i in range(0, len(full_response), 5):
        time.sleep(0.02)
        yield full_response[:i+5]

# Define Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="CrediTrust Financial Complaint Analysis",
    description="Ask questions about customer complaints across our financial products.",
    examples=[
        "What are the main complaints about credit cards?",
        "Why are customers unhappy with BNPL services?",
        "Compare complaints between personal loans and savings accounts"
    ],
    cache_examples=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear"
)

# Add footer with disclaimer
demo.footer = """
<small>Note: This system analyzes real customer complaints but does not provide financial advice.
Responses are based on patterns in complaint data and should be verified with additional sources.</small>
"""

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)