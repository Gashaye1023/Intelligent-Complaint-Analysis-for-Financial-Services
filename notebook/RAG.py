import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Load the FAISS index and metadata
try:
    index = faiss.read_index('vector_store/faiss_index.index')
    metadata_df = pd.read_csv('vector_store/metadata.csv')
except FileNotFoundError as e:
    print(f"Error loading index or metadata: {e}")
    exit(1)  # Exit if files cannot be loaded

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the language model for generation
# Use a different model if gpt-3.5-turbo is not accessible
try:
    generator = pipeline('text-generation', model='gpt-3.5-turbo')  # Or use a public model like 'distilgpt2'
except Exception as e:
    print(f"Error loading model: {e}")
    generator = pipeline('text-generation', model='distilgpt2')  # Fallback to a smaller model

def retrieve_relevant_chunks(question, k=5):
    # Embed the user's question
    question_embedding = model.encode([question])
    
    # Perform similarity search
    distances, indices = index.search(np.array(question_embedding, dtype=np.float32), k)
    
    # Retrieve the corresponding chunks and metadata
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:  # Check for valid index
            retrieved_chunks.append(str(metadata_df.iloc[idx]['original_id']))  # Convert to string
    
    return retrieved_chunks

def generate_answer(question, context):
    # Define the prompt template
    prompt_template = (
        "You are a financial analyst assistant for CrediTrust. "
        "Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information. "
        "Context: {context} "
        "Question: {question} "
        "Answer:"
    )
    
    # Create context string from retrieved chunks
    context_str = "\n".join(context)
    
    # Generate the final prompt
    prompt = prompt_template.format(context=context_str, question=question)
    
    # Get the generated response from the model
    response = generator(prompt, max_length=150)
    
    return response[0]['generated_text']

# Example usage
if __name__ == "__main__":
    sample_question = "What issues do customers face with credit cards?"
    relevant_chunks = retrieve_relevant_chunks(sample_question)
    generated_answer = generate_answer(sample_question, relevant_chunks)
    
    print("Generated Answer:", generated_answer)