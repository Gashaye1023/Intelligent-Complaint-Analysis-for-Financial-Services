import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load the FAISS index and metadata
index = faiss.read_index('/content/faiss_index.index')
metadata_df = pd.read_csv('/content/metadata.csv')
# Load the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')  # Alternative model
# Initialize the language model for text generation
generator = pipeline('text-generation', model='gpt-3.5-turbo')  # Ensure you have access
def retrieve_relevant_chunks(question, k=5):
    # Embed the user's question
    question_embedding = embedding_model.encode([question])
    # Perform similarity search in the FAISS index
    distances, indices = index.search(np.array(question_embedding, dtype=np.float32), k)
    
    # Retrieve the corresponding chunks and metadata
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:  # Ensure the index is valid
            retrieved_chunks.append(metadata_df.iloc[idx]['original_id'])
    
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
