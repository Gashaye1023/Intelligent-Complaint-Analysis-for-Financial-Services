# Import necessary libraries
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss #pip install faiss-cpu
import numpy as np
import warnings
# Load the cleaned dataset
df = pd.read_csv('../data/filtered_complaints.csv')# Define text chunking parameters
chunk_size = 1000  # Max tokens per chunk
chunk_overlap = 0  # Overlapping tokens between chunks

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
# Prepare to store chunks and metadata
chunks = []
metadata = []

# Chunk the narratives and store in a list
for index, row in df.iterrows():
    narrative_chunks = text_splitter.split_text(row['cleaned_narrative'])
    for chunk in narrative_chunks:
        chunks.append(chunk)
        metadata.append({
            'original_id': row['Complaint ID'],  # Assuming 'Complaint ID' is a column in your dataset
            'product': row['Product']
        })

  # Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Generate embeddings for each chunk
embeddings = model.encode(chunks, show_progress_bar=True)
# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance metric
index.add(np.array(embeddings, dtype=np.float32))
# Save the vector store and metadata
faiss.write_index(index, 'vector_store/faiss_index.index')

# Save metadata to a CSV for reference
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('vector_store/metadata.csv', index=False)