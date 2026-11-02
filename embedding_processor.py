
from sentence_transformers import SentenceTransformer

from pdf_processor import (
    extract_pages_from_pdf,
    split_pages_into_chunks
)


# Step 1: Extract text from each PDF page
print("Extracting text from PDF...")

pages = extract_pages_from_pdf("sample.pdf")

print("Total pages:", len(pages))


# Step 2: Split the pages into chunks
print("\nSplitting PDF into chunks...")

chunks = split_pages_into_chunks(pages)

print("Total chunks:", len(chunks))


# Step 3: Load the embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 4: Extract the text from each chunk
chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# Step 5: Convert every chunk into an embedding
print("\nCreating embeddings...")

embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True
)


# Step 6: Display the results
print("\nEmbedding generation complete!")

print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)

if len(embeddings) > 0:
    print("\nFirst embedding (first 10 numbers):")
    print(embeddings[0][:10])

    print("\nPage number of the first chunk:")
    print(chunks[0]["page_number"])
