
from sentence_transformers import SentenceTransformer, util
import ollama

from pdf_processor import extract_pages_from_pdf, split_pages_into_chunks


# Step 1: Load the PDF and preserve page numbers
print("Loading PDF...")

pages = extract_pages_from_pdf("sample.pdf")
chunks = split_pages_into_chunks(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

if not chunks:
    print("No readable text was found in the PDF.")
    raise SystemExit


# Step 2: Load the embedding model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 3: Create embeddings for all chunks
print("\nCreating embeddings...")

chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(
    chunk_texts,
    convert_to_tensor=True
)


# Step 4: Start the assistant
print("\nPDF Assistant is ready!")
print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")


# Step 5: Keep accepting questions
while True:
    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not question:
        continue

    # Step 6: Convert the question into an embedding
    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Step 7: Find the most relevant PDF chunks
    similarity_scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # Retrieve up to three highest-scoring chunks
    k = min(3, len(chunks))
    top_results = similarity_scores.topk(k=k)

    retrieved_chunks = []
    similarity_threshold = 0.35

    print("\nTop similarity scores:")

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        chunk_index = index.item()
        score_value = score.item()

        print(f"Score: {score_value:.4f}")

        if score_value >= similarity_threshold:
            retrieved_chunks.append(chunks[chunk_index])

    # Step 8: Reject questions with weak matches
    if not retrieved_chunks:
        print(
            "\nAssistant: I couldn't find sufficiently relevant "
            "information in the PDF to answer that question."
        )
        print("\n" + "-" * 60 + "\n")
        continue

    # Step 9: Build context from the retrieved chunks
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[PDF page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Get source pages directly from the retrieved chunks
    source_pages = sorted({
        chunk["page_number"]
        for chunk in retrieved_chunks
    })

    # Step 10: Generate an answer using Llama 3.2
    print("\nAssistant: Thinking...\n")

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided PDF context. "
                    "If the answer is not available in the context, "
                    "say you could not find it in the PDF. "
                    "Do not invent facts. "
                    "Do not include source page numbers in your answer."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    # Step 11: Display the answer and source pages
    answer = response["message"]["content"]

    print("Assistant:", answer)
    print("Source pages:", ", ".join(map(str, source_pages)))

    print("\n" + "-" * 60 + "\n")
