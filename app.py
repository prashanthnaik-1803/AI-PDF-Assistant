
import hashlib

import streamlit as st
import pymupdf
import ollama

from sentence_transformers import SentenceTransformer, util
from pdf_processor import split_pages_into_chunks


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Assistant")
st.write(
    "Upload a PDF and ask questions about its content. "
    "Answers are generated using a local AI model."
)


# ==========================================
# 2. SETTINGS
# ==========================================
SIMILARITY_THRESHOLD = 0.35
TOP_K = 3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2"


# ==========================================
# 3. INITIALIZE CHAT HISTORY
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==========================================
# 5. EXTRACT TEXT FROM UPLOADED PDF
# ==========================================
def extract_uploaded_pdf(uploaded_file):
    pages = []

    file_contents = uploaded_file.getvalue()

    with pymupdf.open(
        stream=file_contents,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


# ==========================================
# 6. SPLIT PDF INTO CHUNKS AND CREATE EMBEDDINGS
# ==========================================
def prepare_pdf(uploaded_file):
    pages = extract_uploaded_pdf(uploaded_file)

    chunks = split_pages_into_chunks(pages)

    if not chunks:
        return pages, [], None

    model = load_embedding_model()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        chunk_texts,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    return pages, chunks, embeddings


# ==========================================
# 7. RETRIEVE RELEVANT CHUNKS AND GENERATE ANSWER
# ==========================================
def answer_question(question, chunks, chunk_embeddings):

    model = load_embedding_model()

    # Convert the question into an embedding
    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Compare the question with PDF chunk embeddings
    similarity_scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # Select the top matching chunks
    k = min(TOP_K, len(chunks))

    top_results = similarity_scores.topk(k=k)

    retrieved_chunks = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        if score.item() >= SIMILARITY_THRESHOLD:
            chunk_index = index.item()
            retrieved_chunks.append(chunks[chunk_index])

    # Reject questions with no sufficiently relevant chunks
    if not retrieved_chunks:
        return (
            "I couldn't find sufficiently relevant information "
            "in this PDF to answer that question.",
            []
        )

    # Build context for the language model
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[PDF page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Ask the local language model to generate an answer
    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
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

    answer = response["message"]["content"]

    # Return the answer and the retrieved evidence
    return answer, retrieved_chunks


# ==========================================
# 8. SIDEBAR: UPLOAD PDF
# ==========================================
with st.sidebar:

    st.header("📄 Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    st.caption(
        "The PDF is processed locally by this application. "
        "The language model runs through Ollama on your computer."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 9. PROCESS THE UPLOADED PDF
# ==========================================
if uploaded_file is None:
    st.session_state.pdf_ready = False
    st.session_state.pop("file_signature", None)

    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()


# Identify the PDF using its actual contents
file_contents = uploaded_file.getvalue()
file_signature = hashlib.sha256(file_contents).hexdigest()


# Process the PDF only when its contents change
if st.session_state.get("file_signature") != file_signature:

    # Mark the previous document as no longer ready
    st.session_state.pdf_ready = False
    st.session_state.file_signature = file_signature
    st.session_state.messages = []

    # Remove old document data before processing the new PDF
    st.session_state.pop("pages", None)
    st.session_state.pop("chunks", None)
    st.session_state.pop("embeddings", None)

    try:
        with st.spinner(
            "Reading PDF and creating embeddings..."
        ):
            pages, chunks, embeddings = prepare_pdf(uploaded_file)

        if not chunks:
            st.error(
                "No readable text was found in this PDF. "
                "It may be scanned and require OCR."
            )
            st.stop()

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.pdf_ready = True

    except Exception as error:
        st.session_state.pdf_ready = False
        st.error(f"Could not process this PDF: {error}")
        st.stop()


if not st.session_state.get("pdf_ready", False):
    st.stop()


# ==========================================
# 10. DISPLAY PDF INFORMATION
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "PDF pages",
        len(st.session_state.pages)
    )

with col2:
    st.metric(
        "Text chunks",
        len(st.session_state.chunks)
    )

st.divider()


# ==========================================
# 11. DISPLAY CHAT HISTORY
# ==========================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display source pages and retrieved text
        retrieved_chunks = message.get("retrieved_chunks", [])

        if retrieved_chunks:

            source_pages = sorted({
                chunk["page_number"]
                for chunk in retrieved_chunks
            })

            pages_text = ", ".join(
                map(str, source_pages)
            )

            st.caption(
                f"Retrieved source pages: {pages_text}"
            )

            with st.expander("View retrieved source text"):

                for chunk in retrieved_chunks:

                    st.markdown(
                        f"**PDF page {chunk['page_number']}**"
                    )

                    st.text(chunk["text"])


# ==========================================
# 12. CHAT INPUT
# ==========================================
question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    # Display the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display the assistant's answer
    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching the PDF and generating an answer..."
            ):

                answer, retrieved_chunks = answer_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings
                )

            st.markdown(answer)

            if retrieved_chunks:

                source_pages = sorted({
                    chunk["page_number"]
                    for chunk in retrieved_chunks
                })

                pages_text = ", ".join(
                    map(str, source_pages)
                )

                st.caption(
                    f"Retrieved source pages: {pages_text}"
                )

                # Allow users to inspect the retrieved evidence
                with st.expander("View retrieved source text"):

                    for chunk in retrieved_chunks:

                        st.markdown(
                            f"**PDF page {chunk['page_number']}**"
                        )

                        st.text(chunk["text"])

            # Save the answer and retrieved chunks in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieved_chunks": retrieved_chunks
            })

        except Exception as error:

            st.error(
                "Something went wrong while generating the answer. "
                "Check that Ollama is running and the llama3.2 model "
                f"is available. Details: {error}"
            )
