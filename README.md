# 📚 AI PDF Assistant

An AI-powered PDF question-answering application that allows users to upload PDF documents and ask questions about their content using a local Large Language Model (LLM).

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded document and generate answers based on the extracted text.

## ✨ Features

- 📄 Upload PDF documents through a web interface.
- 📝 Extract text from PDF pages.
- ✂️ Split extracted text into smaller chunks.
- 🧠 Generate text embeddings using Sentence Transformers.
- 🔍 Retrieve relevant chunks using semantic similarity.
- 🤖 Generate answers using the Llama 3.2 language model.
- 📌 Display source page numbers for retrieved text.
- 🔎 Inspect the retrieved source text used to generate answers.
- 🛡️ Reject questions when no sufficiently relevant text is found.
- 💻 Run the language model locally using Ollama.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyMuPDF | Extract text from PDF documents |
| Sentence Transformers | Generate text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Ollama | Run the language model locally |
| Llama 3.2 | Generate answers from retrieved context |
| Streamlit | Build the web interface |

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow:

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** PyMuPDF extracts text from each page.
3. **Text Chunking:** The extracted text is divided into smaller chunks while preserving page numbers.
4. **Embedding Generation:** Sentence Transformers converts each chunk into a numerical vector.
5. **Semantic Search:** The user's question is converted into an embedding and compared with the chunk embeddings using cosine similarity.
6. **Context Retrieval:** The most relevant chunks are selected using a similarity threshold.
7. **Answer Generation:** Llama 3.2 generates an answer using the retrieved text.
8. **Source Inspection:** The application displays source page numbers and allows users to inspect the retrieved text.

## 🏗️ Architecture

```text
             PDF Upload
                 |
                 v
          Text Extraction
                 |
                 v
            Text Chunking
                 |
                 v
          Embedding Generation
                 |
                 v
          Store Chunk Embeddings
                 |
                 v
User Question --> Question Embedding
                 |
                 v
           Semantic Search
                 |
                 v
          Retrieve Relevant Text
                 |
                 v
          Llama 3.2 via Ollama
                 |
                 v
           Generated Answer
                 |
                 v
      Answer + Source Page Numbers
                 |
                 v
        Retrieved Text Inspection
```

## 📋 Requirements

- Python 3.11 or newer, with a compatible package environment
- Ollama installed on your computer
- The Llama 3.2 model downloaded through Ollama
- An internet connection for the initial installation and embedding-model download

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-PDF-Assistant
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install pymupdf sentence-transformers ollama streamlit
```

### 4. Install and prepare Ollama

Install Ollama from its official website:

https://ollama.com/

Download the model:

```bash
ollama pull llama3.2
```

Ensure Ollama is running before using the application.

### 5. Run the application

From the project directory, run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

### 6. Upload a PDF and ask questions

1. Upload a PDF using the sidebar.
2. Wait for the document to be processed.
3. Enter a question about the document.
4. Read the generated answer.
5. Check the retrieved source pages and expand **View retrieved source text** to inspect the supporting text.

## 📁 Project Structure

```text
AI-PDF-Assistant/
│
├── app.py
├── pdf_processor.py
├── embedding_processor.py
├── retrieval_processor.py
├── README.md
└── sample.pdf
```

## ⚠️ Limitations

- The application works best with PDFs containing selectable text.
- Scanned PDFs may require Optical Character Recognition (OCR).
- Answers depend on the relevance and quality of the retrieved text.
- Similarity scores are heuristic relevance measures, not probabilities that an answer is correct.
- The language model may produce incorrect answers even when relevant text is retrieved.
- Processing large documents may require additional memory and time.
- The application requires a compatible local environment and an available Ollama model.

## 🔮 Future Improvements

- Add OCR support for scanned PDFs.
- Improve retrieval quality and chunking strategies.
- Add streaming responses.
- Support multiple uploaded documents.
- Add automated evaluation for retrieval and answer quality.
- Improve error handling and document-processing performance.

## 👨‍💻 Project Purpose

This project demonstrates practical applications of Python, natural language processing, semantic search, vector embeddings, local language models, and Retrieval-Augmented Generation (RAG).

It was developed as a learning project to explore how AI systems can answer questions using information from user-provided documents.
