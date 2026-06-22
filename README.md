# Sajaco RAG

A local, free RAG (Retrieval-Augmented Generation) chatbot using Ollama + LangChain + ChromaDB.

## 🛠️ Setup

### 1. Install Ollama
Download from https://ollama.com/download and install.

Then pull the model:

    ollama pull llama3.2

### 2. Clone this repo

    git clone <repo-url>
    cd sajaco-rag

### 3. Create virtual environment

**Windows:**

    python -m venv venv
    venv\Scripts\activate

**Mac/Linux:**

    python3 -m venv venv
    source venv/bin/activate

### 4. Install dependencies

    pip install -r requirements.txt

### 5. Add documents
Drop PDFs/TXT/DOCX files into the `documents/` folder.

### 6. Build the index

    python rag/indexer.py

### 7. Run the chatbot

    python rag/chatbot.py

## 📁 Project Structure

    sajaco-rag/
    ├── rag/
    │   ├── indexer.py     # Builds the vector DB from documents
    │   └── chatbot.py     # Main chatbot interface
    ├── documents/         # Your source documents (gitignored)
    ├── chroma_db/         # Vector database (gitignored, auto-generated)
    ├── requirements.txt   # Python dependencies
    └── README.md          # This file

## 👥 Team
- Backend (RAG): [layan ]
- Frontend: [front team]
- Docs: [docs team]