"""
indexer.py
-----------
Reads documents from the 'documents/' folder (and ALL subfolders),
splits them into chunks, creates embeddings,
and saves them into a ChromaDB vector database.

Run this script once, and re-run it whenever
you add/change files in the 'documents/' folder.
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- CONFIG ----------
DOCUMENTS_FOLDER = "documents"
DB_FOLDER = "chroma_db"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks (helps context)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------------------


def load_documents(folder_path):
    """Walk through folder AND all subfolders to load supported files."""
    documents = []
    print(f"\n📂 Scanning folder (including subfolders): {folder_path}")

    # os.walk goes through every folder and subfolder
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Show relative path for cleaner output
            rel_path = os.path.relpath(file_path, folder_path)

            try:
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    print(f"  ✅ Loaded PDF: {rel_path}")

                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
                    print(f"  ✅ Loaded TXT: {rel_path}")

                elif filename.lower().endswith((".md", ".markdown")):
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents.extend(loader.load())
                    print(f"  ✅ Loaded MD:  {rel_path}")

                elif filename.lower().endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                    print(f"  ✅ Loaded DOCX: {rel_path}")

                else:
                    print(f"  ⏭️  Skipped (unsupported): {rel_path}")

            except Exception as e:
                print(f"  ❌ Error loading {rel_path}: {e}")

    return documents


def main():
    print("=" * 60)
    print("🚀 SAJACO RAG — Document Indexer")
    print("=" * 60)

    # 1. Load documents
    docs = load_documents(DOCUMENTS_FOLDER)
    if not docs:
        print("\n⚠️  No documents found. Add files to the 'documents/' folder.")
        return
    print(f"\n📄 Total documents loaded: {len(docs)}")

    # 2. Split into chunks
    print("\n✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks.")

    # 3. Create embeddings (free, runs locally)
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First time only: downloads ~90MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Save to Chroma vector database
    print(f"\n💾 Building vector database in '{DB_FOLDER}/'...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER,
    )

    print("\n" + "=" * 60)
    print("✅ DONE! Your knowledge base is ready.")
    print(f"   Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print("=" * 60)


if __name__ == "__main__":
    main()