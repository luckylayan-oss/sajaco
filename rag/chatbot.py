"""
chatbot.py
-----------
The actual chatbot. Asks you a question,
searches your documents, and answers using Llama 3.2.

Run: python rag/chatbot.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


# ---------- CONFIG ----------
DB_FOLDER = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 4  # how many chunks to retrieve per question
# ----------------------------


# Custom prompt — tells the AI HOW to behave
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

If the answer cannot be found in the context, say: "I don't have that information in my documents."

Do not make up information. Be concise and accurate.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def build_chatbot():
    """Set up the embeddings, vector DB, and LLM."""
    print("\n🔧 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("📚 Connecting to knowledge base...")
    vectordb = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings,
    )

    print(f"🤖 Loading LLM: {LLM_MODEL}...")
    llm = Ollama(model=LLM_MODEL)

    return vectordb, llm


def answer_question(question, vectordb, llm):
    """Search docs + ask LLM."""
    # 1. Find relevant chunks from your documents
    results = vectordb.similarity_search(question, k=TOP_K)

    # 2. Combine chunks into one big context
    context = "\n\n".join([doc.page_content for doc in results])

    # 3. Build the prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 4. Ask the LLM
    response = llm.invoke(prompt)

    return response, results


def main():
    print("=" * 60)
    print("💬 SAJACO RAG — Chatbot")
    print("=" * 60)

    vectordb, llm = build_chatbot()

    print("\n✅ Ready! Ask me anything about your documents.")
    print("   Type 'exit' or 'quit' to leave.\n")

    while True:
        question = input("🧑 You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "bye"]:
            print("\n👋 Goodbye!")
            break

        try:
            answer, sources = answer_question(question, vectordb, llm)

            print(f"\n🤖 AI: {answer}\n")

            print("📎 Sources:")
            for i, doc in enumerate(sources, 1):
                source_name = doc.metadata.get("source", "unknown")
                print(f"   {i}. {source_name}")
            print()

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()