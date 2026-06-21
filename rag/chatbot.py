"""
chatbot.py
-----------
The actual chatbot. Asks you a question,
searches your documents, and answers using Llama 3.2.

Features:
- 🧠 Memory (remembers previous questions)
- 🪄 Smart query rewriting (resolves pronouns like "their", "they")
- 🚦 Pronoun detector (skips rewriter when not needed)

Run: python rag/chatbot.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama


# ---------- CONFIG ----------
DB_FOLDER = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 6
MEMORY_TURNS = 5
# ----------------------------


# 🪄 Ultra-simple rewrite prompt — only sees the LAST question
REWRITE_PROMPT = """The user previously asked this question:
"{last_question}"

Now they asked a follow-up question that contains a pronoun (their/they/it/this/he/she). Rewrite the follow-up so the pronoun is replaced with the SUBJECT of the previous question.

Output ONLY the rewritten question. No quotes. No explanation. Just one sentence ending with "?".

Follow-up: {question}

Rewritten:"""


# 🤖 Main answering prompt
ANSWER_PROMPT = """You are a precise assistant for SAJACO precast factory. You answer questions using ONLY the provided context.

STRICT RULES:
1. Answer ONLY what the user asked about. Do NOT mix information from different roles.
2. The context contains chunks from MULTIPLE files. Each chunk is labeled with its source filename. Use ONLY chunks that match the user's question.
3. If the context truly does not contain the answer, reply: "I don't have that information in my documents."
4. Do NOT make up information.
5. Be concise and well-structured. Use bullet points or numbered lists when appropriate.
6. Use the conversation history to understand context, but base your answer on the document chunks below.

CONVERSATION HISTORY:
{history}

DOCUMENT CHUNKS:
{context}

USER'S QUESTION: {question}

YOUR ANSWER:"""


# Pronouns that signal a follow-up question needs rewriting
PRONOUNS = [
    " their ", " their?", " they ", " they?",
    " them ", " them?", " it ", " it?",
    " this role", " this person", " this job",
    " his ", " her ", " he ", " she ",
]


def needs_rewriting(question):
    """Check if a question contains pronouns that need resolving."""
    padded = " " + question.lower().strip() + " "
    return any(pronoun in padded for pronoun in PRONOUNS)


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


def format_context(results):
    """Format retrieved chunks with source labels."""
    formatted_chunks = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        source_name = source.replace("\\", "/").split("/")[-1]
        chunk_text = (
            f"--- CHUNK {i} (from file: {source_name}) ---\n"
            f"{doc.page_content}\n"
        )
        formatted_chunks.append(chunk_text)
    return "\n".join(formatted_chunks)


def format_history(chat_history):
    """Format conversation history for the answering prompt."""
    if not chat_history:
        return "(No previous conversation yet.)"

    formatted = []
    recent = chat_history[-MEMORY_TURNS:]
    for i, (q, a) in enumerate(recent, 1):
        short_a = a if len(a) < 300 else a[:300] + "..."
        formatted.append(f"Q{i}: {q}\nA{i}: {short_a}")
    return "\n\n".join(formatted)


def rewrite_question(question, chat_history, llm):
    """🪄 Rewrite follow-up questions only when needed."""
    if not chat_history:
        return question

    # Skip rewriting if no pronouns
    if not needs_rewriting(question):
        return question

    # Use ONLY the last question (no answer, no earlier turns) — zero ambiguity
    last_question = chat_history[-1][0]

    prompt = REWRITE_PROMPT.format(
        last_question=last_question,
        question=question,
    )

    try:
        rewritten = llm.invoke(prompt).strip()
    except Exception:
        return question

    # Safety checks
    if not rewritten or len(rewritten) > 200 or len(rewritten) < 5:
        return question

    rewritten = rewritten.strip('"').strip("'").strip()
    return rewritten


def answer_question(question, vectordb, llm, chat_history):
    """Search docs + ask LLM (with memory + smart query rewriting)."""
    # 1. 🪄 Rewrite the question to be standalone (only if needed)
    search_query = rewrite_question(question, chat_history, llm)
    print(f"   🔍 Searching for: {search_query}")

    # 2. Find relevant chunks using the rewritten query
    results = vectordb.similarity_search(search_query, k=TOP_K)

    # 3. Format context + history
    context = format_context(results)
    history = format_history(chat_history)

    # 4. Build the answering prompt
    prompt = ANSWER_PROMPT.format(
        history=history,
        context=context,
        question=question,
    )

    # 5. Ask the LLM
    response = llm.invoke(prompt)

    return response, results


def main():
    print("=" * 60)
    print("💬 SAJACO RAG — Chatbot (Memory 🧠 + Smart Rewriting 🪄)")
    print("=" * 60)

    vectordb, llm = build_chatbot()

    print("\n✅ Ready! Ask me anything about your documents.")
    print("   I'll remember our conversation!")
    print("   Type 'exit' or 'quit' to leave.")
    print("   Type 'clear' to wipe my memory and start fresh.\n")

    chat_history = []

    while True:
        question = input("🧑 You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit", "bye"]:
            print("\n👋 Goodbye!")
            break

        if question.lower() == "clear":
            chat_history = []
            print("\n🧹 Memory cleared! Starting fresh.\n")
            continue

        try:
            answer, sources = answer_question(
                question, vectordb, llm, chat_history
            )

            print(f"\n🤖 AI: {answer}\n")

            print("📎 Sources:")
            seen = set()
            for i, doc in enumerate(sources, 1):
                source_name = doc.metadata.get("source", "unknown")
                clean_name = source_name.replace("\\", "/").split("/")[-1]
                if clean_name not in seen:
                    print(f"   {i}. {clean_name}")
                    seen.add(clean_name)
            print()

            chat_history.append((question, answer))

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()