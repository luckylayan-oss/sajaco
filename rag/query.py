"""
query.py
---------
Clean API wrapper for the SAJACO RAG system.
Use this when integrating with an API or other code.

Usage:
    from rag.query import RAGSystem
    
    rag = RAGSystem()
    result = rag.ask("What tools does the welder use?")
    print(result["answer"])
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ---------- CONFIG ----------
DB_FOLDER = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 6
MEMORY_TURNS = 5
# ----------------------------


REWRITE_PROMPT = """The user previously asked this question:
"{last_question}"

Now they asked a follow-up question that contains a pronoun (their/they/it/this/he/she). Rewrite the follow-up so the pronoun is replaced with the SUBJECT of the previous question.

Output ONLY the rewritten question. No quotes. No explanation. Just one sentence ending with "?".

Follow-up: {question}

Rewritten:"""


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


PRONOUNS = [
    " their ", " their?", " they ", " they?",
    " them ", " them?", " it ", " it?",
    " this role", " this person", " this job",
    " his ", " her ", " he ", " she ",
]


class RAGSystem:
    """
    Main RAG system class. Create one instance and reuse it.

    Example:
        rag = RAGSystem()
        result = rag.ask("What does the welder do?")
    """

    def __init__(self):
        """Load embeddings, vector DB, and LLM (one-time setup)."""
        print("🔧 Loading RAG system...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectordb = Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=self.embeddings,
        )
        self.llm = OllamaLLM(model=LLM_MODEL)
        self.chat_history = []
        print("✅ RAG system ready.")

    def _needs_rewriting(self, question):
        padded = " " + question.lower().strip() + " "
        return any(p in padded for p in PRONOUNS)

    def _format_context(self, results):
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

    def _format_history(self):
        if not self.chat_history:
            return "(No previous conversation yet.)"
        formatted = []
        recent = self.chat_history[-MEMORY_TURNS:]
        for i, (q, a) in enumerate(recent, 1):
            short_a = a if len(a) < 300 else a[:300] + "..."
            formatted.append(f"Q{i}: {q}\nA{i}: {short_a}")
        return "\n\n".join(formatted)

    def _rewrite_question(self, question):
        if not self.chat_history or not self._needs_rewriting(question):
            return question
        last_question = self.chat_history[-1][0]
        prompt = REWRITE_PROMPT.format(
            last_question=last_question, question=question
        )
        try:
            rewritten = self.llm.invoke(prompt).strip()
        except Exception:
            return question
        if not rewritten or len(rewritten) > 200 or len(rewritten) < 5:
            return question
        return rewritten.strip('"').strip("'").strip()

    def ask(self, question: str, use_memory: bool = True) -> dict:
        """
        Ask a question and get an answer.

        Args:
            question: The user's question
            use_memory: If True, uses conversation history. If False, treats as standalone.

        Returns:
            dict with keys:
                - "answer": the AI's response (string)
                - "sources": list of source filenames used
        """
        if use_memory:
            search_query = self._rewrite_question(question)
        else:
            search_query = question

        results = self.vectordb.similarity_search(search_query, k=TOP_K)

        context = self._format_context(results)
        history = self._format_history() if use_memory else "(No history.)"
        prompt = ANSWER_PROMPT.format(
            history=history, context=context, question=question
        )

        answer = self.llm.invoke(prompt)

        if use_memory:
            self.chat_history.append((question, answer))

        seen = set()
        sources = []
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            clean = source.replace("\\", "/").split("/")[-1]
            if clean not in seen:
                sources.append(clean)
                seen.add(clean)

        return {
            "answer": answer,
            "sources": sources,
        }

    def clear_memory(self):
        """Wipe conversation history."""
        self.chat_history = []


# Quick test if run directly
if __name__ == "__main__":
    rag = RAGSystem()
    print("\n--- Test Query ---")
    result = rag.ask("What tools does the welder use?")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {result['sources']}")