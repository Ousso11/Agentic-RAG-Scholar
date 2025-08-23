# --- add near the top of main(), after MAX_ROWS ---
MODEL_OPTIONS = [
    "qwen3:14b",          # Qwen3-14B (instruct tag resolves in Ollama)
    "deepseek-r1:14b",    # DeepSeek R1 14B
    "gpt-oss:20b",        # GPT-OSS 20B
    "gemma3:14b",         # Gemma 3 14B (if your Ollama build exposes this tag)
    "granite3.3:8b"       # Granite 3.3 8B
]
DEFAULT_MODEL = "granite3.3:8b"

EMBEDDING_MODEL_NAME = "nomic-embed-text"
CHROMA_DB_PATH = "chroma_db"
VECTOR_SEARCH_K = 8
WEB_SEARCH_AGENT_K = 8
HYBRID_RETRIEVER_WEIGHTS = (0.5, 0.5)
