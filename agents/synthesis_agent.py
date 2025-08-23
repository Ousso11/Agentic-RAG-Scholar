# ============================================================
# Derived Class 1: DocSynthesisAgent (retriever tool + Carnivore MD prompts)
# ============================================================

# -- Retriever → tool (for DocSynthesisAgent)
from langchain.schema import Document
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from agents.base_reflex_agent import BaseReflexAgent

def make_retriever_tool(retriever, name: str = "corpus_search", k: int = 5):
    """Wrap a LangChain retriever as a simple tool returning compact context."""
    @tool(name, return_direct=False)
    def _search(query: str) -> str:
        """Search the local/vector corpus for the query and return up to k compact snippets with their sources."""
        docs: List[Document] = retriever.invoke(query)
        parts = []
        for d in docs[:k]:
            src = d.metadata.get("source") or d.metadata.get("url") or ""
            parts.append(f"[{src}] {d.page_content}".strip())
        print(f"Retrieved {len(parts)} snippets for query: {query}")
        return "\n\n".join(parts).strip() or "(no results)"
    return _search

class DocSynthesisAgent(BaseReflexAgent):
    """
    Uses your local/vector retriever as a tool; prompts follow the Dr. Saladino (Carnivore MD) format.
    """
    def __init__(self, llm: BaseChatModel, retriever, **kwargs):
        self._retriever_tool = make_retriever_tool(retriever)
        super().__init__(llm=llm, **kwargs)

    def _build_tools(self) -> List:
        return [self._retriever_tool]

    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        responder = ChatPromptTemplate.from_messages([
            ("system",
            """You are a rigorous research assistant. 
    You MUST return your output ONLY by calling the tool AnswerTool, nothing else. 
    Follow the schema strictly:

    Guidelines:
    1. {first_instruction}
    2. Use corpus_search to retrieve relevant passages and ground claims.
    3. Synthesize cautiously: identify study type, methods, scope, limitations.
    4. If evidence is missing/uncertain, state so under 'reflection.missing'.
    5. Suggest 1–3 web queries under 'search_queries'.
    """),
            MessagesPlaceholder("history"),
        ])

        reviser = ChatPromptTemplate.from_messages([
            ("system",
            """You are a reviser. 
    You MUST return your output ONLY by calling the tool ReviseTool, nothing else. 
    Follow the schema strictly: 

    Guidelines:
    - Use any new corpus_search snippets if available.
    - Add compact explicit references (URLs or sources).
    - Clarify uncertainties and limitations.
    - If more work is needed, set action="reconsider" and explain why in "feedback".
    - Otherwise, set action="end".
    """),
            MessagesPlaceholder("history"),
        ])
        return responder, reviser


