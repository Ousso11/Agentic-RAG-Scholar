# ============================================================
# Derived Class 1: DocSynthesisAgent (retriever tools)
# ============================================================

from langchain.schema import Document
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from agents.base_reflex_agent import BaseReflexAgent, RespondOutput, ReviseOutput


def make_retriever_tool(retriever, name: str = "corpus_search", k: int = 5):
    """
    Wrap a LangChain retriever as a tool returning compact snippets with sources.
    Only used by the respond node.
    """
    @tool(name, return_direct=True)
    def _search(query: str) -> str:
        """Search the local/vector corpus and return up to `k` snippets with sources."""
        docs: List[Document] = retriever.invoke(query)
        parts = []
        for d in docs[:k]:
            src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
            snippet = d.page_content.replace("\n", " ").strip()
            parts.append(f"[{src}] {snippet}")
        return "\n\n".join(parts).strip() or "(no results)"
    return _search


class DocSynthesisAgent(BaseReflexAgent):
    def __init__(self, llm: BaseChatModel, retriever, **kwargs):
        self._retriever_tool = make_retriever_tool(retriever)
        super().__init__(llm=llm, **kwargs)

    def _build_tools(self) -> List:
        return [self._retriever_tool]

    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        responder = ChatPromptTemplate.from_messages([
            ("system",
             """You are a rigorous research assistant.

You MUST return your output ONLY by calling the structured tool RespondOutput.
Follow the schema strictly.

Guidelines:
1. {first_instruction}
2. Use corpus_search to retrieve relevant passages and ground claims.
3. Summarize cautiously: note study type, methods, scope, numbers, and limitations.
4. If evidence is missing, uncertain, or contradictory, include this in 'feedback'.
5. Always propose 1–3 follow-up queries under 'search_queries'.
"""),
            MessagesPlaceholder("history"),
        ])

        reviser = ChatPromptTemplate.from_messages([
            ("system",
             """You are a reviser.

You MUST return your output ONLY by calling the structured tool ReviseOutput.
Follow the schema strictly.

Guidelines:
- Reassess the draft answer in light of retrieved evidence.
- Add compact references (e.g., [Author, Year] or source string).
- Clarify uncertainties and limitations.
- If more work is needed, set action="reconsider" and explain why in 'feedback'.
- Otherwise, set action="end".
- Optionally provide new queries for the next respond iteration.
"""),
            MessagesPlaceholder("history"),
        ])

        return responder, reviser

