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
    Wrap a LangChain retriever as a simple tool returning compact context.
    Only used by the respond node to fetch relevant passages.
    """
    @tool(name, return_direct=True)
    def _search(query: str) -> str:
        """Search the local/vector corpus and return up to `k` compact snippets with sources."""
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
    Reflex agent using a local/vector retriever tool.
    Respond node fetches documents via corpus_search and synthesizes answers.
    Revise node evaluates draft and decides end/reconsider with optional new queries.
    """

    def __init__(self, llm: BaseChatModel, retriever, **kwargs):
        self._retriever_tool = make_retriever_tool(retriever)
        super().__init__(llm=llm, **kwargs)

    def _build_tools(self) -> List:
        # Only respond node uses this retriever
        return [self._retriever_tool]

    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        # Responder prompt: fetch, synthesize, provide feedback + queries
        responder = ChatPromptTemplate.from_messages([
            ("system",
             """You are a rigorous research assistant.
You MUST return your output ONLY by calling the structured tool RespondOutput.
Follow the schema strictly.

Guidelines:
1. {first_instruction}
2. Use corpus_search to retrieve relevant passages and ground claims.
3. Synthesize cautiously: identify study type, methods, scope, limitations.
4. If evidence is missing or uncertain, note under 'feedback'.
5. Suggest 1–3 follow-up queries under 'search_queries'.
"""),
            MessagesPlaceholder("history"),
        ])

        # Reviser prompt: analyze draft, provide critique, decide action
        reviser = ChatPromptTemplate.from_messages([
            ("system",
             """You are a reviser.
You MUST return your output ONLY by calling the structured tool ReviseOutput.
Follow the schema strictly.

Guidelines:
- Use the draft answer and any feedback from the respond node.
- Add compact explicit references if applicable.
- Clarify uncertainties and limitations.
- If more work is needed, set action="reconsider" and explain in 'feedback'.
- Otherwise, set action="end".
- Optionally provide new queries for the next respond iteration.
"""),
            MessagesPlaceholder("history"),
        ])
        return responder, reviser
