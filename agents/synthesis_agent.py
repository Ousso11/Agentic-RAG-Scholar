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
            """You are a rigorous research-paper assistant using a local corpus.
    1. {first_instruction}
    2. Use the corpus_search tool to retrieve the most relevant passages; ground claims in those snippets.
    3. Synthesize cautiously: identify study type, methods, scope, and limitations; avoid over-generalizing beyond the retrieved evidence.
    4. If evidence is insufficient or uncertain, say so and list what’s missing.
    5. Provide 1–3 suggested web follow-up queries that would likely improve coverage.
    Return by calling AnswerTool."""),
            MessagesPlaceholder("history"),
            ("system", "Answer concisely first, then expand with evidence and a short reflection.")
        ])

        reviser = ChatPromptTemplate.from_messages([
            ("system",
            """Revise the prior answer using any new tool context (retrieved snippets).
    Add compact, explicit references from the corpus (e.g., use each chunk’s source/URL/identifier if available).
    Clarify uncertainties and limitations.
    Decide action in ["reconsider","end"]; provide feedback if reconsider.
    Return by calling ReviseTool."""),
            MessagesPlaceholder("history"),
        ])
        return responder, reviser

