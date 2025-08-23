# -- Web search tool (Scholar via SerpAPI, else DDG) for WebSearchAgent
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from agents.base_reflex_agent import BaseReflexAgent
import json
import os

# ============================================================
# Derived Class 2: WebSearchAgent (Scholar/DDG tool + generic research prompts)
# ============================================================

def make_web_search_tool():
    serp = os.getenv("SERPAPI_KEY", "")
    if serp:
        scholar = GoogleScholarQueryRun(
            api_wrapper=GoogleScholarAPIWrapper(serp_api_key=serp, top_k_results=8, hl="en")
        )

        @tool("web_search", return_direct=False)
        def _search(query: str) -> str:
            """Search the web (Google Scholar via SerpAPI) and return JSON of top results."""
            raw = scholar.run(query)
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                data = [{"snippet": raw}]
            norm = [
                {
                    "title": r.get("title") or r.get("Title") or r.get("heading") or "",
                    "link": r.get("link") or r.get("Link") or r.get("href") or "",
                    "snippet": r.get("snippet") or r.get("Snippet") or r.get("body") or "",
                }
                for r in (data or [])
            ]
            return json.dumps(norm[:8], ensure_ascii=False)

        return _search

    ddg = DuckDuckGoSearchResults()

    @tool("web_search", return_direct=False)
    def _search(query: str) -> str:
        """Search the web (DuckDuckGo fallback) and return JSON of top results."""
        raw = ddg.run(query)
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = [{"snippet": raw}]
        norm = [
            {
                "title": r.get("title") or r.get("Title") or r.get("heading") or "",
                "link": r.get("link") or r.get("Link") or r.get("href") or "",
                "snippet": r.get("snippet") or r.get("Snippet") or r.get("body") or "",
            }
            for r in (data or [])
        ]
        return json.dumps(norm[:8], ensure_ascii=False)

    return _search


class WebSearchAgent(BaseReflexAgent):
    """
    Uses a web search tool (Scholar via SerpAPI or DDG fallback) and neutral research prompts.
    """

    def __init__(self, llm: BaseChatModel, **kwargs):
        self._web_tool = make_web_search_tool()
        super().__init__(llm=llm, **kwargs)

    def _build_tools(self) -> List:
        return [self._web_tool]

    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        responder = ChatPromptTemplate.from_messages([
            ("system",
            """You are a rigorous research-paper assistant.
    1. {first_instruction}
    2. When you call web_search, craft precise scholarly queries (boolean operators, synonyms, venue filters, recency if relevant).
    3. Prefer peer-reviewed papers and credible venues (journals, conferences, preprints like arXiv/bioRxiv/medRxiv), authoritative standards, and high-quality surveys.
    4. Synthesize cautiously: identify study type, population/setting, key methods, notable numbers, and limitations; contrast findings when sources disagree.
    5. Keep a brief reflection of what’s missing vs. superfluous.
    6. Propose 1–3 follow-up search queries.
    Return by calling AnswerTool."""),
            MessagesPlaceholder("history"),
        ])

        reviser = ChatPromptTemplate.from_messages([
            ("system",
            """Revise using any new search evidence; include explicit references (e.g., [Author, Year] – title or DOI/URL).
    State consensus vs. open questions and limitations.
    Decide action in ["reconsider","end"] and provide feedback if reconsider.
    Return by calling ReviseTool."""),
            MessagesPlaceholder("history"),
        ])
        return responder, reviser


