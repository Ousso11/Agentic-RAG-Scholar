# ============================================================
# Derived Class 2: WebSearchAgent (Scholar/DDG tool + reflex agent)
# ============================================================

from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from agents.base_reflex_agent import BaseReflexAgent, RespondOutput, ReviseOutput
import json, os


def make_web_search_tool():
    """
    Wrap web search (Scholar via SerpAPI, else DuckDuckGo) as a tool.
    Only used by the respond node.
    """
    serp = os.getenv("SERPAPI_KEY", "")
    if serp:
        scholar = GoogleScholarQueryRun(
            api_wrapper=GoogleScholarAPIWrapper(serp_api_key=serp, top_k_results=8, hl="en")
        )

        @tool("web_search", return_direct=True)
        def _search(query: str) -> str:
            """Search the web (Scholar via SerpAPI) and return top results as JSON."""
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

    # DuckDuckGo fallback
    ddg = DuckDuckGoSearchResults()

    @tool("web_search", return_direct=True)
    def _search(query: str) -> str:
        """Search the web (DuckDuckGo fallback) and return top results as JSON."""
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
    def __init__(self, llm: BaseChatModel, **kwargs):
        self._web_tool = make_web_search_tool()
        super().__init__(llm=llm, **kwargs)

    def _build_tools(self) -> List:
        return [self._web_tool]

    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        responder_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a rigorous research assistant.

Return ONLY valid JSON strictly matching the RespondOutput schema:
{format_instructions}

Guidelines:
1. Always start with a concise direct answer, then expand with synthesis of the context.
2. If you used web_search, identify the key findings (methods, study type, population, numbers, limitations).
3. Mention disagreements if sources diverge.
4. Provide self-feedback (what's missing, redundant, or uncertain).
5. Propose 1–3 improved search queries for the next round.
"""),
            MessagesPlaceholder("history"),
        ])
        responder_parser = JsonOutputParser(pydantic_object=RespondOutput)
        responder_prompt = responder_prompt.partial(format_instructions=responder_parser.get_format_instructions())

        reviser_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a critical reviser.

Return ONLY valid JSON strictly matching the ReviseOutput schema:
{format_instructions}

Guidelines:
1. Review the draft answer and self-feedback.
2. If additional evidence is needed, suggest 1–3 new queries.
3. Return a fully revised answer (not just feedback).
4. Decide action: "reconsider" (if more search is needed) or "end".
5. Provide constructive feedback for improvement.
"""),
            MessagesPlaceholder("history"),
        ])
        reviser_parser = JsonOutputParser(pydantic_object=ReviseOutput)
        reviser_prompt = reviser_prompt.partial(format_instructions=reviser_parser.get_format_instructions())

        return responder_prompt, reviser_prompt