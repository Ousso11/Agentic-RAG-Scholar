# web_search_agent.py
import os
import json
from typing import List, Dict
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain_community.llms import Ollama

# Simple in-memory "episodic memory" for reflections
class ReflectionMemory:
    def __init__(self):
        self.critiques: List[str] = []

    def add(self, critique: str):
        self.critiques.append(critique)

    def dump(self) -> str:
        return "\n".join(f"- {c}" for c in self.critiques[-5:])  # last 5 reflections

class WebSearchAgent:
    """
    Reflexion-ish loop:
      - ACT (search tool) → OBSERVE (results) → REFLECT (LLM critique) → if weak, iterate.
    """
    def __init__(self, llm_model_name: str, max_iters: int = 2):
        self.llm = Ollama(model=llm_model_name)
        self.memory = ReflectionMemory()
        self.max_iters = max_iters

        # Prefer Google Scholar if SERPAPI_KEY is set, else fallback to DuckDuckGo
        self.tools: List[Tool] = []
        serp = os.getenv("SERPAPI_KEY", "")
        if serp:
            scholar_wrapper = GoogleScholarAPIWrapper(serp_api_key=serp, top_k_results=8, hl="en")
            self.tools.append(GoogleScholarQueryRun(api_wrapper=scholar_wrapper))
        else:
            # Fallback (no Scholar metadata like citations, but still useful)
            self.tools.append(DuckDuckGoSearchResults())

    def _run_tool(self, query: str) -> str:
        # Use the first configured tool
        tool = self.tools[0]
        return tool.run(query)

    def search_papers(self, query: str) -> List[Dict]:
        """
        Returns a list of dicts with {title, link, snippet}
        """
        results_agg: List[Dict] = []
        current_query = query

        for _ in range(self.max_iters):
            raw = self._run_tool(current_query)
            # Tool outputs vary: try to normalize
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    data = [{"snippet": raw}]
            else:
                data = raw

            # Normalize common fields
            normalized = []
            for r in data:
                normalized.append({
                    "title": r.get("title") or r.get("Title") or r.get("heading") or "",
                    "link": r.get("link") or r.get("Link") or r.get("href") or "",
                    "snippet": r.get("snippet") or r.get("Snippet") or r.get("body") or ""
                })
            results_agg = normalized if normalized else results_agg

            # --- Reflexion critique & possible refinement ---
            critique_prompt = f"""
You are a research assistant using a search tool to find academic papers.
Task: Critique the usefulness of these results for the query, then suggest a refined query if needed.

Original Query: {query}

Recent Reflections:
{self.memory.dump()}

Results (first 5):
{json.dumps(results_agg[:5], indent=2)}

Respond in JSON with keys:
- "critique": short critique
- "refined_query": NULL if not needed else a better query
"""
            critique = self.llm.invoke(critique_prompt)
            try:
                crit = json.loads(critique)
            except Exception:
                crit = {"critique": critique.strip(), "refined_query": None}

            self.memory.add(crit.get("critique", "").strip())

            refined = crit.get("refined_query")
            if not refined or str(refined).upper() == "NULL":
                break
            current_query = str(refined)

        return results_agg
