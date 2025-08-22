import os
import json
import logging
from typing import List, Dict, Optional, Literal, TypedDict

from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
# LangGraph (lightweight state machine)
from langgraph.graph import StateGraph, END

from langchain_core.pydantic_v1 import BaseModel, Field


# ------------------------- Structured Schemas ------------------------- #
class Reflection(BaseModel):
    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )

 # ---------- Build LangGraph ---------- #
class GraphState(TypedDict, total=False):
    question: str
    query: str
    results: List[Dict]
    iteration: int
    answer: AnswerQuestion
    revised: ReviseAnswer
    conversation: List[Dict]

# ------------------------- Memory ------------------------- #
class ReflectionMemory:
    def __init__(self):
        self.critiques: List[str] = []

    def add(self, critique: str):
        if critique:
            self.critiques.append(critique)

    def dump(self) -> str:
        return "\n".join(f"- {c}" for c in self.critiques[-5:])  # last 5


# ------------------------- Agent ------------------------- #
class WebSearchAgent:
    """
    Reflexion-ish loop with LangGraph:
      SEARCH → RESPOND → REVISE → (optionally SEARCH again) ... until stop.

    Two LCEL chains:
      - responder_chain: PromptTemplate | Ollama | PydanticOutputParser(AnswerQuestion)
      - revisor_chain:   PromptTemplate | Ollama | PydanticOutputParser(ReviseAnswer)
    """

    def __init__(self, llm_model_name: str, max_iters: int = 2, logger: Optional[logging.Logger] = None):
        self.llm = Ollama(model=llm_model_name)
        self.max_iters = max_iters
        self.memory = ReflectionMemory()
        self.logger = logger or logging.getLogger(__name__)

        # Configure search tool preference
        self.tools: List = []
        serp = os.getenv("SERPAPI_KEY", "")
        if serp:
            scholar_wrapper = GoogleScholarAPIWrapper(serp_api_key=serp, top_k_results=8, hl="en")
            self.tools.append(GoogleScholarQueryRun(api_wrapper=scholar_wrapper))
        else:
            self.tools.append(DuckDuckGoSearchResults())

        self.build_responder_chain()
        self.build_revisor_chain()
        
        graph = StateGraph(GraphState)
        graph.add_node("search", self.node_search)
        graph.add_node("respond", self.node_respond)
        graph.add_node("revise", self.node_revise)

        graph.set_entry_point("search")
        graph.add_edge("search", "respond")
        graph.add_edge("respond", "revise")
        graph.add_conditional_edges("revise", self.should_continue, {"search": "search", "end": END})
        graph.add_edge("respond", END)

        self.graph = graph.compile()


    def build_responder_chain(self):
        # ---------- Build responder chain (AnswerQuestion) ---------- #
        self.answer_parser = JsonOutputParser(pydantic_object=AnswerQuestion)
        responder_tmpl = (
            """
You are a research assistant. Given a user question, recent reflections, and search snippets,
write a concise, factual answer. Be specific. Then self-critique: what is missing and what is unnecessary?
Finally, propose up to 3 concrete search queries to fill gaps or verify claims.

Return ONLY JSON matching this schema:
{format_instructions}

User Question:
{question}

Recent Reflections:
{reflections}

Search Results (first 8, JSON):
{results_json}
            """
        ).strip()
        self.responder_prompt = PromptTemplate(
            template=responder_tmpl,
            input_variables=["question", "reflections", "results_json"],
            partial_variables={"format_instructions": self.answer_parser.get_format_instructions()},
        )
        self.responder_chain = self.responder_prompt | self.llm | self.answer_parser

    
    # ---------- Build revisor chain (ReviseAnswer) ---------- #
    def build_revisor_chain(self):
        self.revise_parser = JsonOutputParser(pydantic_object=ReviseAnswer)
        revisor_tmpl = (
            """
You are revising your previous answer. Using the original answer JSON, reflections, and any new search results,
produce an improved answer. Include explicit references (URLs, DOIs, titles) that justify the update.

Return ONLY JSON matching this schema:
{format_instructions}

User Question:
{question}

Original Answer JSON:
{original_answer_json}

Recent Reflections:
{reflections}

Search Results (JSON):
{results_json}
            """
        ).strip()
        self.revisor_prompt = PromptTemplate(
            template=revisor_tmpl,
            input_variables=["question", "original_answer_json", "reflections", "results_json"],
            partial_variables={"format_instructions": self.revise_parser.get_format_instructions()},
        )
        self.revisor_chain = self.revisor_prompt | self.llm | self.revise_parser

    def _run_tool(self, query: str) -> List[Dict]:
        tool = self.tools[0]
        raw = tool.run(query)
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = [{"snippet": raw}]
        else:
            data = raw
        normalized = []
        for r in data:
            normalized.append(
                {
                    "title": r.get("title") or r.get("Title") or r.get("heading") or "",
                    "link": r.get("link") or r.get("Link") or r.get("href") or "",
                    "snippet": r.get("snippet") or r.get("Snippet") or r.get("body") or "",
                }
            )
        return normalized

    def node_search(self, state: GraphState) -> GraphState:
        
        q = state.get("query") or state["question"]
        results = self._run_tool(q)
        conv = state.get("conversation", []) + [{"action": "search", "query": q, "results": results}]
        return {**state, "results": results, "conversation": conv}

    def node_respond(self, state: GraphState) -> GraphState:
        results_json = json.dumps(state.get("results", [])[:8], indent=2)
        reflections = self.memory.dump()
        answer = self.responder_chain.invoke(
            {
                "question": state["question"],
                "reflections": reflections,
                "results_json": results_json,
            }
        )
        # --- FIX: Ensure we always have a Pydantic object
        if isinstance(answer, dict):
            answer = AnswerQuestion.parse_obj(answer)

        self.memory.add(
            f"Missing: {answer.reflection.missing} | Superfluous: {answer.reflection.superfluous}"
        )
        conv = state.get("conversation", []) + [{"action": "respond", "answer": answer.dict()}]
        return {**state, "answer": answer, "conversation": conv}

    def node_revise(self, state: GraphState) -> GraphState:
        results_json = json.dumps(state.get("results", [])[:8], indent=2)
        reflections = self.memory.dump()
        original_answer_json = state["answer"].json()
        revised: ReviseAnswer = self.revisor_chain.invoke(
            {
                "question": state["question"],
                "original_answer_json": original_answer_json,
                "reflections": reflections,
                "results_json": results_json,
            }
        )
        conv = state.get("conversation", []) + [{"action": "revise", "revised": revised.dict()}]
        return {**state, "revised": revised, "conversation": conv}

    def should_continue(self, state: GraphState) -> Literal["search", "end"]:
        i = int(state.get("iteration", 0))
        if i + 1 >= self.max_iters:
            return "end"
        queries = []
        if "revised" in state and state["revised"].search_queries:
            queries = state["revised"].search_queries
        elif "answer" in state and state["answer"].search_queries:
            queries = state["answer"].search_queries
        if queries:
            return "search"
        return "end"
        
    # ------------------------- Public API ------------------------- #
    def run(self, question: str) -> Dict:
        """
        Execute the LangGraph. Returns a dict with keys:
        - question (the original question)
        - conversation (full sequence of steps with queries, answers, revisions)
        - citations (all references aggregated from revised answers)
        - final_answer (string from the latest revised answer, or responder if no revision)
        """
        initial_state = {
            "question": question,
            "query": question,
            "iteration": 0,
            "results": [],
            "conversation": [],
        }
        final_state = self.graph.invoke(initial_state)

        citations = []
        if final_state.get("revised"):
            citations = final_state["revised"].references

        final_answer = None
        if final_state.get("revised"):
            final_answer = final_state["revised"].answer
        elif final_state.get("answer"):
            final_answer = final_state["answer"].answer

        out = {
            "question": final_state.get("question"),
            "conversation": final_state.get("conversation", []),
            "citations": citations,
            "final_answer": final_answer,
        }
        return out
