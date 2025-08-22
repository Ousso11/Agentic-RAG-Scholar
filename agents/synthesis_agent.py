# agents/doc_synthesis_agent.py
from __future__ import annotations
import logging
from typing import List, Dict, Optional, TypedDict, Literal
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


# -------- State --------
class GraphState(TypedDict, total=False):
    question: str
    query: str
    iteration: int
    results: List[Document]
    answer: Optional["AnswerQuestion"]
    revised: Optional["ReviseAnswer"]
    conversation: List[Dict]


# -------- Output Schemas --------
class Reflection(BaseModel):
    missing: List[str] = Field(default_factory=list)
    superfluous: List[str] = Field(default_factory=list)


class AnswerQuestion(BaseModel):
    answer: str
    reflection: Reflection
    search_queries: List[str] = Field(default_factory=list)


class ReviseAnswer(BaseModel):
    answer: str
    references: List[str] = Field(default_factory=list)
    reflection: Reflection
    search_queries: List[str] = Field(default_factory=list)


# -------- Main Agent --------
class DocSynthesisAgent:
    def __init__(self, llm, retriever, max_iters: int = 2, logger: logging.Logger = None):
        self.llm = llm
        self.retriever = retriever
        self.max_iters = max_iters
        self.logger = logger or logging.getLogger(__name__)

        self.responder_chain = self._build_responder_chain()
        self.revisor_chain = self._build_revisor_chain()
        self.graph = self._build_graph()

    # ---- LLM Chains ----
    def _build_responder_chain(self):
        parser = JsonOutputParser(pydantic_object=AnswerQuestion)

        template = """
You are answering a question from retrieved documents.

Return ONLY valid JSON strictly matching this schema:
{format_instructions}

Rules:
- No explanations outside JSON.
- `answer` must be plain text.
- `reflection.missing` and `reflection.superfluous` must be arrays of strings.
- `search_queries` must be an array of strings.

Question:
{question}

Retrieved context:
{context}
        """.strip()

        prompt = ChatPromptTemplate.from_template(
            template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | self.llm | parser

    def _build_revisor_chain(self):
        parser = JsonOutputParser(pydantic_object=ReviseAnswer)

        template = """
You are revising a previously drafted answer using new retrieved context.

Return ONLY valid JSON strictly matching this schema:
{format_instructions}

Rules:
- No explanations outside JSON.
- `answer` must be plain text.
- `references` is an array of strings (citations if applicable).
- `reflection.missing` and `reflection.superfluous` must be arrays of strings.
- `search_queries` must be an array of strings.

Previous answer:
{prev_answer}

Reflection:
- Missing: {missing}
- Superfluous: {superfluous}

New retrieved context:
{context}
        """.strip()

        prompt = ChatPromptTemplate.from_template(
            template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | self.llm | parser

    # ---- Graph Nodes ----
    def node_retrieve(self, state: GraphState) -> GraphState:
        q = state.get("query") or state["question"]
        results = self.retriever.invoke(q)
        self.logger.info(f"[DocSynth] Retrieved {len(results)} docs for query: {q}")
        conv = state.get("conversation", []) + [{"action": "retrieve", "query": q, "n": len(results)}]
        return {**state, "results": results, "conversation": conv}

    def node_respond(self, state: GraphState) -> GraphState:
        ctx = "\n\n".join(d.page_content for d in state.get("results", [])[:5])
        try:
            answer = self.responder_chain.invoke({"question": state["question"], "context": ctx})
            if isinstance(answer, dict):
                answer = AnswerQuestion(**answer)
        except Exception as e:
            self.logger.error(f"[DocSynth] Responder parse error: {e}")
            answer = AnswerQuestion(answer="", reflection=Reflection(), search_queries=[])

        self.logger.info(f"[DocSynth] Draft answer: {answer.answer[:200]}...")
        i = state.get("iteration", 0) + 1
        conv = state.get("conversation", []) + [{"action": "respond", "answer": answer.dict()}]
        return {**state, "answer": answer, "conversation": conv, "iteration": i}

    def node_revise(self, state: GraphState) -> GraphState:
        prev = state.get("answer")
        if not prev:
            return state
        ctx = "\n\n".join(d.page_content for d in state.get("results", [])[:5])
        try:
            revised = self.revisor_chain.invoke({
                "prev_answer": prev.answer,
                "missing": prev.reflection.missing,
                "superfluous": prev.reflection.superfluous,
                "context": ctx
            })
            if isinstance(revised, dict):
                revised = ReviseAnswer(**revised)
        except Exception as e:
            self.logger.error(f"[DocSynth] Revisor parse error: {e}")
            revised = ReviseAnswer(answer=prev.answer, references=[], reflection=Reflection(), search_queries=[])

        self.logger.info(f"[DocSynth] Revised answer: {revised.answer[:200]}...")
        conv = state.get("conversation", []) + [{"action": "revise", "revised": revised.dict()}]
        return {**state, "revised": revised, "conversation": conv}

    # ---- Controller ----
    def should_continue(self, state: GraphState) -> Literal["retrieve", "respond", "end"]:
        i = state.get("iteration", 0)
        if i >= self.max_iters:
            return "end"

        revised = state.get("revised")
        if revised and revised.search_queries:
            state["query"] = revised.search_queries[0]
            return "retrieve"

        if revised:
            return "respond"

        return "end"

    # ---- Graph ----
    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node("respond", self.node_respond)
        g.add_node("revise", self.node_revise)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "respond")
        g.add_edge("respond", "revise")
        g.add_conditional_edges("revise", self.should_continue, {
            "retrieve": "retrieve",
            "respond": "respond",
            "end": END
        })
        return g.compile()

    # ---- Run ----
    def run(self, question: str) -> Dict:
        init: GraphState = {"question": question, "query": question, "iteration": 0}
        final = self.graph.invoke(init)

        citations = []
        if final.get("revised"):
            citations = final["revised"].references

        final_answer = None
        if final.get("revised"):
            final_answer = final["revised"].answer
        elif final.get("answer"):
            final_answer = final["answer"].answer

        return {
            "question": final.get("question"),
            "conversation": final.get("conversation", []),
            "citations": citations,
            "final_answer": final_answer
        }
