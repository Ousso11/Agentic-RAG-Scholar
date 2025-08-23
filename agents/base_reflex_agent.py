import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, TypedDict, Tuple

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel


class Reflection(BaseModel):
    missing: List[str] = Field(default_factory=list)
    superfluous: List[str] = Field(default_factory=list)


class AnswerTool(BaseModel):
    """Structured answer payload."""
    answer: str
    reflection: Reflection
    search_queries: List[str] = Field(default_factory=list)


class ReviseTool(BaseModel):
    """Structured revision payload + control signal."""
    answer: str
    references: List[str] = Field(default_factory=list)
    reflection: Reflection
    search_queries: List[str] = Field(default_factory=list)
    action: Literal["reconsider", "end"] = "end"
    feedback: str = ""


# ============================================================
# Abstract Base Agent
# ============================================================
class BaseReflexAgent(ABC):
    """
    Reflection agent with reconsider loop:
      respond → revise → (if action='reconsider' → respond) … until max_iters.
    Single external tool call allowed per node (then one follow-up pass).
    Subclasses only implement: prompts + tools.
    """

    class State(TypedDict, total=False):
        question: str
        iteration: int
        conversation: List[Dict]
        draft: AnswerTool
        revised: ReviseTool
        feedback: str

    def __init__(self, llm: BaseChatModel, max_iters: int = 3, logger: Optional[logging.Logger] = None):
        self.llm = llm
        self.max_iters = max_iters
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.responder_prompt, self.revision_prompt = self._build_prompts()
        self.external_tools = self._build_tools()  # e.g., [corpus_search] or [web_search]

        g = StateGraph(self.State)
        g.add_node("respond", self.node_respond)
        g.add_node("revise", self.node_revise)
        g.set_entry_point("respond")
        g.add_edge("respond", "revise")
        g.add_conditional_edges("revise", self._should_continue, {"respond": "respond", "end": END})
        self.graph = g.compile()

    # ----- subclass contract -----
    @abstractmethod
    def _build_prompts(self) -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
        ...

    @abstractmethod
    def _build_tools(self) -> List:
        ...

    # ----- single external tool call per node -----
    def _run_once_with_tools(self, prompt: ChatPromptTemplate, structured_tool, history, first_instruction: str):
        msgs = prompt.format_messages(first_instruction=first_instruction, history=history)
        llm = self.llm.bind_tools(tools=[structured_tool, *self.external_tools])

        # First pass
        resp = llm.invoke(msgs)
        if getattr(resp, "tool_calls", None):
            for tc in resp.tool_calls:
                name, args, tcid = tc["name"], tc.get("args", {}), tc["id"]

                # Structured output directly
                if name == structured_tool.__name__:
                    return tc.get("args", {}), [resp]

                # Allow ONE external tool call, then follow-up
                for t in self.external_tools:
                    if name == t.name:
                        out = t.invoke(args)
                        msgs2 = msgs + [resp, ToolMessage(content=str(out), tool_call_id=tcid)]
                        resp2 = llm.invoke(msgs2)
                        if getattr(resp2, "tool_calls", None):
                            for tc2 in resp2.tool_calls:
                                if tc2["name"] == structured_tool.__name__:
                                    return tc2.get("args", {}), [resp, ToolMessage(content=str(out), tool_call_id=tcid), resp2]

                        # Fallback if the model fails to emit structured tool
                        empty = {"answer": "", "reflection": {"missing": [], "superfluous": []}, "search_queries": []}
                        return empty, [resp, ToolMessage(content=str(out), tool_call_id=tcid), resp2]

        # No tool calls; treat raw content as simple answer
        return (
            {"answer": getattr(resp, "content", "") or "", "reflection": {"missing": [], "superfluous": []}, "search_queries": []},
            [resp],
        )

    # ----- nodes -----
    def node_respond(self, state: State) -> State:
        hist = []
        if state.get("iteration", 0) == 0:
            hist.append(HumanMessage(content=state["question"]))
        else:
            fb = state.get("feedback") or ""
            hist.append(HumanMessage(content=f"Reconsider with this feedback:\n{fb}\n\nOriginal question:\n{state['question']}"))

        args, _ = self._run_once_with_tools(
            prompt=self.responder_prompt,
            structured_tool=AnswerTool,
            history=hist,
            first_instruction="Answer concisely first, then expand."
        )
        draft = AnswerTool(**args) if args.get("answer") is not None else AnswerTool(
            answer="", reflection=Reflection(), search_queries=[]
        )
        conv = state.get("conversation", [])
        conv.append({"action": "respond", "queries": draft.search_queries})
        return {**state, "draft": draft, "conversation": conv}

    def node_revise(self, state: State) -> State:
        draft: AnswerTool = state["draft"]
        prior = f"Previous answer:\n{draft.answer}\n\nReflection:\nmissing={draft.reflection.missing}; superfluous={draft.reflection.superfluous}"

        args, _ = self._run_once_with_tools(
            prompt=self.revision_prompt,
            structured_tool=ReviseTool,
            history=[HumanMessage(content=prior)],
            first_instruction="Critique, improve, then decide: reconsider or end."
        )
        revised = ReviseTool(**args) if args.get("answer") is not None else ReviseTool(
            answer=draft.answer, references=[], reflection=draft.reflection, search_queries=[], action="end"
        )
        i = state.get("iteration", 0) + 1
        conv = state.get("conversation", [])
        conv.append({"action": "revise", "decision": revised.action, "refs": revised.references})
        return {**state, "revised": revised, "iteration": i, "conversation": conv, "feedback": revised.feedback}

    # ----- controller & API -----
    def _should_continue(self, state: State) -> Literal["respond", "end"]:
        if state.get("iteration", 0) >= self.max_iters:
            return "end"
        return "respond" if state["revised"].action == "reconsider" else "end"

    def run(self, question: str) -> Dict:
        init: BaseReflexAgent.State = {"question": question, "iteration": 0, "conversation": []}
        final = self.graph.invoke(init)
        final_answer = (final.get("revised") and final["revised"].answer) or (final.get("draft") and final["draft"].answer) or ""
        citations = (final.get("revised") and final["revised"].references) or []
        return {
            "question": question,
            "final_answer": final_answer,
            "citations": citations,
            "conversation": final.get("conversation", []),
            "iterations": final.get("iteration", 0),
        }

