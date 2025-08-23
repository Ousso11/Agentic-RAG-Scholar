import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal, TypedDict, Tuple

from pydantic import BaseModel, Field
import json
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
        self.logger.debug(f"Prompt messages: {msgs}")
        
        # First pass
        resp = llm.invoke(msgs)
        self.logger.debug(f"Raw LLM response: {resp}")

        if getattr(resp, "tool_calls", None):
            self.logger.debug(f"Tool calls detected: {resp.tool_calls}")

            for tc in resp.tool_calls:
                name, args, tcid = tc.get("name"), tc.get("args", {}), tc.get("id")
                self.logger.debug(f"Processing tool call: name={name}, args={args}, id={tcid}")

                # Clean args keys in case of extra quotes
                if isinstance(args, dict):
                    args = {str(k).strip().strip('"'): v for k, v in args.items()}

                # Structured output directly
                if name == structured_tool.__name__:
                    self.logger.debug(f"Matched structured tool {structured_tool.__name__} with args={args}")
                    return args, [resp]

                # Allow ONE external tool call, then follow-up
                for t in self.external_tools:
                    if name == t.name:
                        self.logger.debug(f"Invoking external tool {t.name} with args={args}")
                        out = t.invoke(args)
                        msgs2 = msgs + [resp, ToolMessage(content=str(out), tool_call_id=tcid)]
                        self.logger.debug(f"External tool {t.name} returned: {out}")

                        resp2 = llm.invoke(msgs2)
                        self.logger.debug(f"Follow-up LLM response: {resp2}")

                        if getattr(resp2, "tool_calls", None):
                            for tc2 in resp2.tool_calls:
                                name2, args2 = tc2.get("name"), tc2.get("args", {})
                                args2 = {str(k).strip().strip('"'): v for k, v in args2.items()}
                                if name2 == structured_tool.__name__:
                                    self.logger.debug(f"Matched structured tool {structured_tool.__name__} on second pass with args={args2}")
                                    return args2, [resp, ToolMessage(content=str(out), tool_call_id=tcid), resp2]

                        # Fallback if no structured tool output
                        empty = {"answer": "", "reflection": {"missing": [], "superfluous": []}, "search_queries": []}
                        self.logger.warning("Model failed to emit structured tool output after external tool call. Returning empty default.")
                        return empty, [resp, ToolMessage(content=str(out), tool_call_id=tcid), resp2]

        # No tool calls; treat raw content as simple answer
        self.logger.debug("No tool calls detected. Falling back to simple AnswerTool structure.")
        return (
            {
                "answer": getattr(resp, "content", "") or "",
                "reflection": {"missing": [], "superfluous": []},
                "search_queries": [],
            },
            [resp],
        )

    # ----- nodes -----
    def node_respond(self, state: State) -> State:
        self.logger.debug(f"Node RESPOND called at iteration {state.get('iteration', 0)}")
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
        self.logger.debug(f"Draft answer: {draft.answer[:200]}... with reflection missing={draft.reflection.missing}, superfluous={draft.reflection.superfluous}, search_queries={draft.search_queries}")
        return {**state, "draft": draft, "conversation": conv}

    def node_revise(self, state: State) -> State:
        self.logger.debug(f"Node REVISE called at iteration {state.get('iteration', 0)}")
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
        self.logger.debug(f"Revised answer: {revised.answer[:200]}... with feedback: {revised.feedback}")
        return {**state, "revised": revised, "iteration": i, "conversation": conv, "feedback": revised.feedback}

    # ----- controller & API -----
    def _should_continue(self, state: State) -> Literal["respond", "end"]:
        self.logger.debug(f"Checking if should continue at iteration {state.get('iteration', 0)}")
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


# ----- helper for cleaning tool call args -----
def _normalize_args(args):
    """
    Ensure the tool call args are a proper dict with clean keys.
    Handles:
    - args as stringified JSON
    - keys with extra quotes
    - nested JSON strings in values
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    if not isinstance(args, dict):
        return {}

    cleaned = {}
    for k, v in args.items():
        key = str(k).strip().strip('"').strip("'")
        # Parse JSON values if they are strings
        if isinstance(v, str):
            try:
                v_parsed = json.loads(v)
                v = v_parsed
            except Exception:
                pass
        cleaned[key] = v
    return cleaned
