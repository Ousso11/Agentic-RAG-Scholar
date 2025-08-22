from typing import Literal
import logging

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class RelevanceResult(BaseModel):
    """Structured result for relevance classification.

    - label: one of CAN_ANSWER, PARTIAL, NO_MATCH
    - relevance: integer 0..5 (how related the passages are to the question)
    - confidence: integer 0..5 (model self-rated certainty)
    - can_answer: True iff label == CAN_ANSWER
    """

    label: Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"] = Field(
        ..., description="Classification label"
    )
    relevance: int = Field(
        ..., description="Relevance score from 0 (unrelated) to 5 (fully about the question)", ge=0, le=5
    )
    confidence: int = Field(
        ..., description="Self-rated certainty from 0 to 5 inclusive", ge=0, le=5
    )
    can_answer: bool = Field(
        ..., description='True if and only if label == "CAN_ANSWER"'
    )


class RelevanceChecker:
    def __init__(self, model_name: str, logger: logging.Logger | None = None):
        """
        Upgraded version that:
        - Uses a PromptTemplate + LCEL chain (prompt | llm | parser)
        - Produces a structured JSON result with a Pydantic parser
        """
        self.llm = Ollama(model=model_name)
        self.logger = logger or logging.getLogger(__name__)

        # Structured output via Pydantic
        self.parser = JsonOutputParser(pydantic_object=RelevanceResult)

        # Prompt template with format instructions injected once at init
        template = (
            """
You are an AI relevance checker between a user's question and provided document content.

Classify and score how well the document content addresses the user's question.

Return ONLY valid JSON that matches this schema:
{format_instructions}

Rules:
- "label" must be exactly one of: CAN_ANSWER, PARTIAL, NO_MATCH.
- If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, use "PARTIAL" rather than "NO_MATCH".
- "relevance" is an integer from 0 to 5 inclusive (0 = unrelated, 5 = fully about the question).
- "confidence" is an integer from 0 to 5 inclusive.
- "can_answer" must be true if and only if "label" == "CAN_ANSWER".

Question:
{question}

Passages:
{passages}
            """
        ).strip()

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["question", "passages"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        # Build the chain once: Prompt -> LLM -> Pydantic parser
        self.chain = self.prompt | self.llm | self.parser

    def check(self, question: str, retriever, k: int = 3) -> dict:
        """
        1) Retrieve top-k chunks
        2) Build inputs and run LCEL chain
        3) Return structured JSON dict with: label, relevance (0..5), confidence (0..5), can_answer (bool)
        """
        self.logger.debug(
            f"RelevanceChecker.check called with question='{question}' and k={k}"
        )

        top_docs = retriever.invoke(question) if retriever else []
        if not top_docs:
            self.logger.debug(
                "No documents from retriever.invoke(). Returning NO_MATCH with zeros."
            )
            fallback = RelevanceResult(
                label="NO_MATCH", relevance=0, confidence=0, can_answer=False
            )
            return fallback.dict()

        # Join top-k page contents; be resilient if objects lack page_content
        document_content = "\n\n".join(
            getattr(doc, "page_content", str(doc)) for doc in top_docs[:k]
        )
        self.logger.debug(f"[DEBUG] Top-{k} document content for relevance check:\n{document_content}")
        try:
            result: RelevanceResult = self.chain.invoke(
                {"question": question, "passages": document_content}
            )

            if isinstance(result, dict):
                result = RelevanceResult.parse_obj(result)
                
            # Enforce logical consistency & clamp numeric ranges defensively
            result.can_answer = result.label == "CAN_ANSWER"
            result.relevance = max(0, min(5, int(result.relevance)))
            result.confidence = max(0, min(5, int(result.confidence)))

            self.logger.debug(f"Structured LLM result: {result.json()}")
            return result.dict()
        except Exception as e:
            self.logger.error(f"Ollama inference error: {e}")
            fallback = RelevanceResult(
                label="NO_MATCH", relevance=0, confidence=0, can_answer=False
            )
            return fallback.dict()

