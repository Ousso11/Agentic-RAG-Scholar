# relevance_checker.py
import logging
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self, model_name: str):
        """
        Keep the exact logic you had, just replace WatsonX with Ollama.
        """
        self.llm = Ollama(model=model_name)

    def check(self, question: str, retriever, k: int = 3) -> str:
        """
        1) Retrieve top-k chunks
        2) Build prompt
        3) Return one of: CAN_ANSWER, PARTIAL, NO_MATCH
        """
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        top_docs = retriever.invoke(question) if retriever else []
        if not top_docs:
            logger.debug("No documents from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""
You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

**Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

**Question:** {question}
**Passages:** {document_content}

**Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """.strip()

        try:
            llm_response = self.llm.invoke(prompt).strip().upper()
            logger.debug(f"LLM response: {llm_response}")
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        return llm_response if llm_response in valid_labels else "NO_MATCH"
