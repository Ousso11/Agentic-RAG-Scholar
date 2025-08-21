# Agentic-RAG-Scholar
An Agentic RAG system for intelligent research paper analysis. This application uses an agentic approach to retrieve, synthesize, and cite information from academic papers with natural language queries.
# Agentic RAG for Research Papers

### Project Overview
This application is a sophisticated, multi-agent system that goes beyond traditional keyword search to provide comprehensive, cited answers to natural language questions about research papers. Powered by a **Reflexion Agent** and a dynamic routing mechanism, it intelligently processes documents and synthesizes information to deliver high-quality, transparent insights.

***

### Agentic Workflow

1.  **Document Processing Agent:**
    * **Role:** The system's **ingestion specialist**.
    * **Process:** Takes a PDF/URL, uses **Docling** for structured parsing, and splits the content into meaningful chunks. It then creates a new, dedicated collection in **ChromaDB** for the document's content and a high-level metadata entry in a central **Collection Hub**. 

2.  **Relevance Retriever Agent:**
    * **Role:** The **preliminary search agent**.
    * **Process:** This is the first agent to run on a query. It performs a fast, high-level search by querying the central **Collection Hub** using a combination of vector search and keyword matching. It assesses whether the current corpus contains relevant documents for the user's query and identifies the specific collections to search.

3.  **Relevance Checking Agent:**
    * **Role:** The intelligent **gatekeeper**.
    * **Process:** Based on the results from the **Relevance Retriever Agent**, this agent decides whether to proceed with the core RAG process. If the results are promising, it activates the **Core Retriever Agent**. If not, it activates the **Web Search Agent**.

4.  **Web Search Agent:**
    * **Role:** The **external resource finder**.
    * **Process:** This agent is a fallback, activated only when the **Relevance Checking Agent** determines no suitable documents exist in the local database. Its purpose is to perform a targeted web search and provide the user with links to relevant external papers.

5.  **Core Retriever Agent:**
    * **Role:** The **deep search agent**.
    * **Process:** This agent performs the core retrieval task. It receives instructions from the **Reflexion Synthesis Agent** and performs a deep, targeted search within the specific document collections identified earlier to retrieve the most relevant chunks.

6.  **Reflexion Synthesis Agent:**
    * **Role:** The **iterative reasoner**.
    * **Process:** Using **LangGraph** to manage its state and cycles, this is the core of the system. It enters a ReAct-based loop to refine its answer.
        * **Reasoning:** The agent analyzes the user's query and identifies gaps in the retrieved context.
        * **Acting:** It calls the **Core Retriever Agent** as a tool to perform a targeted search.
        * **Observing:** It critically evaluates the new information.
        * **Reflecting:** If the answer isn't yet satisfactory, it adjusts its plan and initiates another reasoning/acting cycle.

7.  **Final Response:**
    * **Role:** Provides a complete, transparent answer.
    * **Process:** The Reflexion agent, once satisfied, synthesizes the final answer, including direct citations to the original document chunks and a list of all sources and links used.