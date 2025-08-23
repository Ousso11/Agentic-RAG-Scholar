# Agentic-RAG Scholar

## Project Overview

**Agentic-RAG Scholar** is an advanced question-answering system built on **Retrieval-Augmented Generation (RAG)** and agent orchestration. It enables users to query scientific documents (PDFs, web links) and augment responses with web searches, producing accurate, evidence-backed, and traceable answers. The user interface is powered by **Gradio**.

The project uses **two main agents**:

1. **Retriever-based Reflexive Agent** – Iterates over documents using a hybrid retriever (BM25 + ChromaDB embeddings) to generate and refine answers.
2. **Web Search Reflexive Agent** – Uses web search tools (Google Scholar via SerpAPI or DuckDuckGo) to provide answers when documents are insufficient.

Both agents are reflexive, meaning they self-critique and iteratively refine their responses for accuracy.

Key libraries include **LangChain**, **LangGraph**, **ChromaDB**, and **Gradio**.

---

## Key Features

* Upload documents (PDFs or web links)
* Hybrid search combining vector search and keyword search (Retriever Agent)
* Web search fallback when documents are insufficient (Web Search Agent)
* Reflexive answer generation with iterative self-critique
* Evidence-backed answers with citations
* Simple and intuitive web interface (Gradio)
* Powered by **LangChain**, **LangGraph**, and **ChromaDB** for retrieval and orchestration

---

## Architecture and Workflow

### 1. Main Agents

#### a) Retriever-based Reflexive Agent

Uses a **hybrid retriever** (ChromaDB embeddings + BM25) to find relevant document chunks and generate answers iteratively with self-reflection.

#### b) Web Search Reflexive Agent

Uses **Google Scholar** or **DuckDuckGo** tools wrapped as a reflexive agent to answer questions when documents are not available.

### 2. Reflexive Synthesis

Each agent follows a reflexive process:

1. Generate an initial answer.
2. Critique and refine the response.
3. Propose follow-up queries if needed.
4. Stop when answer is satisfactory or maximum iterations reached.

### 3. Orchestration Workflow

The workflow (`agents/workflow.py`) works as follows:

1. Receive the user question.
2. Search uploaded documents with the Retriever Agent.
3. If needed, perform a web search with the Web Search Agent.
4. Use reflexive synthesis to generate the final answer.
5. Return the answer with citations and conversation trace.

---

## Installation

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) (for local LLMs)
* [Homebrew](https://brew.sh/) (for macOS)
* Python libraries: **langchain**, **langgraph**, **chromadb**, **gradio**, and others listed in `requirements.txt`

### 1. Clone the repository

```bash
git clone https://github.com/Ousso11/Agentic-RAG-Scholar.git
cd Agentic-RAG-Scholar
```

### 2. Setup the environment

```bash
./setup.sh
conda activate agentic-rag-env
```

---

## Running the Application

```bash
python main.py
# or
python3 main.py
```

The Gradio interface will be available at [http://localhost:7860](http://localhost:7860).

---

## Usage

1. Enter your question in the main input field.
2. Add documents (PDFs or web links).
3. Start the search.
4. Review the answer, citations, and conversation trace.

---

## Folder Structure

* `agents/` : contains all agents (retriever, web search, synthesis, etc.)
* `document_processing/` : document extraction and management
* `chroma_db/` : local vector database
* `main.py` : Gradio app entry point
* `ollama_engine.py` : Ollama integration and model management
* `config.py` : configuration for models and parameters
* `logger.py` : colored logger for debugging

---

## Customization

Change the models used in `config.py` (`MODEL_OPTIONS`, `DEFAULT_MODEL`, etc.) to experiment with different LLMs.

---

## Acknowledgements

This project is inspired by recent advances in **RAG**, **reflective agents**, and open-source LLMs. Thanks to the open-source community for making this work possible.
