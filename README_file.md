<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

# D&D Rules RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** pipeline using LangChain 1.0 to answer questions about D&D rules from markdown documents processed via olmOCR.

## Features Covered

*   **Document Indexing**: Load markdown files, split into chunks, and store embeddings in Qdrant.
*   **RAG Agent**: Flexible multi-step queries using `create_agent` with a retrieval tool.
*   **RAG Chain**: Fast single-call Q&A using middleware for context injection.
*   **LangSmith Tracing**: Built-in observability for debugging and monitoring.

## Pre-Reqs

You'll need to ensure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) before proceeding.

```bash
# Install dependencies
uv sync
```

You will also need an **OpenAI API Key** (`OPENAI_API_KEY`) and optionally a **LangSmith API Key** (`LANGSMITH_API_KEY`) for tracing.

## Preparing Documents with olmOCR

Before running the RAG pipeline, convert your PDF rulebooks to markdown using olmOCR:

```bash
# Pull the Docker image (large, includes the model, ~30GB)
docker pull alleninstituteforai/olmocr:latest-with-model

# Convert PDFs to markdown
docker run --gpus all \
  -v "$(pwd)":/workspace \
  alleninstituteforai/olmocr:latest-with-model \
  -c "python -m olmocr.pipeline /workspace/output --markdown --pdfs /workspace/PDFs/*.pdf"
```

Place the generated markdown files in the `PDFs/` directory.

## Running the Example

1.  **Launch the Notebook**:
    ```bash
    uv run jupyter notebook RAG_Pipeline.ipynb
    ```

2.  **Run the Cells**:
    *   Set up your environment and API keys.
    *   Index your D&D rulebook markdown files into Qdrant.
    *   Test the RAG Agent for flexible, multi-step queries.
    *   Test the RAG Chain for fast, single-call Q&A.
    *   Use the interactive demo to ask your own questions!

## Project Structure

*   `RAG_Pipeline.ipynb`: The main interactive RAG pipeline notebook.
*   `PDFs/`: Directory containing markdown files (converted from PDFs via olmOCR).
*   `pyproject.toml`: Dependency management.

## Key Components

| Component | Purpose |
|-----------|---------|
| `DirectoryLoader` | Load markdown files from disk |
| `RecursiveCharacterTextSplitter` | Split documents into retrievable chunks |
| `QdrantVectorStore` | Store and search embeddings |
| `@tool` decorator | Create retrieval tool for agent |
| `create_agent` | Build LangChain 1.0 agent |
| `AgentMiddleware` | Inject context for RAG chain |

## Credits

Built using [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), and [olmOCR](https://github.com/allenai/olmocr).
