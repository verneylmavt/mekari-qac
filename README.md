# Mekari Associate AI Engineer Challenge Test: Q&A Chatbot

This projects implements a Q&A Chatbot, which focuses on building a robust internal system capable of answering fraud-related questions using two fundamentally different sources of information: [a tabular credit-card transaction dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraud%20dataset) and [a document explaining real-world fraud mechanisms](https://popcenter.asu.edu/sites/g/files/litvpz3631/files/problems/credit_card_fraud/PDFs/Bhatla.pdf). The primary challenge is to design an intelligent agent that can understand a user’s question, determine the appropriate knowledge source, extract and synthesize correct information, and deliver clear, accurate insights.

At its core, this project is engineered as a modular pipeline that separates concerns cleanly: data processing, PostgreSQL relational database for transaction dataset, Qdrant vector database for document embedding, FastAPI backend for LLM orchestration, and Streamlit frontend for interaction.

[Click here to learn more about the project: mekari-qac/assets/Mekari - AI Engineer.pdf](https://github.com/verneylmavt/mekari-qac/blob/ec7788fa0749925197eb3379c2ed9b6e56e4d5f2/assets/Mekari%20-%20AI%20Engineer.pdf).

## 📁 Project Structure

```
mekari-qac
│
├── data/                                     # Dataset and data processing
│   ├── fraudData/
│   │   ├── fraudTrain.csv                    # Training split of credit card transaction dataset
│   │   ├── fraudTest.csv                     # Test split of credit card transaction dataset
│   │   ├── data_processing_fraudData.ipynb   # Data processing notebook for credit card transaction dataset
│   │   ├── fraudData_snapshot.dump           # DB snapshot
│   │   └── requirements.txt
│   │
│   └── Understanding Credit Card Frauds/
│       ├── Bhatla.pdf                                              # Credit card fraud document
│       ├── data_processing_Understanding Credit Card Frauds.ipynb  # Data processing notebook for credit card fraud document
│       ├── Bhatla_chunks.json                                      # Cleaned and segmented text chunks
│       ├── Bhatla_embeddings.npy                                   # Precomputed dense embeddings
│       └── requirements.txt
│
├── backend/                                  # FastAPI backend
│   ├── app/
│   │   ├── main.py                           # REST API: /health, /chat
│   │   ├── config.py                         # Environment variables + global configuration
│   │   ├── db.py                             # PostgreSQL engine creation + connection handling
│   │   ├── schemas.py                        # Pydantic request/response models
│   │   │
│   │   ├── agent/
│   │   │   ├── state.py                      # Central AgentState + shared memory fields
│   │   │   ├── graph.py                      # Routing graph: data, document, fallback, scoring
│   │   │   ├── router.py                     # LLM question router: data vs document vs none
│   │   │   ├── data_nodes.py                 # SQL generator, SQL executor, and data explanation nodes
│   │   │   ├── doc_nodes.py                  # Qdrant retrieval + RAG answer generator
│   │   │   └── scoring_node.py               # Quality-scoring node for evaluating LLM answers
│   │   │
│   │   ├── llm/
│   │   │   └── client.py                     # GPT-5-Nano/Mini wrappers for chat/completions
│   │   │
│   │   ├── rag/
│   │   │   └── qdrant_client.py              # Embedding, retrieval, reranking + Qdrant connection
│   │   │
│   │   └── repositories/
│   │       └── metrics_repo.py               # SQL execution helper for querying analytics tables/views
│   │
│   └── requirements.txt
│
├── frontend/                                 # Streamlit frontend
│   ├── app.py                                # Streamlit interface: health check, chat UI
│   └── requirements.txt
│
├── scripts/                                  # Initialization scripts
│   ├── init_postgresql.py                    # Script to initialize PostgreSQL
│   └── init_qdrant.py                        # Script to initialize Qdrant
│
├── assets/
│   ├── q&a_chatbot_fastapi_demo.mp4          # Demo video for FastAPI Server
│   └── q&a_chatbot_streamlit_demo.mp4        # Demo video for Streamlit UI
│
├── .env
└── requirements.txt
```

## 🧩 Components

- **PostgreSQL Relational Database for Credit Card Transaction Dataset**  
   - **Implementation**  
      The credit card transaction data from `data/fraudData/fraudTrain.csv` and `data/fraudData/fraudTest.csv` is first combined and processed inside `data/fraudData/data_processing_fraudData.ipynb`, where it undergoes extensive normalization, parsing, and feature engineering. This includes converting timestamps and identifiers into proper formats, creating calendar fields (year, month, year-month, day-of-week, hour), computing customer age at the time of each transaction, and calculating the customer-to-merchant distance using the Haversine formula. The notebook then models the dataset as a full analytical star schema: `dim_customer,` `dim_merchant`, `dim_category`, `dim_date`, and `fact_transactions` and loads it into a PostgreSQL database. Indexes and materialized views are created to support fast analytical queries for fraud rates, merchant/category breakdowns, and time-series patterns.

      After preprocessing, the notebook exports the fully populated database into `data/fraudData/fraudData_snapshot.dump`, which captures all tables, indexes, and materialized views. At runtime, `scripts/init_postgresql.py restores` this snapshot into a Dockerized PostgreSQL instance, ensuring that the backend starts with a ready-to-query analytical warehouse. This allows the FastAPI service to immediately access all aggregated fraud metrics through a clean relational model without reprocessing the raw CSVs.

   - **Future Improvements**  
      - Introduce Database Migrations: Replace snapshot-only initialization with migrations that can introduce indexes, partitions, and optimized datatypes incrementally without full reloads, improving iteration speed and avoiding unnecessary downtime.
      - Columnar Storage or Compression Extensions: Consider PostgreSQL extensions such as TimescaleDB, Citus, or columnar storage (e.g., cstore_fdw or zheap) to accelerate analytical workloads and reduce I/O.
      - Vectorized and Parallel Query Optimization: Tune PostgreSQL for parallel query execution, adjust work_mem and shared_buffers for large joins, and analyze expected query patterns to ensure the planner uses indexes and parallel workers effectively.


- **Qdrant Vector Database for Credit Card Fraud Document**
  - **Implementation**  
      The credit card fraud document from `data/Understanding Credit Card Frauds/Bhatla.docx` is transformed into a searchable vector corpus inside the notebook `data/Understanding Credit Card Frauds/data_processing_Understanding Credit Card Frauds.ipynb`. The document is parsed into structured sections and paragraphs, split into `Blocks`, and then segmented into sentence-based `Chunks` with controlled length and minimal overlap. Each chunk is assigned a UUID and exported to `data/Understanding Credit Card Frauds/Bhatla_chunks.json`, while the BGE embedding model (`BAAI/bge-base-en-v1.5`) encodes every chunk into a 768-dimensional vector saved in `data/Understanding Credit Card Frauds/Bhatla_embeddings.npy`. A reranker model (`BAAI/bge-reranker-base`) is also initialized for improved relevance scoring during retrieval.

      To build the vector store, the script `scripts/init_qdrant.py` launches a Qdrant instance, recreates the target collection (`bhatla_credit_fraud`) with cosine similarity, and uploads all chunks in batches with their corresponding metadata and embeddings. At runtime, the FastAPI backend uses this populated collection for dense retrieval and reranking, enabling grounded, document-based answers within the chatbot’s RAG pipeline.

   - **Future Improvements**  
      - Persistent Qdrant Storage Configuration: Ensure durable storage via Docker volumes or mounted paths so embeddings and payloads remain loaded between restarts, preventing costly full re-index operations.
      - Payload-Aware Filtering for Faster Retrieval: Use metadata filters (e.g., section tags, topic tags, fraud categories) to shrink candidate sets before dense scoring, reducing retrieval latency and reranker load.
      - Query Preprocessing and Hybrid Retrieval: Apply lightweight query rewriting (synonym expansion, acronym resolution) and hybrid search (lexical + vector) to improve recall while reducing the reranker’s workload on irrelevant candidates.

- **FastAPI Backend Server**
  - **Implementation**  
      The FastAPI backend exposes two primary endpoints: `/health `for liveness checks and `/chat` for serving Q&A responses. `backend/app/main.py` handles request routing, loads configuration via `backend/app/config.py`, manages CORS for local development, and performs connectivity checks against PostgreSQL (through a cached SQLAlchemy engine in `backend/app/db.py`) and Qdrant (through the shared client in `backend/app/rag/qdrant_client`.py). When a chat request arrives, the backend converts the conversation history into a minimal `{role, content}` form and invokes `run_agent()` from `backend/app/agent/graph.py`, later packaging the agent’s final answer, metadata, and source previews into a strongly typed `ChatResponse` defined in `backend/app/schemas.py`.

      Runtime configuration and external integrations are encapsulated cleanly. `backend/app/config.py `centralizes all environment-driven settings such as DB credentials, Qdrant location, and LLM model names; `backend/app/db.py` constructs and caches the SQLAlchemy engine; and `backend/app/rag/qdrant_client.py` loads the embedding and reranker models once at startup, providing convenient helpers for query embedding, dense search, and reranking. This design keeps model loading, Qdrant access, and connection handling isolated from the core logic.

      The conversational intelligence is implemented as a LangGraph state machine wired in `backend/app/agent/graph.py`. It orchestrates the end-to-end flow: the router (`backend/app/agent/router.py`) classifies each question as data-focused, document-focused, or out-of-scope; the data path (`backend/app/agent/data_nodes.py`) generates SQL, executes it with `backend/app/repositories/metrics_repo.py.run_sql_query`, and summarizes the results; the document path (`backend/app/agent/doc_nodes.py`) retrieves and reranks relevant Qdrant chunks from `backend/app/rag/qdrant_client.py.run_sql_query` before generating a grounded RAG answer; and the fallback route produces a safe message for unsupported queries. All paths conclude with the `backend/app/agent/scoring_node.py`, which computes an LLM-based quality score based on the answer and its evidence.

      LLM calls and data access are abstracted behind stable interfaces. `backend/app/llm/client.py` provides thin wrappers around GPT-5 Nano and Mini, ensuring that every part of the pipeline (router, SQL generator, RAG answerer, scorer) uses consistent model invocation logic. Together, these components form a cohesive backend that retrieves the right information source, synthesizes grounded answers, scores them, and then exposes everything through a simple and predictable `/chat` API.

   - **Future Improvements**  
      - Batch and Asynchronous Execution: Move to async FastAPI endpoints and async database + LLM clients, enabling concurrency scaling and significantly improving throughput under parallel user queries.
      - Agent Graph Optimization and Caching: Cache routing decisions, SQL snippets, and Qdrant retrieval results for repeated or similar queries to reduce redundant LLM calls and improve overall response latency.

- **Streamlit Frontend UI**
  - **Implementation**  
      The Streamlit interface in `frontend/app.py` provides a simple chat surface that communicates with the FastAPI backend over HTTP. It initializes the backend URL from the `FRAUD_API_BASE_URL` environment variable, exposes this setting in the sidebar, and allows users to run a `/health` check that reports the status of PostgreSQL, Qdrant, and the active LLM model. Conversation state is stored in `st.session_state.messages`, which holds a list of user and assistant turns. Each user input is immediately rendered, while the sidebar and helper utilities (`init_session_state`, `build_history_for_backend`, `call_health`) maintain a consistent UI state.

      When the user submits a question, the UI sends a POST request to `/chat` using `call_chat()`, passing the cleaned conversation history in the format required by the backend. The backend’s response, containing the generated answer, answer type, quality score, optional SQL, and supporting sources, is appended as an assistant message and displayed using `render_assistant_message()`. This renderer supports expandable previews for SQL result samples and retrieved document chunks, ensuring transparency in how each answer was generated. Errors from the backend are caught and displayed as assistant messages so that the chat view remains stable even under failure conditions.
   - **Future Improvements**  
      - Streaming Responses: Support incremental token streaming from the backend so the UI stays responsive during long LLM generations and provides faster perceived latency.
      - Asynchronous Backend Requests: Use async HTTP clients (e.g., httpx) and background tasks so the UI remains interactive while waiting for long-running backend computations.

## 🔌 API

1. **Health Check**  
   `GET /health`: to verify that the FastAPI server, PostgreSQL, Qdrant is running
   - Request: `None`
   - Response: `'status', 'db_ok', 'qdrant_ok', 'model'`
   ```bash
   curl "http://localhost:8000/health"
   ```
2. **Chat w/ Fraud Q&A Chatbot**  
   `POST /chat`: to ask the chatbot about credit card transaction or credit card fraud
   - Request: `ChatRequest`
   - Response: `ChatResponse`
   ```bash
   curl -X PUT "http://localhost:8000/chat" \
   -H "Content-Type: application/json" \
   -d '{
      "question": "{question}",
      "history": [
            {"role": "user", "content": "{user_content}"},
            {"role": "assistant", "content": "{assistant_content}"}
        ]
   }'
   ```

## 🖥️ Demo Video

- **FastAPI Server**
  ![FastAPI Server](https://media.githubusercontent.com/media/verneylmavt/mekari-qac/refs/heads/main/assets/q%26a_chatbot_fastapi_demo.gif)

- **Streamlit UI**
  ![Streamlit UI](https://media.githubusercontent.com/media/verneylmavt/mekari-qac/refs/heads/main/assets/q%26a_chatbot_streamlit_demo.gif)

## ⚙️ Local Setup

0. Make sure to have the prerequisites:

   - Git
   - Git Large File Storage
   - Python
   - Conda or venv
   - Docker
   - NVIDIA Driver + CUDA Toolkit (optional)

1. Clone the repository:

   ```bash
    git clone https://github.com/verneylmavt/mekari-qac.git
    cd mekari-qac
   ```

2. Create environment and install dependencies:

   ```bash
   conda create -n mekari-qac python=3.11 -y
   conda activate mekari-qac

   pip install -r requirements.txt
   ```

3. Fill the required `OPENAI_API_KEY` in `.env`

4. Initialize and run the required components:

   - Initialize the PostgreSQL:
     ```bash
     python scripts/init_postgresql.py
     ```
   - Initialize the Qdrant:
     ```bash
     python scripts/init_qdrant.py
     ```
   - Run the FastAPI backend server:
     ```bash
     uvicorn backend.app.main:app --reload --port 8000
     ```
   - Run the Streamlit frontend UI:
     ```bash
     streamlit run frontend/app.py
     ```

5. Open the API documentation to make an API call:
   ```bash
   start "http://127.0.0.1:8000/docs"
   ```
   Or alternatively, open the UI and interact with the app:
   ```bash
   start "http://127.0.0.1:8501"
   ```
