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
