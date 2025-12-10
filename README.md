# Mekari Associate AI Engineer Challenge Test: Q&A Chatbot

This projects implements a Q&A Chatbot, which focuses on building a robust internal system capable of answering fraud-related questions using two fundamentally different sources of information: [a tabular credit-card transaction dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraud%20dataset) and [a document explaining real-world fraud mechanisms](https://popcenter.asu.edu/sites/g/files/litvpz3631/files/problems/credit_card_fraud/PDFs/Bhatla.pdf). The primary challenge is to design an intelligent agent that can understand a user’s question, determine the appropriate knowledge source, extract and synthesize correct information, and deliver clear, accurate insights.

At its core, this project is engineered as a modular pipeline that separates concerns cleanly: data processing, PostgreSQL relational database for transaction dataset, Qdrant vector database for document embedding, FastAPI backend for LLM orchestration, and Streamlit frontend for interaction.

[Click here to learn more about the project: mekari-qac/assets/Mekari - AI Engineer.pdf](https://github.com/verneylmavt/mekari-qac/blob/ec7788fa0749925197eb3379c2ed9b6e56e4d5f2/assets/Mekari%20-%20AI%20Engineer.pdf).

## 📁 Project Structure

```
mekari-qac
│
├── data/
│   ├── fraudData/
│   │   ├── fraudTrain.csv
│   │   ├── fraudTest.csv
│   │   ├── fraudData_snapshot.dump
│   │   └── data_processing_fraudData.ipynb
│   │
│   └── Understanding Credit Card Frauds/
│       ├── Bhatla.pdf
│       ├── Bhatla.docx
│       ├── Bhatla_Description.docx
│       ├── Bhatla_chunks.json
│       ├── Bhatla_embeddings.npy
│       ├── data_processing_Understanding Credit Card Frauds.ipynb
│       └── qdrant/              # Local Qdrant storage (if using volume mapping)
│
├── backend/
│   └── app/
│       ├── main.py              # FastAPI entrypoint (/health, /chat)
│       ├── config.py            # Pydantic settings (env-based)
│       ├── db.py                # SQLAlchemy engine
│       ├── schemas.py           # Pydantic request/response models
│       │
│       ├── agent/
│       │   ├── state.py         # Agent state TypedDict
│       │   ├── router.py        # LLM router (data / document / none)
│       │   ├── graph.py         # LangGraph wiring of all nodes
│       │   ├── data_nodes.py    # LLM-to-SQL, SQL execution, data answer
│       │   ├── doc_nodes.py     # Qdrant retrieval + RAG answer
│       │   └── scoring_node.py  # Answer quality scoring
│       │
│       ├── llm/
│       │   └── client.py        # Thin wrapper around OpenAI (GPT-5-Nano / GPT-5-Mini)
│       │
│       ├── rag/
│       │   └── qdrant_client.py # BGE embedder + Qdrant search & rerank
│       │
│       └── repositories/
│           └── metrics_repo.py  # Thin SQL execution layer for analytics
│
└── frontend/
    └── streamlit_app.py        # Streamlit chat interface

│
├── assets/
│   └── Mekari - AI Engineer.*   # Challenge description (PDF/DOCX)
│
├── .env
└── requirements.txt
```

## 🧩 Components

## 🔌 API

1. **Health Check**
   - `GET /health`: to verify that the backend is running correctly, responding to requests, and using the expected LLM configuration
     - Request: `None`
     - Response: `'status', 'model'`
     ```bash
     curl "http://localhost:8000/health"
     ```
2. **User Preferences**
   - `GET /api/preferences/{user_id}`: to retrieve the user’s saved travel preferences or automatically initialize defaults if none exist
     - Request: `user_id`
     - Response: `PreferencesResponse`
     ```bash
     curl "http://localhost:8000/api/preferences/{user_id}"
     ```
   - `PUT /api/preferences/{user_id}`: to update the user’s travel preferences with new budgets, origins, interests, or other settings
     - Request: `PreferencesUpdateRequest`
     - Response: `PreferencesResponse`
     ```bash
     curl -X PUT "http://localhost:8000/api/preferences/{user_id}" \
     -H "Content-Type: application/json" \
     -d '{
        "home_city": "{home_city}",
        "default_currency": "{default_currency}",
        "max_budget_total": {max_budget_total},
        "max_budget_per_day": {max_budget_per_day},
        "interests": ["{interest_1}", "{interest_i}", "{interest_n}"],
        "travel_style": "{travel_style}",
        "preferred_airlines": ["{preferred_airline_1}", "{preferred_airline_i}", "{preferred_airline_n}"],
        "preferred_hotel_types": ["{preferred_hotel_type_1}", "{preferred_hotel_type_i}", "{preferred_hotel_type_n}"]
     }'
     ```
3. **Chat w/ LLM**

   - `POST /api/chat`: to send a natural-language message to the AI vacation-planning agent and receive a reply, potentially including a generated VacationPlan
     - Request: `ChatRequest`
     - Response: `ChatResponse`
     ```bash
     curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{
        "session_id": "{session_id}",
        "user_id": "{user_id}",
        "message": "{message_to_llm}",
        "allow_booking": {bool}
     }'
     ```

4. **Book Plan**
   - `POST /api/book`: to confirm and record a booking for the latest AI-generated vacation plan within the user’s session, using a provided payment token
     - Request: `BookRequest`
     - Response: `BookResponse`
     ```bash
     curl -X POST "http://localhost:8000/api/book" \
     -H "Content-Type: application/json" \
     -d '{
        "session_id": "{session_id}",
        "user_id": "{user_id}",
        "payment_token": "{payment_token}"
     }'
     ```

## 🧩 Demo Video

- FastAPI Server
  ![FastAPI Server](https://raw.githubusercontent.com/verneylmavt/synapsis-pd-pt-pc/refs/heads/main/assets/Dashboard.gif)

- Streamlit UI
  ![Streamlit UI](https://raw.githubusercontent.com/verneylmavt/synapsis-pd-pt-pc/refs/heads/main/assets/Dashboard.gif)

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
