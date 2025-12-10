# Mekari Associate AI Engineer Challenge Test: Q&A Chatbot

This projects implements a Q&A Chatbot, which focuses on building a robust internal system capable of answering fraud-related questions using two fundamentally different sources of information: a tabular credit-card transaction dataset and a domain document explaining real-world fraud mechanisms. The primary challenge is to design an intelligent agent that can understand a user’s question, determine the appropriate knowledge source, extract and synthesize correct information, and deliver clear, accurate insights. Rather than relying on a single model prompt, this system integrates multi-stage reasoning, retrieval-augmented generation, LLM-generated SQL analytics, and dynamic routing to handle the full spectrum of analytical and conceptual questions presented in the challenge.

At its core, this project is engineered as a production-ready, modular pipeline that separates concerns cleanly: data preprocessing, database analytics, vector-store retrieval, LLM orchestration, and a lightweight frontend for interaction. A FastAPI backend hosts a LangGraph-based agent capable of routing questions to either a Data Analysis Agent (using SQL over a star-schema PostgreSQL warehouse) or a Document RAG Agent (using Qdrant, BGE embeddings, and a reranker). This architecture ensures extensibility, interpretability, and efficiency. A Streamlit interface wraps the full system into an accessible UI, complete with backend health checks, conversation history, transparency over SQL queries, and a quality-scoring mechanism that evaluates each answer for reliability and correctness.

[Click here to learn more about the project: assistx-vp/assets/Task - AI Engineer (LLM) Revised.pdf](<https://github.com/verneylmavt/assistx-vp/blob/fa8ebaab0b877b795af87f442eb632d78826cb3b/assets/Task%20-%20AI%20Engineer%20(LLM)%20Revised.pdf>).

## 📁 Project Structure

```
assistx-vp
│
├─ app/                              # Solution app
│  ├─ config.py                      # Configuration files
│  ├─ main.py                        # FastAPI app
│  │
│  ├─ agent/
│  │  └─ vacation_agent.py           # PydanticAI agent with tools
│  │
│  ├─ models/
│  │  ├─ api.py                      # API models
│  │  └─ domain.py                   # Domain models
│  │
│  ├─ services/
│  │  ├─ bookings.py                 # Booking service
│  │  ├─ calendar.py                 # Calendar service
│  │  ├─ preferences.py              # Preferences service
│  │  ├─ sessions.py                 # Session helper
│  │  └─ travel_search.py            # Travel search service (for mock flights/hotels)
│  │
│  └─ storage/
│     └─ in_memory.py                # In-memory storage
│
├─ assets/
│  ├─ vacation_planner_solution.pdf  # Solution report
│  └─ vacation_planner_demo.gif      # Solution demo video
│
├─ .env
└─ requirements.txt
```

## 💡 Solution Report and Solution Demo Video

- The solution report includes overview, solution, and vulnerability and risk. [Click here to learn more about the solution report: assistx-vp/assets/vacation_planner_solution.pdf](https://github.com/verneylmavt/assistx-vp/blob/392517c31b3a6190a7c442b79437368a83ac4b44/assets/vacation_planner_solution.pdf).
- The solution demo video shows the working app, accessible via api call. [Click here to learn more about the solution demo video: assistx-vp/assets/vacation_planner_demo.gif](https://github.com/verneylmavt/assistx-vp/blob/392517c31b3a6190a7c442b79437368a83ac4b44/assets/vacation_planner_demo.gif).

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
   - Run the FastAPI server:
     ```bash
     uvicorn backend.app.main:app --reload --port 8000
     ```
   - Run the Streamlit UI:
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
