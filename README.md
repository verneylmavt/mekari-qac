# Mekari Associate AI Engineer Challenge Test: Q&A Chatbot

This project implements a proof-of-concept AI vacation planner that can understand natural-language travel requests, interpret user preferences, check (mock) calendar availability, search for mock flights and hotels, and assemble a structured, day-by-day VacationPlan using an LLM-powered agent equipped with carefully defined tools. It also provides a dedicated booking endpoint that allows users to confirm a proposed itinerary once they explicitly approve it. The implementation combines FastAPI for the backend API, Pydantic v2 for data modeling and validation, PydanticAI for agent orchestration, OpenAI’s GPT-5-Nano for natural-language reasoning, and a lightweight in-memory storage layer to keep the proof-of-concept fully self-contained and easy to run.

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
   - Python
   - Conda or venv

1. Clone the repository:

   ```bash
    git clone https://github.com/verneylmavt/assistx-vp.git
    cd assistx-vp
   ```

2. Create environment and install dependencies:

   ```bash
   conda create -n assistx-vp python=3.11 -y
   conda activate assistx-vp

   pip install -r requirements.txt
   ```

3. Fill the required OpenAI API key in `.env`

4. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the API documentation to make an API call and interact with the app:
   ```bash
   start "http://127.0.0.1:8000/docs"
   ```
