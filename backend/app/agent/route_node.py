# backend/app/agent/router.py

from .state import AgentState
from ..llm.openai_client import call_gpt5_nano, call_gpt5_mini

ROUTER_SYSTEM_PROMPT = (
    "You are a router for an internal fraud-analytics assistant.\n"
    "This assistant has access to exactly two internal knowledge sources:\n\n"
    "1) data: tabular credit-card transaction data\n"
    "   This data contains credit-card transactions where each row represents a single purchase "
    "   made by a cardholder and includes detailed information about the transaction time, amount, "
    "   merchant, category, customer demographics, home location, merchant location, and a binary "
    "   label indicating whether the transaction is fraudulent.\n\n"
    "2) document: a conceptual white paper called \"Understanding Credit Card Frauds\"\n"
    "   This document contains a 2003-era white paper that explains the growing problem of credit "
    "   card fraud, detailing how fraud is committed (including application fraud, lost/stolen and "
    "   counterfeit cards, skimming, merchant collusion, triangulation schemes, and internet-based "
    "   attacks such as site cloning, fake merchant sites, and card number generators), "
    "   quantifying global and country-specific loss trends, and analyzing the impact on cardholders "
    "   (limited liability), merchants (full liability, chargebacks, fees, admin overhead, and "
    "   reputation damage), and banks (direct losses plus high prevention and operational costs). "
    "   It reviews both basic and advanced fraud prevention methods—manual review, Address "
    "   Verification System, card verification codes, negative/positive lists, payer authentication "
    "   (e.g., Verified by Visa), lockout mechanisms, and blacklists of fraudulent merchants—then "
    "   describes more sophisticated techniques such as rule-based systems, statistical risk "
    "   scoring, neural networks, biometrics, and smart card (EMV) technology. The paper’s central "
    "   thesis is that effective fraud management is about minimizing the \"total cost of fraud\"—the "
    "   sum of actual fraud losses and the cost of prevention—by using these tools to segment and "
    "   prioritize transactions so that only the riskiest subset is subject to intensive review, "
    "   thereby achieving an optimal balance between security, cost, and customer experience.\n\n"
    "Your task:\n"
    "- Choose 'data' if the question is primarily about analyzing the tabular credit-card "
    "  transaction data (e.g., fraud rates, time trends, top merchants/categories, transaction "
    "  patterns, customer or merchant-level statistics).\n"
    "- Choose 'document' if the question is primarily about fraud concepts, mechanisms, definitions, "
    "  or the authors' opinions as described in the white paper.\n"
    "- Choose 'none' if the question is clearly unrelated to both the transaction dataset and the "
    "  document, or if it cannot reasonably be answered using these two sources.\n\n"
    "Return exactly one word: data, document, or none."
)


def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    user_prompt = f"Question: {question}\n\nAnswer with exactly one word: data, document, or none."

    raw = call_gpt5_nano(
        ROUTER_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=4,
    )
    print("Router (by GPT-5 Nano) is Called")
    route = raw.strip().lower()

    if route not in ("data", "document", "none"):
        # Fallback heuristic if the model returns something unexpected
        q_lower = question.lower()

        # Heuristic for data questions
        if any(
            word in q_lower
            for word in [
                "rate",
                "trend",
                "daily",
                "monthly",
                "time series",
                "merchant",
                "category",
                "transaction",
                "amount",
                "volume",
                "count",
                "share",
                "proportion",
                "distribution",
            ]
        ):
            route = "data"
        # Heuristic for document questions
        elif any(
            word in q_lower
            for word in [
                "application fraud",
                "lost/stolen",
                "lost or stolen",
                "counterfeit",
                "skimming",
                "merchant collusion",
                "triangulation",
                "internet",
                "site cloning",
                "neural network",
                "fraud detection system",
                "total cost of fraud",
                "prevention",
                "white paper",
            ]
        ):
            route = "document"
        else:
            route = "none"

    state["route"] = route  # type: ignore
    return state


FALLBACK_SYSTEM_PROMPT = (
    "You are an assistant for an internal fraud-analytics tool. This tool is limited to:\n"
    "- A specific credit-card transaction dataset (tabular data with transactions and fraud labels).\n"
    "- A specific conceptual document: \"Understanding Credit Card Frauds\" (a 2003-era white paper).\n\n"
    "If the user's question is outside the scope of these two resources, you must NOT pretend to "
    "answer it using unrelated knowledge. Instead, explain clearly that this particular chatbot "
    "is specialized for that dataset and document only, and that the question appears to be "
    "outside its domain.\n\n"
    "Be brief and honest. Do NOT mention any internal routing logic."
)


def fallback_answer_node(state: AgentState) -> AgentState:
    question = state["question"]

    user_prompt = (
        f"User question:\n{question}\n\n"
        "Explain that this tool is limited to the fraud dataset and the fraud document described "
        "in the system prompt, and that the question does not seem to be answerable within that scope."
    )

    answer = call_gpt5_mini(
        system_prompt=FALLBACK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=256,
    )

    state["answer"] = answer  # type: ignore
    state["answer_type"] = "other"  # type: ignore
    return state