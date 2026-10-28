"""Q3 demo: The LLM as the reasoning engine of a sales agent.

For every customer request the agent must DECIDE between three actions:
    retrieve_customer  |  recommend_product  |  escalate
The LLM looks at the request + what it already knows, reasons about it, and picks the next action.
The agent code only executes the action; the DECISION comes from the reasoning step.

Run:  python q03_llm_reasoning/demo.py
Set ANTHROPIC_API_KEY in .env to let a real LLM do the reasoning instead of the rule-based mock.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import has_api_key, run_tool_loop

HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS = json.load(open(os.path.join(HERE, "data", "customers.json")))
PRODUCTS = json.load(open(os.path.join(HERE, "data", "products.json")))
ESCALATIONS = []


# ================================================================ THE THREE ACTIONS (tools)
def retrieve_customer(customer_id: str):
    return CUSTOMERS.get(customer_id, {"error": f"{customer_id} not found"})


def recommend_product(category: str, budget: int):
    """Best-rated product in the category within budget."""
    fits = [p for p in PRODUCTS if p["category"] == category and p["price"] <= budget]
    if not fits:
        return {"error": f"No {category} within budget {budget}"}
    return max(fits, key=lambda p: p["rating"])


def escalate(reason: str):
    ESCALATIONS.append(reason)
    return {"escalated_to": "human sales manager", "reason": reason}


# ================================================================ MOCK LLM REASONING
ANGRY = ("late", "refund", "cancel", "angry", "complaint", "worst", "legal", "lawyer", "terrible")
CATEGORIES = {"laptop": "laptop", "headphone": "audio", "earphone": "audio", "audio": "audio"}


def mock_reason(message: str, state: dict):
    """Stand-in for the LLM. Returns (reasoning, action, args, confidence).
    It reads the request AND the current state (what has already been retrieved)."""
    m = message.lower()
    cid = re.search(r"CUST-\d+", message)
    discount = re.search(r"(\d+)\s*%", m)
    qty = re.search(r"(\d+)\s*(laptops|units|pieces)", m)

    # Rule A: sensitive / out-of-authority situations -> escalate
    if any(w in m for w in ANGRY):
        return ("The customer sounds upset and is raising a complaint/refund issue. "
                "This needs empathy and authority I do not have.",
                "escalate", {"reason": "Customer complaint / refund demand"}, 0.95)
    if (discount and int(discount.group(1)) > 20) or (qty and int(qty.group(1)) >= 50):
        return ("The discount or order size is beyond my approval limit (max 20% discount, "
                "under 50 units). A manager must approve.",
                "escalate", {"reason": "Discount/bulk order above agent authority"}, 0.9)

    # Rule B: identified customer but profile not loaded yet -> retrieve
    if cid and state["customer"] is None:
        return (f"The customer gave an ID ({cid.group(0)}). Before recommending anything "
                "I should load their profile, budget and history.",
                "retrieve_customer", {"customer_id": cid.group(0)}, 0.9)

    # Rule C: have enough information -> recommend
    if state["customer"] is not None:
        cust = state["customer"]
        category = next((c for k, c in CATEGORIES.items() if k in m), None)
        source = "the request"
        if category is None and cust.get("preferred_category"):
            category, source = cust["preferred_category"], "their purchase history"
        budget_in_msg = re.search(r"(?:under|below|within)\s*(?:rs\.?|₹)?\s*(\d[\d,]*)", m)
        budget = int(budget_in_msg.group(1).replace(",", "")) if budget_in_msg else cust["budget"]
        if category:
            return (f"I know the customer's profile. Category '{category}' comes from {source}; "
                    f"budget is {budget}. I have enough information to recommend.",
                    "recommend_product", {"category": category, "budget": budget}, 0.85)

    return ("I do not have enough information to decide safely.",
            "escalate", {"reason": "Insufficient information"}, 0.4)


# ================================================================ AGENT LOOP (mock)
def run_mock(request: str):
    state = {"customer": None}
    for step in range(1, 5):
        reasoning, action, args, conf = mock_reason(request, state)
        print(f"  [Step {step}] LLM REASONING: {reasoning}")
        print(f"  [Step {step}] DECISION     : {action}  (confidence {conf:.0%})")
        if action == "retrieve_customer":
            out = retrieve_customer(**args)
            state["customer"] = out
            print(f"  [Step {step}] TOOL RESULT  : {out}")
            continue                      # reasoning continues with the new knowledge
        if action == "recommend_product":
            out = recommend_product(**args)
            print(f"  [Step {step}] TOOL RESULT  : {out}")
            return f"I recommend the {out['name']} (Rs {out['price']}, rated {out['rating']})."
        out = escalate(**args)
        print(f"  [Step {step}] TOOL RESULT  : {out}")
        return "I have passed your request to a human sales manager who will contact you shortly."
    return "Stopped after max steps."


def run_real(request: str):
    tools = [
        {"name": "retrieve_customer", "description": "Load a customer profile by ID.",
         "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
        {"name": "recommend_product", "description": "Recommend the best product for a category within a budget (category: laptop or audio).",
         "input_schema": {"type": "object", "properties": {"category": {"type": "string"}, "budget": {"type": "integer"}}, "required": ["category", "budget"]}},
        {"name": "escalate", "description": "Hand over to a human sales manager.",
         "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
    ]
    system = ("You are a sales agent. For each request decide ONE next action: retrieve_customer if a customer ID "
              "is given and the profile is not loaded; recommend_product when you know category and budget; "
              "escalate for complaints, refunds, discounts above 20%, orders of 50+ units, or missing information. "
              "Explain your reasoning briefly before each action.")
    return run_tool_loop(system, request, tools,
                         {"retrieve_customer": retrieve_customer, "recommend_product": recommend_product, "escalate": escalate})


if __name__ == "__main__":
    requests = [
        "Hi, I'm CUST-201. I need a laptop under 60000 for coding.",
        "Hello, CUST-202 here. Looking for something new, not sure what.",
        "This is the third time my order is late! I want a refund and I am cancelling.",
        "Can I get a 40% discount if I buy 200 laptops for my company?",
    ]
    print(f"SALES AGENT DEMO  [{'REAL LLM' if has_api_key() else 'MOCK reasoning'}]")
    for i, req in enumerate(requests, 1):
        print(f"\n{'=' * 78}\nREQUEST {i}: {req}\n{'-' * 78}")
        answer = run_real(req) if has_api_key() else run_mock(req)
        print(f"\n  AGENT REPLY: {answer}")
    print(f"\nEscalations raised: {len(ESCALATIONS)}")
