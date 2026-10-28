"""Q1 demo: Traditional chatbot vs AI agent.

Run:  python q01_agent_vs_chatbot/demo.py                 (all three orders, mock mode)
      python q01_agent_vs_chatbot/demo.py ORD-1002        (one order)
Set ANTHROPIC_API_KEY in .env to let a real LLM drive the agent instead of the mock planner.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import has_api_key, run_tool_loop

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS = json.load(open(os.path.join(HERE, "data", "orders.json")))
POLICY = open(os.path.join(HERE, "data", "refund_policy.txt")).read()
TICKETS, NOTIFICATIONS = [], []   # stand-ins for a ticketing system and an email service


# ---------------------------------------------------------------- 1) THE CHATBOT
FAQ = {
    "refund": "Our refund policy: physical items within 30 days of delivery. Please contact support.",
    "order": "You can track your order from the 'My Orders' page.",
}

def chatbot(message: str) -> str:
    """Only matches keywords and replies with canned text. It cannot DO anything."""
    for key, answer in FAQ.items():
        if key in message.lower():
            return answer
    return "Sorry, I didn't understand. Please contact support."


# ---------------------------------------------------------------- 2) THE AGENT'S TOOLS
def check_order(order_id: str):
    o = ORDERS.get(order_id)
    return o if o else {"error": f"Order {order_id} not found"}

def verify_refund_policy(order_id: str):
    o = ORDERS.get(order_id)
    if not o:
        return {"eligible": False, "reason": "Order not found"}
    if o["category"] == "digital":
        return {"eligible": False, "reason": "Digital products are non-refundable (policy rule 2)"}
    if o["status"] != "delivered":
        return {"eligible": False, "reason": "Order not delivered yet (policy rule 3)"}
    if o["delivered_days_ago"] > 30:
        return {"eligible": False, "reason": f"Delivered {o['delivered_days_ago']} days ago, outside 30-day window (policy rule 1)"}
    return {"eligible": True, "reason": "Within 30-day window (policy rule 1)"}

def create_ticket(order_id: str, summary: str):
    tid = f"TKT-{len(TICKETS) + 1:04d}"
    TICKETS.append({"id": tid, "order_id": order_id, "summary": summary})
    return {"ticket_id": tid, "status": "created"}

def notify_customer(email: str, message: str):
    NOTIFICATIONS.append({"to": email, "message": message})
    return {"sent_to": email}

TOOL_FUNCS = {"check_order": check_order, "verify_refund_policy": verify_refund_policy,
              "create_ticket": create_ticket, "notify_customer": notify_customer}

TOOL_SCHEMAS = [
    {"name": "check_order", "description": "Look up an order by ID.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "verify_refund_policy", "description": "Check if an order is eligible for refund under company policy.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "create_ticket", "description": "Create a refund ticket in the ticketing system (only if eligible).",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "summary": {"type": "string"}}, "required": ["order_id", "summary"]}},
    {"name": "notify_customer", "description": "Send the customer an email with the outcome.",
     "input_schema": {"type": "object", "properties": {"email": {"type": "string"}, "message": {"type": "string"}}, "required": ["email", "message"]}},
]

SYSTEM = ("You are a customer-service AI agent. Use tools to: check the order, verify the refund "
          "policy, create a ticket ONLY if eligible, and ALWAYS notify the customer of the outcome.\n\n"
          "Company policy:\n" + POLICY)


# ---------------------------------------------------------------- 3) MOCK PLANNER (no API key)
def mock_agent(request: str, log=print) -> str:
    """Rule-based stand-in for the LLM. Same loop shape: decide -> act -> observe -> decide."""
    m = re.search(r"ORD-\d+", request)
    order_id = m.group(0) if m else None
    log("[Step 1] THINK : I need the order ID, then I must check the order before anything else.")
    if not order_id:
        return "Could you please share your order ID?"

    log(f"[Step 2] ACT   : check_order({{'order_id': '{order_id}'}})")
    order = check_order(order_id); log(f"[Step 2] OBSERVE: {order}")
    if "error" in order:
        return order["error"]

    log("[Step 3] THINK : Order found. Next I must verify it against the refund policy.")
    log(f"[Step 3] ACT   : verify_refund_policy({{'order_id': '{order_id}'}})")
    verdict = verify_refund_policy(order_id); log(f"[Step 3] OBSERVE: {verdict}")

    if verdict["eligible"]:
        log("[Step 4] THINK : Eligible, so I should create a refund ticket.")
        log(f"[Step 4] ACT   : create_ticket(order_id='{order_id}')")
        t = create_ticket(order_id, f"Refund approved for {order['item']} ({order['amount']})")
        log(f"[Step 4] OBSERVE: {t}")
        msg = (f"Hi {order['customer']}, your refund for {order['item']} is approved. "
               f"Ticket {t['ticket_id']} has been created.")
    else:
        log("[Step 4] THINK : Not eligible, so NO ticket. I still must inform the customer why.")
        msg = f"Hi {order['customer']}, we cannot refund {order['item']}: {verdict['reason']}."

    log(f"[Step 5] ACT   : notify_customer(email='{order['email']}')")
    log(f"[Step 5] OBSERVE: {notify_customer(order['email'], msg)}")
    return msg


# ---------------------------------------------------------------- 4) RUN
def run(order_id: str):
    request = f"Hi, I want a refund for my order {order_id}."
    line = "=" * 70
    print(f"\n{line}\nCUSTOMER: {request}\n{line}")

    print("\n--- A) TRADITIONAL CHATBOT ---")
    print("Bot:", chatbot(request))
    print("(Nothing was checked, no ticket created, customer not notified.)")

    mode = "REAL LLM" if has_api_key() else "MOCK planner"
    print(f"\n--- B) AI AGENT [{mode}] ---")
    if has_api_key():
        final = run_tool_loop(SYSTEM, request, TOOL_SCHEMAS, TOOL_FUNCS)
    else:
        final = mock_agent(request)
    print("\nAgent final answer:", final)
    print(f"Tickets created: {len(TICKETS)} | Notifications sent: {len(NOTIFICATIONS)}")


if __name__ == "__main__":
    ids = sys.argv[1:] or list(ORDERS)
    for oid in ids:
        run(oid)
    print("\nSUMMARY of side effects:", json.dumps({"tickets": TICKETS, "notifications": NOTIFICATIONS}, indent=2))
