# Q1. What is an AI agent, and how is it different from a traditional chatbot?

## Theory
A **traditional chatbot** is a conversational interface that matches the user's message to
scripted or retrieved answers. It talks, but it does not act: it has no access to company
systems and no ability to pursue a goal across several steps.

An **AI agent** is a system that uses an LLM as its reasoning engine to pursue a goal. It
decides what to do next, calls **tools** (databases, APIs, ticketing and email systems),
observes the results, and repeats until the task is complete (the *perceive -> reason -> act
-> observe* loop).

| Feature | Chatbot | AI Agent |
|---|---|---|
| Goal | Answer a question | Complete a task |
| Actions | None (text reply only) | Calls tools / changes systems |
| Steps | Single turn | Multi-step, plans and adapts |
| Data access | Static FAQ | Live data (orders, CRM, policies) |
| Autonomy | Low | Decides its own next step |

## Case study: why the refund system is an AI agent
The new system checks orders, verifies refund policy, creates tickets and notifies customers.
It is an agent because it:
1. **Has a goal** (resolve the refund request), not just a question to answer.
2. **Uses tools** (`check_order`, `verify_refund_policy`, `create_ticket`, `notify_customer`).
3. **Reasons and decides** which step comes next and whether to create a ticket at all.
4. **Acts on external systems**, producing real side effects (a ticket, an email).
5. **Observes results and adapts**: an ineligible order gets a rejection and no ticket.

## Demo
```bash
python q01_agent_vs_chatbot/demo.py              # all 3 sample orders
python q01_agent_vs_chatbot/demo.py ORD-1002     # a single order
```
| Order | Situation | Agent behaviour |
|---|---|---|
| ORD-1001 | Physical, delivered 5 days ago | Approves, creates ticket, notifies |
| ORD-1002 | Physical, delivered 45 days ago | Rejects (30-day window), no ticket, notifies |
| ORD-1003 | Digital e-book | Rejects (non-refundable), no ticket, notifies |

The chatbot gives the same canned reply for all three and changes nothing.
