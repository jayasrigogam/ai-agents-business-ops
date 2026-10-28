# Q3. How does an LLM act as the reasoning engine of an AI agent?

## Theory
In an AI agent the LLM is the "brain". It does not just produce text: it **reads the situation,
reasons about it, and chooses the next action**. The surrounding code (the "body") only executes
that action by calling a tool and returns the result to the LLM, which reasons again.

The LLM supports decisions by:
1. **Understanding intent** from natural language (a complaint, a purchase need, a bulk request).
2. **Assessing what it knows** and what is missing (is the customer profile loaded yet?).
3. **Choosing an action** from the available tools, with a reason and a confidence.
4. **Reading the result** and deciding whether to continue, finish or hand over to a human.
5. **Applying policy** it is given in its instructions (discount limit, escalation rules).

## Case study: sales agent
The agent must decide among three actions:

| Situation in the request | LLM's reasoning | Decision |
|---|---|---|
| Customer ID given, profile not loaded | I need budget and history before recommending | `retrieve_customer` |
| Need and budget are known | I have enough information | `recommend_product` |
| Complaint, refund demand, discount above 20%, 50+ units, or too little information | Beyond my authority or unsafe to guess | `escalate` |

## Demo
```bash
python q03_llm_reasoning/demo.py
```
| Request | Reasoning path |
|---|---|
| "I'm CUST-201. I need a laptop under 60000" | retrieve profile -> recommend CodeBook 15 |
| "CUST-202 here. Something new, not sure what" | retrieve profile -> infer *audio* from purchase history -> recommend StudioPods Pro |
| "Third time my order is late! Refund, I'm cancelling" | **escalate** (complaint) |
| "40% discount if I buy 200 laptops?" | **escalate** (above authority) |

Each step prints `LLM REASONING`, `DECISION` (with confidence) and `TOOL RESULT`, so the
decision logic is visible. The mock mode uses rules to stand in for the LLM; with an
`ANTHROPIC_API_KEY` a real LLM produces the reasoning and choices.
