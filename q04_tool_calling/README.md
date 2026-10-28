# Q4. What is tool / function calling, and why is it important for business agents?

## Theory
**Tool (function) calling** lets an LLM request that the program run a function on its behalf.
The LLM is shown a list of tools (name, description, parameters). When it needs something it
cannot do itself, it replies with *which tool to call and with what arguments*. The program
validates the call, runs it, and gives the result back to the LLM, which continues.

```
User request -> LLM chooses tool + arguments -> program validates & runs it
             -> result returned to LLM -> LLM calls another tool or writes the answer
```

Why it matters for business agents:
1. **Live, private data.** An LLM was trained on public text and knows nothing about the company's database.
2. **Accuracy.** Exact arithmetic and queries are done by code, not guessed by the model.
3. **Actions.** Agents can create tickets, send emails, update the CRM, not just talk.
4. **Control and safety.** Every call is validated and logged; bad arguments are rejected.
5. **Auditability.** Each figure in a report can be traced to a query.

## Case study: finance agent
The agent must retrieve sales figures from a database and generate a report. It needs tools
rather than the LLM's internal knowledge because that knowledge does not contain the company's
sales, is out of date, and an LLM may *invent* plausible numbers (hallucination).

| Tool | Purpose |
|---|---|
| `query_sales_db(region, month_from, month_to)` | Reads the real SQLite sales database (validates region and dates) |
| `calculate_totals(query_id)` | Exact totals, revenue by month and product, growth |
| `generate_report(query_id, title)` | Writes the formatted report file |

## Demo
```bash
python q04_tool_calling/demo.py
```
| Request | What happens |
|---|---|
| Report for the South region, 2026-07 to 2026-09 | query -> calculate -> generate; total revenue Rs 16,635,000 |
| Report for all regions, 2026-06 to 2026-09 | Same tools, different arguments; total Rs 48,983,000 |
| Report for the **Mars** region | Tool **rejects** the bad argument; the agent reports the error instead of inventing data |

Reports are written to `q04_tool_calling/output/`. With an `ANTHROPIC_API_KEY`, a real LLM
chooses the tools and arguments itself; the tools and validation are identical.
