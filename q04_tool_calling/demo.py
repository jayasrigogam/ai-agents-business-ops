"""Q4 demo: Tool / function calling for a finance agent.

An LLM only knows its training data. It has never seen the company's sales database.
So the agent is given TOOLS and the LLM decides which tool to call, with which arguments:

    query_sales_db(region, month_from, month_to)  -> reads the real database
    calculate_totals(query_id)                    -> exact arithmetic (LLMs are unreliable at maths)
    generate_report(query_id, title)              -> writes the report file

The program validates every call, runs the tool, and feeds the result back to the LLM.

Run:  python q04_tool_calling/demo.py
Set ANTHROPIC_API_KEY in .env to let a real LLM choose the tools instead of the mock planner.
"""
import csv, json, os, re, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import has_api_key, run_tool_loop

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ the company database (SQLite)
DB = sqlite3.connect(":memory:")
DB.execute("CREATE TABLE sales (month TEXT, region TEXT, product TEXT, units INT, revenue INT)")
with open(os.path.join(HERE, "data", "sales.csv")) as f:
    DB.executemany("INSERT INTO sales VALUES (?,?,?,?,?)",
                   [(r["month"], r["region"], r["product"], int(r["units"]), int(r["revenue"]))
                    for r in csv.DictReader(f)])
VALID_REGIONS = {r[0] for r in DB.execute("SELECT DISTINCT region FROM sales")}
RESULTS = {}   # query_id -> rows, so later tools can refer to earlier results


# ------------------------------------------------------------------ TOOL SCHEMAS (what the LLM is shown)
TOOL_SCHEMAS = [
    {"name": "query_sales_db",
     "description": "Retrieve sales rows from the company database. region is optional (North or South); "
                    "month_from and month_to are in YYYY-MM format. Returns a query_id.",
     "input_schema": {"type": "object",
                      "properties": {"region": {"type": "string"}, "month_from": {"type": "string"}, "month_to": {"type": "string"}},
                      "required": ["month_from", "month_to"]}},
    {"name": "calculate_totals",
     "description": "Compute total revenue, units, revenue by product and month, and growth for a query_id.",
     "input_schema": {"type": "object", "properties": {"query_id": {"type": "string"}}, "required": ["query_id"]}},
    {"name": "generate_report",
     "description": "Write a formatted sales report file for a query_id and return its path.",
     "input_schema": {"type": "object", "properties": {"query_id": {"type": "string"}, "title": {"type": "string"}}, "required": ["query_id", "title"]}},
]


# ------------------------------------------------------------------ TOOL IMPLEMENTATIONS (validated)
def query_sales_db(month_from: str, month_to: str, region: str = None):
    for label, val in (("month_from", month_from), ("month_to", month_to)):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(val)):
            return {"error": f"Invalid {label} '{val}'. Use YYYY-MM."}
    if region is not None and region not in VALID_REGIONS:
        return {"error": f"Unknown region '{region}'. Valid regions: {sorted(VALID_REGIONS)}"}
    sql = "SELECT month, region, product, units, revenue FROM sales WHERE month BETWEEN ? AND ?"
    params = [month_from, month_to]
    if region:
        sql += " AND region = ?"; params.append(region)
    rows = DB.execute(sql, params).fetchall()
    if not rows:
        return {"error": "No data for that period"}
    qid = f"Q{len(RESULTS) + 1}"
    RESULTS[qid] = {"rows": rows, "region": region or "All regions", "from": month_from, "to": month_to}
    return {"query_id": qid, "row_count": len(rows)}


def calculate_totals(query_id: str):
    if query_id not in RESULTS:
        return {"error": f"Unknown query_id {query_id}"}
    rows = RESULTS[query_id]["rows"]
    by_month, by_product = {}, {}
    for month, _, product, units, revenue in rows:
        by_month[month] = by_month.get(month, 0) + revenue
        by_product[product] = by_product.get(product, 0) + revenue
    months = sorted(by_month)
    growth = None
    if len(months) >= 2 and by_month[months[0]]:
        growth = round((by_month[months[-1]] - by_month[months[0]]) / by_month[months[0]] * 100, 1)
    return {"total_revenue": sum(by_month.values()), "total_units": sum(r[3] for r in rows),
            "revenue_by_month": {m: by_month[m] for m in months},
            "revenue_by_product": dict(sorted(by_product.items(), key=lambda kv: -kv[1])),
            "growth_first_to_last_month_pct": growth}


def generate_report(query_id: str, title: str):
    if query_id not in RESULTS:
        return {"error": f"Unknown query_id {query_id}"}
    meta, t = RESULTS[query_id], calculate_totals(query_id)
    lines = [title, "=" * len(title),
             f"Region : {meta['region']}", f"Period : {meta['from']} to {meta['to']}", "",
             f"Total revenue : Rs {t['total_revenue']:,}", f"Total units   : {t['total_units']:,}", "",
             "Revenue by month:"] + [f"  {m}: Rs {v:,}" for m, v in t["revenue_by_month"].items()]
    lines += ["", "Revenue by product:"] + [f"  {p}: Rs {v:,}" for p, v in t["revenue_by_product"].items()]
    if t["growth_first_to_last_month_pct"] is not None:
        lines += ["", f"Growth (first to last month): {t['growth_first_to_last_month_pct']}%"]
    best = next(iter(t["revenue_by_product"]))
    lines += [f"Top product: {best}", "", "Source: company sales database (queried live)."]
    path = os.path.join(OUT_DIR, f"report_{query_id}.txt")
    open(path, "w").write("\n".join(lines))
    return {"report_path": os.path.relpath(path, os.path.dirname(HERE)), "preview": "\n".join(lines)}


TOOL_FUNCS = {"query_sales_db": query_sales_db, "calculate_totals": calculate_totals, "generate_report": generate_report}


# ------------------------------------------------------------------ MOCK PLANNER
def mock_agent(request: str):
    """Stand-in for the LLM: reads the request and chooses tools + arguments."""
    region = next((r for r in re.findall(r"\b([A-Z][a-z]+)\s+region", request)), None)
    months = re.findall(r"\d{4}-\d{2}", request)
    print(f"  [THINK] I need real figures from the database. I must not guess numbers.")
    args = {"month_from": months[0], "month_to": months[1]}
    if region:
        args["region"] = region
    print(f"  [CALL ] query_sales_db({args})")
    out = query_sales_db(**args); print(f"  [RESULT] {out}")
    if "error" in out:
        print("  [THINK] The tool rejected my arguments. I will report the problem instead of inventing data.")
        return f"I could not retrieve the data: {out['error']}"
    print(f"  [THINK] Data retrieved. Totals must be computed exactly, so I call calculate_totals.")
    print(f"  [CALL ] calculate_totals(query_id='{out['query_id']}')")
    totals = calculate_totals(out["query_id"]); print(f"  [RESULT] {totals}")
    title = f"Sales Report - {args.get('region', 'All regions')} ({args['month_from']} to {args['month_to']})"
    print(f"  [THINK] Now generate the formatted report.")
    print(f"  [CALL ] generate_report(query_id='{out['query_id']}', title='{title}')")
    rep = generate_report(out["query_id"], title)
    print(f"  [RESULT] report saved to {rep['report_path']}")
    print("\n  --- REPORT PREVIEW ---\n" + "\n".join("  " + l for l in rep["preview"].splitlines()))
    return f"Report generated: {rep['report_path']}"


def real_agent(request: str):
    system = ("You are a finance agent. You have no sales data in memory. Use query_sales_db to get data, "
              "calculate_totals for figures, and generate_report to produce the report. Never invent numbers. "
              "If a tool returns an error, explain it.")
    return run_tool_loop(system, request, TOOL_SCHEMAS, TOOL_FUNCS)


if __name__ == "__main__":
    print(f"FINANCE AGENT DEMO  [{'REAL LLM' if has_api_key() else 'MOCK planner'}]")

    print("\n" + "=" * 78)
    print("WHY TOOLS?  An LLM alone cannot see your database.")
    print("-" * 78)
    print("  Request : 'What were South region sales from 2026-07 to 2026-09?'")
    print("  LLM only: has no access to company data. It can only refuse, or worse, GUESS")
    print("            a plausible-looking number (a hallucination).")
    print("  With tools: the LLM chooses query_sales_db, and the answer comes from real data.")
    print("\n  Tools the LLM is given:")
    for t in TOOL_SCHEMAS:
        print(f"    - {t['name']}: {t['description'][:70]}...")

    requests = [
        "Prepare a sales report for the South region from 2026-07 to 2026-09.",
        "Prepare a sales report for all regions from 2026-06 to 2026-09.",
        "Prepare a sales report for the Mars region from 2026-07 to 2026-09.",
    ]
    for i, req in enumerate(requests, 1):
        print(f"\n{'=' * 78}\nREQUEST {i}: {req}\n{'-' * 78}")
        answer = real_agent(req) if has_api_key() else mock_agent(req)
        print(f"\n  AGENT REPLY: {answer}")
