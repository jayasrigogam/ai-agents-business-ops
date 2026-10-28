"""Q2 demo: The 5 components of an AI agent, shown with an HR agent.

  1. LLM             -> understands the question and writes the answer
  2. Memory          -> remembers who the employee is and the conversation so far
  3. Tools           -> get_employee_record (HR records)
  4. Knowledge base  -> HR handbook search
  5. Decision logic  -> routing rules + security guardrail + escalation

Run:  python q02_agent_components/demo.py
Set ANTHROPIC_API_KEY in .env to let a real LLM do the reasoning instead of the mock.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import has_api_key, run_tool_loop

HERE = os.path.dirname(os.path.abspath(__file__))
EMPLOYEES = json.load(open(os.path.join(HERE, "data", "employees.json")))
HANDBOOK = open(os.path.join(HERE, "data", "hr_handbook.txt")).read()
SECTIONS = [s.strip() for s in HANDBOOK.split("\n\n") if s.strip()]   # 1 section = 1 chunk
STOP = {"the", "a", "an", "is", "are", "what", "how", "do", "i", "my", "me", "of", "to", "for",
        "in", "on", "and", "can", "get", "does", "about", "tell", "please", "many", "much", "have",
        "policy", "policies", "company", "employee", "employees"}


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


# ============================================================ COMPONENT 2: MEMORY
class Memory:
    """Short-term memory: who the employee is + the conversation history."""
    def __init__(self):
        self.employee_id = None
        self.history = []          # list of (question, answer)

    def remember_turn(self, q, a):
        self.history.append((q, a))

    def summary(self):
        return " | ".join(f"Q: {q} A: {a[:60]}" for q, a in self.history[-3:]) or "(none)"


# ============================================================ COMPONENT 4: KNOWLEDGE BASE
def search_handbook(question: str):
    """Keyword retrieval over the HR handbook. Returns (best_section, score)."""
    q = words(question)
    best, best_score = None, 0
    for sec in SECTIONS:
        score = len(q & words(sec))
        if score > best_score:
            best, best_score = sec, score
    return best, best_score


# ============================================================ COMPONENT 3: TOOLS
def get_employee_record(employee_id: str):
    return EMPLOYEES.get(employee_id, {"error": f"No employee {employee_id}"})


# ============================================================ COMPONENT 1: LLM (mock stand-in)
def mock_llm_classify(message: str) -> str:
    """Stand-in for the LLM's intent understanding."""
    m = message.lower()
    if any(k in m for k in ("balance", "salary", "my record", "profile", "department", "left")):
        return "record_lookup"
    return "policy_question"


def mock_llm_compose(kind: str, data) -> str:
    """Stand-in for the LLM writing the final reply from retrieved facts."""
    if kind == "policy":
        return "According to the HR handbook: " + data.split("\n", 1)[1].strip()
    if kind == "record":
        return (f"{data['name']} ({data['department']}): {data['casual_leave_left']} casual leaves "
                f"and {data['sick_leave_left']} sick leaves left.")
    return data


# ============================================================ COMPONENT 5: DECISION LOGIC + AGENT
class HRAgent:
    def __init__(self, log=print):
        self.memory = Memory()
        self.log = log

    # ---- mock mode: every component announces itself
    def handle_mock(self, message: str) -> str:
        log, mem = self.log, self.memory

        # MEMORY: capture identity if the employee states it
        m = re.search(r"EMP-\d+", message)
        if m and re.search(r"i am|i'm|my id", message.lower()):
            mem.employee_id = m.group(0)
            log(f"  [MEMORY]   Stored employee_id = {mem.employee_id}")
        else:
            log(f"  [MEMORY]   Loaded employee_id = {mem.employee_id} | history = {len(mem.history)} turns")

        # LLM: understand the question
        intent = mock_llm_classify(message)
        log(f"  [LLM]      Intent understood: {intent}")

        # DECISION LOGIC: route
        if intent == "record_lookup":
            asked = re.findall(r"EMP-\d+", message)
            target = asked[0] if asked and not re.search(r"i am|i'm|my id", message.lower()) else mem.employee_id
            if target is None:
                log("  [DECISION] No employee identified -> ask for ID")
                return "Please tell me your employee ID first (for example: I am EMP-101)."
            if target != mem.employee_id:
                log(f"  [DECISION] GUARDRAIL: {target} is not the logged-in employee -> BLOCKED")
                answer = "Sorry, I can only share your own HR records."
            else:
                log("  [DECISION] Record question about own data -> use TOOL")
                log(f"  [TOOL]     get_employee_record('{target}')")
                rec = get_employee_record(target)
                log(f"  [TOOL]     returned {rec}")
                answer = mock_llm_compose("record", rec)
        else:
            log("  [DECISION] Policy question -> search KNOWLEDGE BASE")
            section, score = search_handbook(message)
            if score == 0:
                log("  [KB]       No relevant handbook section found")
                log("  [DECISION] Cannot answer reliably -> ESCALATE to HR team")
                answer = "I could not find this in the HR handbook, so I have forwarded it to the HR team."
            else:
                log(f"  [KB]       Best match: '{section.splitlines()[0]}' (score {score})")
                answer = mock_llm_compose("policy", section)

        log(f"  [LLM]      Final reply written")
        mem.remember_turn(message, answer)
        log(f"  [MEMORY]   Turn saved ({len(mem.history)} total)")
        return answer

    # ---- real LLM mode: LLM chooses between the two tools; guardrail still enforced in code
    def handle_real(self, message: str) -> str:
        mem = self.memory
        m = re.search(r"EMP-\d+", message)
        if m and re.search(r"i am|i'm|my id", message.lower()):
            mem.employee_id = m.group(0)

        def guarded_record(employee_id: str):
            if employee_id != mem.employee_id:
                return {"error": "BLOCKED: you can only access your own record"}
            return get_employee_record(employee_id)

        def kb(question: str):
            sec, score = search_handbook(question)
            return sec if score else "NOT FOUND - tell the user HR will follow up"

        tools = [
            {"name": "search_handbook", "description": "Search the HR handbook for policy information.",
             "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
            {"name": "get_employee_record", "description": "Get leave balance and details of an employee by ID.",
             "input_schema": {"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}},
        ]
        system = ("You are an HR assistant. Use search_handbook for policy questions and "
                  "get_employee_record for personal data. Answer only from tool results. "
                  "If the handbook has no answer, say HR will follow up.")
        user = (f"[Memory] Logged-in employee: {mem.employee_id}\n"
                f"[Memory] Recent conversation: {mem.summary()}\nEmployee says: {message}")
        answer = run_tool_loop(system, user, tools,
                               {"search_handbook": kb, "get_employee_record": guarded_record}, log=self.log)
        mem.remember_turn(message, answer)
        return answer

    def handle(self, message):
        return self.handle_real(message) if has_api_key() else self.handle_mock(message)


if __name__ == "__main__":
    conversation = [
        "Hi, I am EMP-101. How many casual leaves do I get?",
        "What is my leave balance?",
        "Show me the leave balance of EMP-102",
        "How do I claim travel reimbursement?",
        "What is the policy on parental sabbatical abroad?",
    ]
    print(f"HR AGENT DEMO  [{'REAL LLM' if has_api_key() else 'MOCK mode'}]")
    agent = HRAgent()
    for i, q in enumerate(conversation, 1):
        print(f"\n{'=' * 72}\nTURN {i} | EMPLOYEE: {q}\n{'-' * 72}")
        print("\nAGENT:", agent.handle(q))
