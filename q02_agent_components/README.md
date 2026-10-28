# Q2. What are the main components of an AI agent?

## Theory
An AI agent is built from five cooperating components:

| Component | Role | Business example |
|---|---|---|
| **LLM (reasoning engine)** | Understands language, interprets intent, plans, writes the final reply | Understands "what is my leave balance?" and composes the answer |
| **Memory** | Keeps context: who the user is and what was said earlier | Remembers the employee ID so it is not asked again |
| **Tools** | Functions that let the agent act on real systems (APIs, databases) | `get_employee_record` reads HR records |
| **Knowledge base** | Trusted company documents the agent looks up instead of guessing | HR handbook searched for leave and reimbursement rules |
| **Decision logic** | Rules and policies that route requests, enforce security, trigger escalation | Policy question -> knowledge base; other person's data -> blocked; no answer -> HR team |

## Case study: HR agent
| Component | Role in the HR agent |
|---|---|
| LLM | Classifies the employee's intent and writes a clear reply from retrieved facts |
| Memory | Stores the verified employee ID and the recent conversation |
| Tools | `get_employee_record` returns leave balance and details |
| Knowledge base | `hr_handbook.txt`: leave, attendance, reimbursement and dress code policies |
| Decision logic | Routes policy vs record questions; **blocks access to other employees' data**; escalates when the handbook has no answer |

## Demo
```bash
python q02_agent_components/demo.py
```
Five turns of one conversation, with every line tagged by the component acting:

| Turn | Employee asks | What happens |
|---|---|---|
| 1 | "I am EMP-101. How many casual leaves do I get?" | Memory stores the ID; KB returns the Leave Policy |
| 2 | "What is my leave balance?" | Memory supplies the ID (not asked again); the tool fetches the record |
| 3 | "Show me the leave balance of EMP-102" | Decision logic **blocks** it (privacy guardrail) |
| 4 | "How do I claim travel reimbursement?" | KB returns the Reimbursement Policy |
| 5 | "Policy on parental sabbatical abroad?" | Nothing in the KB, so it **escalates to HR** instead of guessing |

Note: the demo treats "I am EMP-101" as the login. A real system would authenticate the employee first.
