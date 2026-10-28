# AI Agents for Business Operations - Classroom Projects

Working demos for 14 theory + case-study questions. Each folder has a `README.md`
(theory answer + case study) and a runnable `demo.py` with sample data.

## Setup
```bash
git clone <your-repo-url> && cd ai-agents-business-ops
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
**Mock mode (default):** every demo runs offline with no API key.
**Real LLM mode (optional):** copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY`.

## Projects
| # | Topic | Folder | Status |
|---|---|---|---|
| 1 | AI agent vs chatbot | `q01_agent_vs_chatbot` | Done |
| 2 | Components of an AI agent | `q02_agent_components` | Done |
| 3 | LLM as reasoning engine | `q03_*` | Pending |
| 4 | Tool / function calling | `q04_*` | Pending |
| 5 | RAG | `q05_*` | Pending |
| 6 | Memory | `q06_*` | Pending |
| 7 | Planning & task decomposition | `q07_*` | Pending |
| 8 | Agentic workflow | `q08_*` | Pending |
| 9 | Customer service automation | `q09_*` | Pending |
| 10 | Marketing operations | `q10_*` | Pending |
| 11 | HR operations | `q11_*` | Pending |
| 12 | Recruitment | `q12_*` | Pending |
| 13 | Sales operations | `q13_*` | Pending |
| 14 | Inventory & supply chain | `q14_*` | Pending |
