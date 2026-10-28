# Q5 - RAG (Retrieval-Augmented Generation)

**Problem:** An LLM does not know a company's private HR policies and may hallucinate.
**Solution (RAG):** retrieve the relevant handbook passages first, then answer only from them.

Pipeline: hr_handbook_2026.txt -> chunking -> TF-IDF index -> top-3 retrieval -> grounded answer.
Out-of-scope questions are refused (similarity below threshold).

Run (from repo root):
    pip install scikit-learn anthropic
    python q05_rag/demo.py
Optional LLM answers:  export ANTHROPIC_API_KEY="your-key"
