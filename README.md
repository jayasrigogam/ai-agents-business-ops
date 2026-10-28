# Q5. What is RAG, and how can it improve business AI agents?

## Theory
**RAG (Retrieval-Augmented Generation)** gives an LLM the right information *at question time*.
Instead of relying on what the model memorised during training, the system first **retrieves**
relevant passages from the company's own documents, then asks the LLM to answer **using only
those passages**.

```
Offline (once):  Documents -> split into chunks -> index (vectors)
At question time: Question -> retrieve top chunks -> add to prompt -> LLM answers with citation
```

How RAG improves business agents:
1. **Up-to-date answers.** Update the document, re-index, done. No retraining.
2. **Company-specific knowledge.** Policies, manuals and contracts the LLM never saw.
3. **Fewer hallucinations.** The LLM is told to answer only from the retrieved text.
4. **Traceability.** Every answer can cite its source section.
5. **Safe fallback.** If nothing relevant is retrieved, the agent says so and escalates.
6. **Cost.** Far cheaper than fine-tuning a model every time a policy changes.

## Case study: leave policy from the latest HR handbook
An employee asks about leave. The agent must answer from the latest handbook:

| RAG step | In this project |
|---|---|
| 1. Load | `hr_handbook_2026.txt` (effective 1 April 2026) |
| 2. Chunk | Split by section, then into 2-sentence chunks that keep their section name |
| 3. Index | TF-IDF vectors for every chunk |
| 4. Retrieve | The question is vectorised; the top matching chunks are returned with scores |
| 5. Augment | Prompt = instructions + retrieved chunks + question |
| 6. Generate | LLM answers from the chunks only and cites the section |

## Demo
```bash
python q05_rag/demo.py
```
| Part | What it shows |
|---|---|
| 1 | LLM without RAG gives a generic, outdated answer (10 casual leaves) |
| 2 | Knowledge base built: load, chunk (11 chunks, 7 sections), index |
| 3 | 4 questions: casual leave, sick-leave certificate, maternity leave, and **stock options** (not in the handbook, so the agent escalates to HR instead of guessing) |
| 4 | HR changes the handbook (12 -> 14 casual leaves); only a re-index is needed and the answer changes immediately |

The index is plain TF-IDF so the demo needs no extra libraries. Production systems use
embeddings and a vector database; the pipeline is identical. With an `ANTHROPIC_API_KEY`
a real LLM writes the final answer from the retrieved chunks.
