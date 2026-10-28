"""Q5 - Retrieval-Augmented Generation (RAG) over the HR handbook.

Run from the repo root:   python q05_rag/demo.py
Optional custom question: python q05_rag/demo.py "How many sick days do I get?"

Pipeline:  load handbook -> chunk -> index (TF-IDF) -> retrieve top-k -> answer from context
Answer step: uses Claude if ANTHROPIC_API_KEY is set, otherwise a built-in extractive answer,
so the demo always runs.  Out-of-scope questions are refused (no hallucination).
"""
import os, re, sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
HANDBOOK = ROOT / "hr_handbook_2026.txt"
OUT_FILE = Path(__file__).resolve().parent / "output.txt"
TOP_K = 3
MIN_SCORE = 0.08          # below this -> "not in handbook"
LOG = []


def show(text=""):
    print(text)
    LOG.append(str(text))


# ---------------------------------------------------------------- 1. LOAD + CHUNK
def load_chunks(path, max_chars=500):
    """One chunk per paragraph (prefixed with its section heading); long paragraphs are split by sentence."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    chunks, heading = [], ""
    for p in paras:
        flat = " ".join(p.split())
        if len(flat) < 80 and not flat.endswith((".", "?", "!")):   # short line, no full stop = heading
            heading = flat
            continue
        prefix = f"{heading}: " if heading else ""
        if len(flat) <= max_chars:
            chunks.append(prefix + flat)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", flat):
            if buf and len(buf) + len(sent) > max_chars:
                chunks.append(prefix + buf); buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            chunks.append(prefix + buf)
    return chunks


# ---------------------------------------------------------------- 2. RETRIEVE
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(chunks)

    def search(self, query, k=TOP_K):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i]), self.chunks[i]) for i in order]


# ---------------------------------------------------------------- 3. GENERATE
def extractive_answer(query, hits):
    """Fallback: pick the sentence(s) from the best chunk that overlap most with the question."""
    qwords = set(re.findall(r"\w+", query.lower()))
    body = hits[0][2].split(": ", 1)[-1]          # drop the heading prefix
    sents = re.split(r"(?<=[.!?])\s+", body)
    ranked = sorted(sents, key=lambda s: len(qwords & set(re.findall(r"\w+", s.lower()))), reverse=True)
    return " ".join(ranked[:1])


def llm_answer(query, hits):
    import anthropic
    context = "\n\n".join(f"[Chunk {i}] {t}" for i, _, t in hits)
    msg = anthropic.Anthropic().messages.create(
        model="claude-sonnet-4-5", max_tokens=300,
        system="Answer ONLY from the provided HR handbook context. Cite chunk numbers like [Chunk 2]. "
               "If the answer is not in the context, say you cannot find it in the handbook.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])
    return msg.content[0].text


def answer(query, retriever):
    hits = retriever.search(query)
    show(f"\nQUESTION: {query}")
    show("-" * 70)
    show("Retrieved chunks:")
    for i, score, text in hits:
        show(f"  Chunk {i} | similarity {score:.2f}\n    {text[:200].replace(chr(10), ' ')}...")
    if hits[0][1] < MIN_SCORE:
        show("\nANSWER: I cannot find this in the HR handbook. (No relevant chunk - refusing to guess.)")
        return
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    try:
        ans = llm_answer(query, hits) if use_llm else extractive_answer(query, hits)
        mode = "Claude + retrieved context" if use_llm else "extractive (no API key)"
    except Exception as e:
        ans, mode = extractive_answer(query, hits), f"extractive (LLM error: {type(e).__name__})"
    show(f"\nANSWER [{mode}]:\n  {ans}")


# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    if not HANDBOOK.exists():
        sys.exit(f"Handbook not found at {HANDBOOK}. Run from the repo root.")
    chunks = load_chunks(HANDBOOK)
    show("=" * 70)
    show("Q5 - RAG DEMO: HR HANDBOOK ASSISTANT")
    show("=" * 70)
    show(f"Loaded {HANDBOOK.name}: {len(chunks)} chunks | retrieval: TF-IDF top-{TOP_K}")
    retriever = Retriever(chunks)

    questions = sys.argv[1:] or [
        "How many days of annual leave do employees get?",
        "What is the policy on overtime?",
        "Can I work from home?",
        "What is the company's stock option plan?",      # not in handbook -> should refuse
    ]
    for q in questions:
        answer(q, retriever)
    OUT_FILE.write_text("\n".join(LOG), encoding="utf-8")
    show(f"\n(Output saved to {OUT_FILE.relative_to(ROOT)})")
