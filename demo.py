"""Q5 demo: Retrieval-Augmented Generation (RAG) for an HR agent.

Problem: an LLM knows only what it was trained on. Company policies change, and are private.
RAG fixes this by fetching the relevant text from the CURRENT handbook at question time and
giving it to the LLM as context. The LLM then answers from that text, with a citation.

Pipeline (each step is printed):
    1. LOAD the handbook          2. CHUNK it into small pieces      3. INDEX the chunks (TF-IDF vectors)
    4. RETRIEVE the top chunks    5. AUGMENT the prompt with them     6. GENERATE a cited answer

The index uses plain TF-IDF written from scratch (no extra libraries), so the demo runs anywhere.
Production systems use embeddings + a vector database; the pipeline is the same.

Run:  python q05_rag/demo.py
Set ANTHROPIC_API_KEY in .env to let a real LLM write the final answer from the retrieved chunks.
"""
import math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm import has_api_key

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = {"the", "a", "an", "is", "are", "what", "how", "do", "i", "my", "me", "of", "to", "for", "in", "on",
        "and", "can", "get", "does", "about", "tell", "please", "many", "much", "have", "it", "be", "or",
        "policy", "company", "employee", "every", "per", "year", "with", "at", "by", "that", "than", "this", "them"}
MIN_SCORE = 0.12          # below this, the handbook has no relevant answer


def tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)     # naive stemming: leaves -> leave
    return out


# ------------------------------------------------------------ STEP 2: CHUNK
def chunk_handbook(text):
    """Split by section heading, then into chunks of up to 2 sentences. Keep section as metadata."""
    chunks, section, cid = [], None, 0
    for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
        lines = block.splitlines()
        if lines[0].isupper():                                    # a section heading
            section, body = lines[0].title(), " ".join(lines[1:])
        else:
            section, body = "Document header", block
        sentences = re.split(r"(?<=\.)\s+", body.strip())
        for i in range(0, len(sentences), 2):
            cid += 1
            chunks.append({"id": f"C{cid}", "section": section, "text": " ".join(sentences[i:i + 2])})
    return chunks


# ------------------------------------------------------------ STEP 3: INDEX (TF-IDF)
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        docs = [tokens(c["section"] + " " + c["text"]) for c in chunks]     # section title helps retrieval
        n = len(docs)
        df = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}
        self.vectors = [self._vec(d) for d in docs]

    def _vec(self, toks):
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        v = {t: c * self.idf.get(t, 0) for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    # ---------------------------------------------------- STEP 4: RETRIEVE
    def search(self, question, k=3):
        q = self._vec(tokens(question))
        scored = [(sum(w * v.get(t, 0) for t, w in q.items()), c) for v, c in zip(self.vectors, self.chunks)]
        return sorted(scored, key=lambda x: -x[0])[:k]


# ------------------------------------------------------------ STEPS 5-6: AUGMENT + GENERATE
def build_prompt(question, hits):
    ctx = "\n".join(f"[{c['id']} | {c['section']}] {c['text']}" for _, c in hits)
    return ("Answer the employee's question using ONLY the handbook excerpts below. "
            "Cite the section. If the answer is not in the excerpts, say you could not find it.\n\n"
            f"HANDBOOK EXCERPTS:\n{ctx}\n\nQUESTION: {question}")


def generate(question, hits):
    if has_api_key():
        import anthropic
        resp = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), max_tokens=400,
            messages=[{"role": "user", "content": build_prompt(question, hits)}])
        return "".join(b.text for b in resp.content if b.type == "text")
    # MOCK generator: answers strictly from the retrieved chunks and cites them
    top = hits[0][0]
    used = [c for s, c in hits if s >= max(MIN_SCORE, 0.5 * top)][:2]
    used.sort(key=lambda c: int(c['id'][1:]))          # keep handbook order for readability
    return " ".join(c["text"] for c in used) + "  [Source: " + ", ".join(f"{c['section']} ({c['id']})" for c in used) + "]"


def rag_answer(index, question, show_prompt=False):
    hits = index.search(question)
    print("  [RETRIEVE] Top matching chunks:")
    for s, c in hits:
        print(f"      score {s:.2f}  {c['id']} [{c['section']}] {c['text'][:70]}...")
    if hits[0][0] < MIN_SCORE:
        print(f"  [DECIDE  ] Best score {hits[0][0]:.2f} < {MIN_SCORE}: nothing relevant in the handbook")
        return "I could not find this in the HR handbook. I have forwarded your question to the HR team."
    print("  [AUGMENT ] Prompt = instructions + retrieved chunks + question")
    if show_prompt:
        print("      " + build_prompt(question, hits[:2]).replace("\n", "\n      "))
    print(f"  [GENERATE] Answer written from the retrieved text only [{'real LLM' if has_api_key() else 'mock'}]")
    return generate(question, hits)


if __name__ == "__main__":
    handbook = open(os.path.join(HERE, "data", "hr_handbook_2026.txt")).read()
    print(f"RAG DEMO - HR AGENT  [{'REAL LLM' if has_api_key() else 'MOCK generator'}]")

    print("\n" + "=" * 78 + "\nPART 1: THE PROBLEM - an LLM without RAG\n" + "-" * 78)
    print("  Q: How many casual leaves do I get?")
    print("  LLM only (trained on an older, public policy): 'Companies typically give 10 casual leaves.'")
    print("  -> Outdated and generic. It cannot know this company's 2026 handbook says 12.")

    print("\n" + "=" * 78 + "\nPART 2: BUILD THE KNOWLEDGE BASE (done once, before any question)\n" + "-" * 78)
    print(f"  [LOAD ] Handbook loaded: {len(handbook.split())} words, first line: '{handbook.splitlines()[0]}'")
    chunks = chunk_handbook(handbook)
    sections = sorted({c['section'] for c in chunks})
    print(f"  [CHUNK] Split into {len(chunks)} chunks across {len(sections)} sections: {', '.join(sections)}")
    index = Index(chunks)
    print(f"  [INDEX] Built TF-IDF index: {len(index.idf)} distinct terms")

    print("\n" + "=" * 78 + "\nPART 3: ASK QUESTIONS (RAG pipeline runs for each)\n" + "-" * 78)
    questions = [
        "How many casual leaves do I get and can I carry them forward?",
        "Do I need a medical certificate for sick leave?",
        "How long is maternity leave?",
        "What is the policy on stock options?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\nQUESTION {i}: {q}")
        print("\n  ANSWER:", rag_answer(index, q, show_prompt=(i == 1)))

    print("\n" + "=" * 78 + "\nPART 4: THE HANDBOOK CHANGES - no retraining needed\n" + "-" * 78)
    updated = handbook.replace("12 casual leaves", "14 casual leaves")
    print("  HR edits the handbook: 12 casual leaves -> 14 casual leaves. We only RE-INDEX (seconds).")
    index2 = Index(chunk_handbook(updated))
    q = "How many casual leaves do I get?"
    print(f"\nQUESTION: {q}")
    print("\n  ANSWER:", rag_answer(index2, q))
