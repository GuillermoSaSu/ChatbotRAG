# RAG chatbot — how it works

A small chatbot that answers questions about a PDF document using local
embeddings + semantic search, then asks Claude to answer grounded in the
retrieved context.

## Pipeline

1. **Extract** — pull raw text out of the PDF (`pypdf`).
2. **Chunk** — split the text into overlapping fragments (`CHUNK_SIZE=800`
   chars, `SOLAPE_CHUNK=150` chars of overlap so ideas aren't cut mid-sentence).
3. **Embed** — turn each fragment into a vector using a local model
   (`paraphrase-multilingual-MiniLM-L12-v2`, runs on CPU, no API call).
4. **Retrieve** — embed the user's question, compare it against every
   fragment vector, keep the `TOP_K` most similar ones.
5. **Generate** — send the retrieved fragments + the question to Claude,
   instructed to answer only from that context.

## The math: how similarity is measured

Each fragment (and the question) is represented as a vector **v** in a
high-dimensional space (384 dimensions, for this model). Two texts with
similar meaning end up as vectors pointing in similar directions.

**Dot product** between two vectors *a* and *b*:

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**Cosine similarity** normalizes that by the length (magnitude) of each
vector, so it only measures *direction*, not size:

```
             a · b
cos(θ) = ───────────
           ‖a‖ · ‖b‖
```

- `‖a‖` is the magnitude (length) of vector *a*: `√(a₁² + a₂² + ... + aₙ²)`

**The shortcut used in the code:** if every vector is pre-normalized to
length 1 (`‖a‖ = ‖b‖ = 1`), the formula above collapses to just the dot
product:

```
cos(θ) = a · b     (when ‖a‖ = ‖b‖ = 1)
```

That's exactly what `normalize_embeddings=True` does when generating
embeddings — it lets the code skip the division and just use `@` (matrix
multiplication / dot product) directly:

```python
similarities = vector_fragments @ vector_query
```

## Reading the score

| cos(θ) | Meaning |
|---|---|
| ≈ 1 | Same direction → very similar meaning |
| ≈ 0 | Perpendicular → unrelated |
| ≈ -1 | Opposite direction → opposite meaning (rare with real text) |

In practice, relevant fragments typically score **0.5–0.8**, unrelated ones
**0.1–0.3**. The code doesn't use a fixed cutoff — it just takes the
`TOP_K` highest scores, whatever they are.

## Notes

- Chunking, embeddings, and retrieval all run **locally** — nothing leaves
  the machine except the 2–3 retrieved fragments sent to the Claude API
  per question.
- `SOLAPE_CHUNK` must always be smaller than `CHUNK_SIZE` — otherwise the
  sliding window moves backward instead of forward and never finishes.
