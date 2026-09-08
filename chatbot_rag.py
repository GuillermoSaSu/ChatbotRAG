import os
import sys
import glob

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

MODEL_CLAUDE = "claude-sonnet-5"

#Local embeddings model. Small, fast
MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

DOCUMENT_FOLDER = "documents"

# Size of each text chunk, in characters. Smaller chunks
# mean more precise search but less context per chunk.
CHUNK_SIZE = 800

# characters repeated between consecutive fragments
SOLAPE_CHUNK = 150

#How many relevant fragments we feed to Claude per question.
TOP_K = 3

MAX_TOKENS = 1024

# -- First step: extract text from PDF --

def extract_text_pdf(pdf_path: str) -> str:
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# -- Second step: break the text into fragments --
def break_text(text: str, size: int, solape: int) -> list[str]:
    fragments = []
    start = 0
    while start < len(text):
        end = start + size
        fragments.append(text[start:end].strip())
        start += size - solape
    #Delete empty fragments or ones which are too short.
    return [f for f in fragments if len(f) > 30]

# -- Third step: prepare the embedding index --
def build_index(model_embeddings: SentenceTransformer, fragments: list[str]):
    """
    Convert each text fragment into a vector (embedding).
    We return the matrix of vectors so that searches can be performed on it.
    """
    vectors = model_embeddings.encode(fragments, normalize_embeddings=True)
    return np.array(vectors)

# -- Forth step: semantic search --
def search_relevant_fragments(
        question: str,
        model_embeddings: SentenceTransformer,
        fragments: list[str],
        vector_fragments: np.ndarray,
        top_k: int,
) -> list[str]:
    """
    Given a question, we convert it into a vector and calculate its similarity
    (dot product, since the vectors are normalized = cosine similarity)
    with each chunk. We return the top_k most similar ones.
    """
    vector_query = model_embeddings.encode([question], normalize_embeddings=True)[0]
    similarities = vector_fragments @ vector_query #dot product
    better_index = np.argsort(similarities)[::-1][:top_k]
    return [fragments[i] for i in better_index]


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set the environment variable ANTHROPIC_API_KEY")
        sys.exit(1)

    pdf_paths = glob.glob(os.path.join(DOCUMENT_FOLDER, "*.pdf"))
    if not pdf_paths:
        print("No PDFs found")
        sys.exit(1)

    print(f"📄 Process: {pdf_paths[0]}")
    text = extract_text_pdf(pdf_paths[0])

    print("✂️  Fragment the document...")
    fragments = break_text(text, CHUNK_SIZE, SOLAPE_CHUNK)
    print(f" {len(fragments)} fragments generated.")

    print("🧠 Loading local embeddings model (takes longer only the first time)...")
    model_embeddings = SentenceTransformer(MODEL_EMBEDDING)

    print("🔢 Generating embeddings for the fragments...")
    vector_fragments = build_index(model_embeddings, fragments)

    client = Anthropic()
    history = []

    print("\n🤖 Ready. Ask me anything about the document. Type 'exit' to finish.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("See you!")
            break
        if question.lower() == "exit":
            print("See you!")
            break
        if not question:
            continue

        # 1. Search relevant fragments for this question.
        relevent_fragments = search_relevant_fragments(
            question, model_embeddings, fragments, vector_fragments, TOP_K
        )

        # 2. We build an "increased" prompt: we provide the relevant context
        #  BEFORE the question, so that Claude answers based on it.
        context = "\n\n---\n\n".join(relevent_fragments)
        increased_message = (
            f"Context extracted from a file (use only this information to "
            f"answer; if the answer is not here, state that clearly):\n\n"
            f"{context}\n\n---\n\nQuestion: {question}"
        )

        history.append({"role": "user", "content": increased_message})

        print("Claude: ", end="", flush=True)
        complete_response = ""
        with client.messages.stream(
            model = MODEL_CLAUDE,
            max_tokens = MAX_TOKENS,
            system = ("You are an assistant that answers questions based "
                "SOLELY on the context provided in each "
                "message. Do not invent information that is not in the context."
            ),
            messages = history,
        ) as stream:
            for text_out in stream.text_stream:
                print(text_out, end="", flush=True)
                complete_response += text_out

        print("\n")
        history.append({"role": "assistant", "content": complete_response})

if __name__ == "__main__":
    main()
