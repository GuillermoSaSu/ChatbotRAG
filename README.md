# ChatbotRAG

A chatbot that answers questions about your own document (e.g. your CV in PDF),
using local semantic search + the Claude API. Read the file "How it works" to know the mathematical behaviour behind the program.

## Getting it running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   The first run will take a bit longer: it downloads a small embeddings model
   (~90MB) that runs on your own machine, at no cost and without sending
   anything to the internet for this step.

2. **Drop your CV (PDF)** into the `documentos/` folder (already created).
   The script automatically picks up the first PDF it finds there.

3. **Make sure** you have the `ANTHROPIC_API_KEY` environment variable set
   (same as in the previous project).

4. **Run it:**
   ```bash
   python rag_chatbot.py
   ```

5. Ask it things about your CV: "what Python experience do I have?",
   "summarize my career", "what did I study?"...

## What you'll learn from this code

- **PDF text extraction** (`pypdf`).
- **Chunking**: why and how a long document gets split into pieces.
- **Local embeddings**: turning text into meaning-vectors without using any
  external API (we use `sentence-transformers`, an open source library).
- **Semantic search**: finding the most relevant fragments using cosine
  similarity (mathematically, a dot product between normalized vectors).
- **Augmented prompt**: how the final message sent to Claude is built,
  combining the retrieved context + the original question.

## About your document's privacy

- **PDF processing and fragment retrieval happen 100% on your own machine**
  (the embeddings library is local, it doesn't send anything to a server
  for that step).
- The only thing that leaves your machine is: the 2-3 relevant fragments
  for each question you ask, sent to the Claude API to generate the answer
  (same as in the previous chatbot). The full PDF is never sent, only the
  chunks relevant to each specific question.

## Ideas to keep experimenting

- Try a smaller or bigger `CHUNK_SIZE` and see how answer quality changes.
- Bump `TOP_K` to 5 and compare whether answers actually improve or just get longer/pricier.
- Add several PDFs at once (e.g. your CV + a cover letter) and adapt the code to process all of them together.
- Print out which fragments got retrieved for each question (this is sometimes called "showing sources") — very useful for debugging why the bot answers what it answers.
