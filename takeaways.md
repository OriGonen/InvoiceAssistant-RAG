\## Step 0: Setup

\- The API key lives in an environment variable loaded from a git-ignored `.env` file; it never appears in code or git history.

\- The model name is configurable via `.env`, so switching models (e.g. cheap Flash-Lite vs stronger Flash) needs no code change.

\- I use Google's current `google-genai` SDK; the older `google-generativeai` package is legacy and lacks newer features.



\## Step 1: Sample data

\- Two data sources, one per tool: a SQLite `invoices` table for data questions and markdown rule documents for document questions.

\- Money is stored as integer cents, not floats, because floats can't represent decimal amounts exactly.

\- Dates are ISO strings (YYYY-MM-DD), so they sort and compare correctly as text.

\- CHECK constraints make the database reject invalid rows itself (negative amounts, "paid" with no payment date, due date before issue date).

\- Inserts use parameterised queries (`?` placeholders), so values can never be executed as SQL.

\- The data is hand-written and reproducible from a seed script, so every test question has a known correct answer.



\## Step 2: RAG retrieval

\- An embedding model turns text into a vector (384 numbers for MiniLM) so that texts with similar meaning have similar vectors.

\- Chunking is structure-first: one chunk per `##` section, with a size-based fallback (word windows with overlap) for long sections. Each piece keeps its section heading as metadata.

\- Chunk size is bounded by the embedding model: MiniLM silently truncates input beyond 256 word-pieces, so chunks stay well under that.

\- Each chunk is prefixed with its document title, so short chunks keep their context.

\- Embeddings are computed locally, so document text never leaves the machine. That matters for accounting data.

\- Documents and queries must be embedded with the same model; vectors from different models aren't comparable.

\- Chroma stores chunks with their vectors and finds the nearest ones by cosine distance (lower = more similar), using an HNSW index so it doesn't compare against every vector.

\- Retrieval always returns the top k, even for off-topic questions (weather: \~0.85 distance vs \~0.15–0.45 for real matches). So the LLM must be told to say when the retrieved text doesn't answer the question.

