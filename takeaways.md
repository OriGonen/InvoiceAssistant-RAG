## Step 0: Setup
- The API key lives in an environment variable loaded from a git-ignored `.env` file; it never appears in code or git history.
- The model name is configurable via `.env`, so switching models (e.g. cheap Flash-Lite vs stronger Flash) needs no code change.
- I use Google's current `google-genai` SDK; the older `google-generativeai` package is legacy and lacks newer features.

## Step 1: Sample data
- Two data sources, one per tool: a SQLite `invoices` table for data questions and markdown rule documents for document questions.
- Money is stored as integer cents, not floats, because floats can't represent decimal amounts exactly.
- Dates are ISO strings (YYYY-MM-DD), so they sort and compare correctly as text.
- CHECK constraints make the database reject invalid rows itself (negative amounts, "paid" with no payment date, due date before issue date).
- Inserts use parameterised queries (`?` placeholders), so values can never be executed as SQL.
- The data is hand-written and reproducible from a seed script, so every test question has a known correct answer.

## Step 2: RAG retrieval
- An embedding model turns text into a vector (384 numbers for MiniLM) so that texts with similar meaning have similar vectors.
- Chunking is structure-based: one chunk per `##` section, because each section is one self-contained rule. Long unstructured documents would need size-based splitting with overlap, and the chunk size would be bounded by the embedding model (MiniLM truncates input beyond 256 word-pieces).
- Each chunk is prefixed with its document title, so short chunks keep their context.
- Embeddings are computed locally, so document text never leaves the machine. That matters for accounting data.
- Documents and queries must be embedded with the same model; vectors from different models aren't comparable.
- Chroma stores chunks with their vectors and finds the nearest ones by cosine distance (lower = more similar), using an HNSW index so it doesn't compare against every vector.
- Retrieval always returns the top k, even for off-topic questions (weather: ~0.85 distance vs ~0.15–0.45 for real matches). So the LLM must be told to say when the retrieved text doesn't answer the question.

## Step 3: Tools
- A tool is a Python function plus a declaration (name, description, parameter schema). The LLM only sees the declaration and can only *request* a call; my code decides whether and how it runs.
- Descriptions are the interface: they tell the LLM when to use each tool (e.g. "rules and definitions, NOT the user's own invoices"), so they're written as carefully as the code.
- I chose predefined, parameterised query functions over LLM-written SQL: safer, predictable and testable, at the cost of only answering anticipated questions. Allowing LLM SQL would need a read-only restricted view, a SELECT-only check, row limits and timeouts.
- Defense in depth: besides fixed queries, the database is opened read-only (`mode=ro`), so even a buggy query can't modify data.
- Only tools in an explicit allowlist can run; an unknown tool name returns an error.
- All values go through `?` parameters; only fixed SQL fragments are concatenated. Client names with apostrophes would break naive string-built SQL. That's SQL injection in miniature.
- Totals are computed in code from integer cents, not by the LLM, because LLMs are unreliable at arithmetic.
- Errors are returned as structured data, so the agent can read them and retry instead of crashing.
- "Today" is configurable (APP_TODAY), so date-dependent answers like "overdue" are reproducible in tests.
- Date-dependent results include the date they were computed against (`"today"` in the output), so the LLM can state "as of" correctly. The date is deliberately not a tool argument: facts like the current date come from the system, not from the model.
