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

## Step 4a — Hand-written agent loop
- An "agent" is a loop: model either answers or requests a tool call → my code
  runs the tool → result goes back → repeat until a text answer. The model never
  executes code itself.
- Used Gemini's Interactions API (Google's recommended API since mid-2026;
  generateContent is legacy). It has no automatic function calling in Python, so
  the loop is explicit and every tool call is visible and logged.
- Ran stateless (store=False): I resend the full history each turn instead of
  letting Google store interactions server-side. This keeps invoice data out of
  provider logs, at the cost of slightly larger requests.
- Model-generated steps (including thoughts) are passed back unchanged; each
  function_result is matched to its call via call_id, which supports parallel
  tool calls.
- Safety: tools are an allowlist (run_tool), the DB is read-only, and a
  max-steps guard stops runaway loops.
- System prompt pins "today", forbids guessing numbers or doing arithmetic,
  and requires saying when the docs don't cover a question.
- Error handling: transient errors (timeouts, connection errors, 429 rate limits,
  5xx) are retried with a short pause and count against the same MAX_STEPS
  budget as tool rounds, so the agent can never loop forever. Permanent errors
  (400 bad request, 401 auth) fail immediately with a clear message.
- Matched retryable errors by class name instead of importing them, because the
  SDK only defines them in a private module that can change between releases.
- Observed 10–27 s per model round on the free tier. Thinking was already at the
  lowest level, so the latency is on the API side; set a 60 s client timeout.
## Step 4b — Same agent with LangChain
- LangChain's create_agent is the same model ↔ tools loop as my hand-written
  version, prebuilt as a two-node LangGraph graph (model node, tools node).
- Reused my existing JSON-schema tool declarations via
  StructuredTool.from_function(args_schema=dict), routing every call through the
  same run_tool allowlist — one source of truth for tools, prompt and limits.
- Step limit: recursion_limit counts graph steps (~2 per round), so I set it to
  2 * MAX_STEPS + 1 and catch GraphRecursionError.
- Trade-offs vs. hand-written: much less code, built-in retries and thought-signature
  handling; but less visibility (no per-round timing without callbacks), retries
  no longer count against the step budget, and langchain-google-genai uses Gemini's
  older generate_content API rather than the newer Interactions API, so options like
  store=False aren't available.
- Interview point: I built the loop by hand first, so I know what the framework
  abstracts away and can debug it when it misbehaves.
- LangChain = standard interfaces (chat models, messages, tools, retrievers) in
  langchain-core, one integration package per provider (langchain-google-genai
  wraps the google-genai SDK), and higher-level pieces like create_agent built on
  LangGraph. Swapping providers means changing the model line, not the agent.
- Both agents gave identical tool calls and correct answers on the 5-question test.
## Step 5: FastAPI endpoint

- **What the API does:** `POST /ask` takes `{"question": "..."}` and returns `{answer, status, tool_calls, rounds, seconds}`. It wraps the LangChain agent; the web layer only translates between HTTP and Python.
- **Core vs. edge:** the agent returns a plain dataclass (`AgentResult`) and knows nothing about HTTP. Pydantic models (`AskRequest`, `AskResponse`) live at the API edge, where data comes from outside and must be validated.
- **Dataclass:** a class where Python generates `__init__`, `__repr__` and `__eq__` from the annotated fields. `field(default_factory=list)` gives each instance its own new list (a shared mutable default is a classic bug).
- **Validation for free:** `Field(min_length=1, max_length=500)` plus `str_strip_whitespace` → FastAPI returns 422 for empty/whitespace/overlong questions before the agent runs (also caps token cost).
- **Response model as a filter:** the return type `-> AskResponse` defines exactly what leaves the server and documents it in `/docs`.
- **`def` vs `async def`:** the agent makes blocking network calls. A plain `def` endpoint runs in FastAPI's threadpool, so the server stays responsive. `async def` with blocking code inside would freeze the event loop for every request. Rule: `async def` only if everything inside is truly async.
- **Meaningful status codes:** 200 ok, 422 invalid input (automatic), 502 when the upstream LLM call fails, 504 when the agent doesn't finish within its step budget. Errors carry a generic message; full exceptions stay in server logs.
- **Startup cost:** the embedding model and LLM client load once on import, not per request. Dev mode (`fastapi dev`) imports twice (app discovery + reload process); lazy loading is a possible improvement.
- **Two agents, one interface:** both the hand-written loop and the LangChain agent expose `run_agent(question) -> AgentResult`. LangChain backs the API; the hand-written one shows I understand what `create_agent` does internally.
- **Testing:** `/docs` (Swagger UI) for manual tests, `/health` as a cheap liveness check, `Invoke-RestMethod` from PowerShell; forced a 504 with `MAX_STEPS = 1` and a 502 with an invalid model name.
- **Security talking points:** read-only DB plus parameterised queries, tool allowlist, input limits, filtered output, secrets in `.env`. For production: auth plus rate limiting, no logging of client data, HTTPS, provider-side data retention.