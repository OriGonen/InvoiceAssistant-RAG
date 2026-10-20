""" agent loop """
import json
import os
import sys
from datetime import date
from dotenv import load_dotenv
from google import genai
from app.tools import TOOL_DECLARATIONS, run_tool
import time

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL")
TODAY = os.getenv("APP_TODAY") or date.today().isoformat()

GEMINI_TOOLS = [{"type": "function", **d} for d in TOOL_DECLARATIONS]
client = genai.Client() # reads GEMINI_API_KEY from the environment

MAX_STEPS = 6 # limit the loop so it doesn't loop forever

SYSTEM_PROMPT = f"""
You are an assistant for a small accounting firm.
Today is {TODAY}. Use it to resolve phrases like "this year" or "in September".
Rules:
- Answer only from tool results. Never guess or invent numbers.
- Invoice questions: use the invoice tools. VAT, payment terms, reverse charge, invoice requirements: use search_documents.
- If the documents don't cover the question, say so plainly.
- Don't do arithmetic yourself; use the totals the tools return.
- Amounts are in EUR.
Be concise.
"""


# Errors worth retrying: the same request may succeed a moment later.
RETRYABLE_ERRORS = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}

def ask(question: str) -> str:
    """ ask the agent a question """
    history = [{
        "type": "user_input",
        "content": [{"type": "text", "text": question}]
    }]

    for loop_step in range(MAX_STEPS):
        t0 = time.perf_counter()
        try:
            interaction = client.interactions.create(
                model=MODEL,
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
                input=history,
                store=False,
                timeout=60,
            )
        except Exception as e:
            name = type(e).__name__
            if name in RETRYABLE_ERRORS:
                print(f"\t(round {loop_step}: {name} after {time.perf_counter() - t0:.1f}s, retrying)")
                time.sleep(2)  # short pause before trying again
                continue  # same history -> same request; counts against MAX_STEPS
            return f"Model call failed: {name}: {e}"
        print(f"\t(round {loop_step}: {time.perf_counter() - t0:.1f}s)")

        for step in interaction.steps:
            history.append(step.model_dump())

        calls = [s for s in interaction.steps if s.type == "function_call"]
        if not calls: # agent did not ask for any calls, so it's done
            return interaction.output_text or "(no answer)"

        for call in calls:
            args = call.arguments or {}
            print(f"\t[{loop_step}] {call.name}({json.dumps(args)})")
            result = run_tool(call.name, args)

            history.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type":"text", "text": json.dumps(result)}],
            })

    return f"\tstopped: no final answer within {MAX_STEPS} rounds (tool calls or retries)."


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    if not q:
        sys.exit('Usage: python -m app.agent "your question"')

    print(f"Q: {q}")
    print(f"A: {ask(q)}")

