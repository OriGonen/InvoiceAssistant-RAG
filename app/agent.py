""" agent loop """
import json
import os
import sys
from datetime import date
from dotenv import load_dotenv
from google import genai
from app.tools import TOOL_DECLARATIONS, run_tool
import time
from dataclasses import dataclass, field

@dataclass
class ToolCall:
    name: str
    args: dict

@dataclass
class AgentResult:
    answer: str
    status: str="ok" # "ok" | "step_limit" | "model_error"
    tool_calls: list[ToolCall] = field(default_factory=list)
    rounds: int=0
    seconds: float=0.0


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
- If the question doesn't specify a time period, state the period your figures cover (e.g. "all time" or "2026").
Be concise.
"""


# Errors worth retrying: the same request may succeed a moment later.
RETRYABLE_ERRORS = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}

def run_agent(question: str) -> AgentResult:
    """ runs the agent a question. returns an AgentResult with trace """
    started = time.perf_counter()
    tool_calls: list[ToolCall] = []
    rounds = 0

    history = [{
        "type": "user_input",
        "content": [{"type": "text", "text": question}]
    }]

    for loop_step in range(MAX_STEPS):
        rounds+=1
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
            print(f"\t(round {loop_step}: model call failed: {name}: {e})")
            return AgentResult(
                answer=f"Model call failed ({name})",
                status="model_error",
                tool_calls=tool_calls,
                rounds=rounds,
                seconds=time.perf_counter() - started,
            )
        print(f"\t(round {loop_step}: {time.perf_counter() - t0:.1f}s)")

        for step in interaction.steps:
            history.append(step.model_dump())

        calls = [s for s in interaction.steps if s.type == "function_call"]
        if not calls: # agent did not ask for any calls, so it's done
            return AgentResult(  # CHANGE 4: success result
                answer=interaction.output_text or "(no answer)",
                tool_calls=tool_calls,
                rounds=rounds,
                seconds=time.perf_counter() - started,
            )

        for call in calls:
            args = call.arguments or {}
            tool_calls.append( ToolCall(name=call.name, args=dict(args)) )
            print(f"\t[{loop_step}] {call.name}({json.dumps(args)})")
            result = run_tool(call.name, args)

            history.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type":"text", "text": json.dumps(result)}],
            })

    return AgentResult(
        answer=f"Stopped: no final answer within {MAX_STEPS} rounds",
        status="step_limit",
        tool_calls=tool_calls,
        rounds=rounds,
        seconds=time.perf_counter() - started,
    )


def ask(question: str) -> str:  # old name, old behavior
    """ returns only the answer text """
    return run_agent(question).answer

if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    if not q:
        sys.exit('Usage: python -m app.agent "your question"')

    print(f"Q: {q}")
    print(f"A: {run_agent(q)}")

