""" the same agent as in `agent.py`, but built with LangChain """
import json
import sys

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError

from app.agent import MAX_STEPS, MODEL, SYSTEM_PROMPT # same as from `agent.py`
from app.tools import TOOL_DECLARATIONS, run_tool

import time
from langchain_core.messages import AIMessage
from app.agent import AgentResult, ToolCall

def as_langchain_tool(decl: dict) -> StructuredTool:
    """ wrap a tool declaration as a LangChain tool

    Reuses the existing JSON Schema, and routes every call through run_tool,
    so the allowlist and error handling are the same as in `agent.py`.
    """
    name = decl["name"]

    def call(**kwargs):
        print(f"\t[lc] {name}({json.dumps(kwargs)})")
        return run_tool(name, kwargs)

    return StructuredTool.from_function(
        func=call,
        name=name,
        description=decl["description"],
        args_schema=decl["parameters"],
    )

LC_TOOLS = [as_langchain_tool(decl) for decl in TOOL_DECLARATIONS]



llm = ChatGoogleGenerativeAI(model=MODEL, timeout=60, max_retries=2)
agent = create_agent(llm, LC_TOOLS, system_prompt=SYSTEM_PROMPT)








def run_agent(question: str) -> AgentResult:
    """ ask the LangChain agent a question.  returns an AgentResult with trace """
    started = time.perf_counter()

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            # each round has a model step and a tools step, so 2*MAX_STEPS+1 allows MAX_STEPS rounds
            config={"recursion_limit": 2 * MAX_STEPS + 1},
        )
    except GraphRecursionError:
        return AgentResult(
            answer=f"Stopped: no final answer within {MAX_STEPS} rounds",
            status="step_limit",
            seconds=time.perf_counter() - started,
        )
    except Exception as e:
        print(f"Model call failed: {type(e).__name__}: {e}")
        return AgentResult(
            answer=f"Model call failed ({type(e).__name__}).",
            status="model_error",
            seconds=time.perf_counter() - started,
        )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    tool_calls = [
        ToolCall(name=tc["name"], args=dict(tc["args"]))
        for m in ai_messages
        for tc in m.tool_calls
    ]
    return AgentResult(
        answer=result["messages"][-1].text or "(no answer)",
        tool_calls=tool_calls,
        rounds=len(ai_messages),
        seconds=time.perf_counter()-started,
    )


def ask(question: str) -> str:
    """ for backwards compatibility. returns only the answer text """
    return run_agent(question).answer



if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    if not q:
        sys.exit('Usage: python -m app.agent_lc "your question"')
    print(f"Q: {q}")
    print(f"A: {ask(q)}")




