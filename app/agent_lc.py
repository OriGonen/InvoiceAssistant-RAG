""" the same agent as in `agent.py`, but built with LangChain """
import json
import sys

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError

from app.agent import MAX_STEPS, MODEL, SYSTEM_PROMPT # same as from `agent.py`
from app.tools import TOOL_DECLARATIONS, run_tool



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

def ask(question: str) -> str:
    """ ask the agent a question """
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            # each round has a model step and a tools step, so 2*MAX_STEPS+1 allows MAX_STEPS rounds
            config={"recursion_limit": 2 * MAX_STEPS + 1},
        )
    except GraphRecursionError:
        return f"Stopped: no final answer within {MAX_STEPS} rounds."
    except Exception as e:
        return f"Model call failed: {type(e).__name__}: {e}"

    return result["messages"][-1].text  # return the final message as the answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    if not q:
        sys.exit('Usage: python -m app.agent_lc "your question"')
    print(f"Q: {q}")
    print(f"A: {ask(q)}")




