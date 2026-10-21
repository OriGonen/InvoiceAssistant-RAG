""" HTTP API for the Invoice Assistant """
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.agent_langchain import run_agent

app = FastAPI(
    title="Invoice Assistant",
    description="Demo agent answering questions about invoices and tax rules",
    version="0.1.0"
)

class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=500)


class ToolCallOut(BaseModel):
    name: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    status: str
    tool_calls: list[ToolCallOut]
    rounds: int
    seconds: float



ERROR_STATUS = {"model_error": 502, "step_limit": 504}
OK = "ok"


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    result = run_agent(req.question)
    if result.status != OK:
        raise HTTPException(status_code=ERROR_STATUS.get(result.status, 500), detail=result.answer)

    return AskResponse(**asdict(result))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}