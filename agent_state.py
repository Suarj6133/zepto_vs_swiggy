import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict

load_dotenv()


class AgentState(TypedDict):
    """Shared state shape for all agents (master + functional sub-agents).
    Every agent only needs a messages list, so one generic state
    is reused instead of defining a separate class per agent."""
    messages: Annotated[list, add_messages]


class AgentResponse(BaseModel):
    """Structured final response from a sub-agent, returned to orchestrator
    instead of a plain string."""
    answer: str
    needs_input: bool = Field(
        description=(
            "True if the agent cannot proceed without more information "
            "from the user (e.g. picking an address, confirming a detail, "
            "or answering a clarifying question). False if the answer is "
            "already complete."
        )
    )


class _NeedsInputCheck(BaseModel):
    needs_input: bool = Field(
        description=(
            "True if this reply is asking the user a question or requires "
            "them to provide/confirm information before the assistant can "
            "proceed. False if the reply is a complete, final answer."
        )
    )


_needs_input_llm = None

_NEEDS_INPUT_SYSTEM_PROMPT = """Classify whether the following assistant
reply is a COMPLETE final answer, or if it is asking the user a question /
waiting on more input before it can proceed (e.g. "which address should I
use?", "please confirm the address", "which platform did you mean?").

Respond with JSON matching this exact schema: {"needs_input": true} or
{"needs_input": false}. Only output the JSON object, nothing else.
"""


def _get_needs_input_llm():
    """Build (once) a small Gemini LLM configured to classify needs_input."""
    global _needs_input_llm
    if _needs_input_llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        _needs_input_llm = llm.with_structured_output(
            _NeedsInputCheck, method="json_mode"
        )
    return _needs_input_llm


def extract_text(content) -> str:
    """Normalize a message's .content into a plain string.
    Some models (e.g. Gemini) return content as a list of content blocks
    like [{'type': 'text', 'text': '...', 'extras': {...}}] instead of a
    plain string. This flattens either shape into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


async def make_agent_response(answer_text) -> AgentResponse:
    """Wrap a sub-agent's answer into a structured AgentResponse,
    classifying needs_input via a lightweight LLM call. Accepts either a
    plain string or a list of content blocks (normalized via extract_text)."""
    answer_text = extract_text(answer_text)
    llm = _get_needs_input_llm()
    check = await llm.ainvoke(
        [("system", _NEEDS_INPUT_SYSTEM_PROMPT), ("user", answer_text)]
    )
    return AgentResponse(answer=answer_text, needs_input=check.needs_input)


def agent_state_logs(result, label: str = "AGENT") -> str:
    """Print all messages in the state for debugging, and return the same
    text (so callers like a Streamlit app can display it in a tab)."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"{label} STATE MESSAGES LOG")
    lines.append("=" * 60)
    for i, msg in enumerate(result["messages"]):
        role = msg.type
        lines.append(f"\n[{i}] {role.upper()}")
        if msg.content:
            lines.append(f"    Content: {msg.content[:200]}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                lines.append(f"    Tool Call: {call['name']}({call['args']})")
        if hasattr(msg, "name") and msg.name:
            lines.append(f"    Tool Name: {msg.name}")
    lines.append("=" * 60 + "\n")

    log_text = "\n".join(lines)
    print(log_text)
    return log_text
