#address_agent.py

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tool_utils import get_prefixed_swiggy_tools, get_prefixed_zepto_tools
from agent_state import AgentState, AgentResponse, make_agent_response

load_dotenv()

# Global graph — built once on first call
_address_graph = None

ADDRESS_SYSTEM_PROMPT = """You are an address management assistant for Swiggy
Instamart and Zepto. You can list, add, select, and delete delivery addresses
on both platforms.

Tool names are prefixed by platform:
- swiggy_get_addresses, swiggy_create_address, swiggy_delete_address
- zepto_list_saved_addresses, zepto_select_saved_address, zepto_add_saved_address

Use the platform-specific tool that matches what the user is asking about.
If the platform isn't specified and context doesn't make it clear, ask the user
which platform (Swiggy or Zepto) they mean, or handle both if the request implies it.

If you cannot proceed without more information from the user (e.g. you need
them to pick which saved address to use, or confirm details for a new one),
clearly ask the user for what you need.

IMPORTANT: Whenever you confirm which address is active/selected (whether
suggesting a default or confirming the user's choice), you MUST include the
exact address ID string in your reply (e.g. "cu00l0hd19iktphcvgag__AMV2xwSgg40J9mKzhf6RHg"
for Swiggy, or a Zepto address ID). Other parts of the system need this
exact ID to perform searches and cart operations — do not omit it or
paraphrase it, even though it's not human-friendly. You may show it in a
short technical note, e.g. "(ID: <the id>)" alongside the readable address.

IMPORTANT: If you have successfully selected/confirmed an address (i.e. you
did NOT need to ask the user to pick between multiple options), state this
as a plain FACT and STOP there. Do NOT ask a follow-up question like "would
you like me to proceed with X?" or "should I search for Y?" — even if the
user's original message also mentioned searching, adding to cart, or
ordering. Those are handled by a different part of the system automatically
right after you; you have no visibility into whether they will happen, so
do not comment on them or ask for confirmation about them. Only ask a
question if YOU specifically still need more information to complete YOUR
OWN task (e.g. multiple ambiguous saved addresses to choose from).
"""


async def _build_graph():
    """Build the address sub-agent with Swiggy + Zepto address tools."""
    swiggy_tools = await get_prefixed_swiggy_tools(
        ["get_addresses", "create_address", "delete_address"]
    )
    zepto_tools = await get_prefixed_zepto_tools(
        ["list_saved_addresses", "select_saved_address", "add_saved_address"]
    )
    tools = swiggy_tools + zepto_tools

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)  #our address agent 

    def assistant(state: AgentState):
        messages = [("system", ADDRESS_SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_assistant(state: AgentState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", route_assistant)
    builder.add_edge("tools", "assistant")

    return builder.compile()


async def address_agent_question(user_question: str, history: list = None) -> tuple[AgentResponse, list]:
    """Ask the address agent a question about Swiggy or Zepto addresses.
    Pass history from previous call to maintain conversation.
    Returns (AgentResponse, updated_history)."""
    global _address_graph

    if _address_graph is None:
        _address_graph = await _build_graph()

    messages = history or []
    messages.append(("user", user_question))

    result = await _address_graph.ainvoke({"messages": messages})
    answer_text = result["messages"][-1].content

    response = await make_agent_response(answer_text)
    return response, result["messages"]


if __name__ == "__main__":
    pass
