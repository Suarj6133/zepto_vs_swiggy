import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tool_utils import get_prefixed_swiggy_tools, get_prefixed_zepto_tools
from agent_state import AgentState, AgentResponse, make_agent_response

load_dotenv()

# Global graph — built once on first call
_search_graph = None

SEARCH_SYSTEM_PROMPT = """You are a product search assistant for Swiggy
Instamart and Zepto. You can search for products, look up past ordered
items, and fetch product details on both platforms.

Tool names are prefixed by platform:
- swiggy_search_products, swiggy_your_go_to_items
- zepto_search_products, zepto_search_multiple_products,
  zepto_get_product_details, zepto_get_past_order_items

Use the platform-specific tool that matches what the user is asking about.
If the platform isn't specified, search on both and present both results.

IMPORTANT — SWIGGY REQUIRES addressId ON EVERY SEARCH CALL:
Unlike Zepto (which has a separate select-address step that sets store
context for the whole session), Swiggy has NO such step. swiggy_search_products
takes an addressId parameter directly, and you MUST pass it on every call.
- Check the conversation context for an address ID that was already
  confirmed/mentioned earlier (e.g. an address_agent response showing an
  address like "[Home test] (d9sboac1d96lknb51ra0...)" — the parenthetical
  value or similar ID-looking string is the addressId to use).
- If no addressId is available anywhere in context, you do NOT have enough
  information to search Swiggy. Do NOT call swiggy_search_products without
  an addressId. Instead, tell the user you need their Swiggy address first.

If a search tool response indicates that a delivery address or store context
is required before results can be shown, do NOT try to pick or guess an
address yourself. Clearly tell the user an address is needed.
"""


async def _build_graph():
    """Build the search sub-agent with Swiggy + Zepto search tools."""
    swiggy_tools = await get_prefixed_swiggy_tools(
        ["search_products", "your_go_to_items"]
    )
    zepto_tools = await get_prefixed_zepto_tools(
        [
            "search_products",
            "search_multiple_products",
            "get_product_details",
            "get_past_order_items",
        ]
    )
    tools = swiggy_tools + zepto_tools

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: AgentState):
        messages = [("system", SEARCH_SYSTEM_PROMPT)] + state["messages"]
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


async def search_agent_question(user_question: str, history: list = None) -> tuple[AgentResponse, list]:
    """Ask the search agent a question about products on Swiggy or Zepto.
    Pass history from previous call to maintain conversation.
    Returns (AgentResponse, updated_history)."""
    global _search_graph

    if _search_graph is None:
        _search_graph = await _build_graph()

    messages = history or []
    messages.append(("user", user_question))

    result = await _search_graph.ainvoke({"messages": messages})
    answer_text = result["messages"][-1].content

    response = await make_agent_response(answer_text)
    return response, result["messages"]


if __name__ == "__main__":
    pass
