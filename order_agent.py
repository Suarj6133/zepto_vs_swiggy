import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tool_utils import get_prefixed_swiggy_tools, get_prefixed_zepto_tools
from agent_state import AgentState, AgentResponse, make_agent_response

load_dotenv()

# Global graph — built once on first call
_order_graph = None

ORDER_SYSTEM_PROMPT = """You are an order management assistant for Swiggy
Instamart and Zepto. You can check out, place orders, track orders, check
payment options/status, and view order history on both platforms.

Tool names are prefixed by platform:
- swiggy_checkout, swiggy_get_orders, swiggy_track_order,
  swiggy_get_delivery_status, swiggy_get_payment_options,
  swiggy_check_payment_status, swiggy_confirm_order
- zepto_get_payment_methods, zepto_create_order,
  zepto_create_online_payment_order, zepto_create_wallet_order,
  zepto_create_upi_reserve_pay_order, zepto_check_payment_status,
  zepto_get_order_detail, zepto_list_order_history

Use the platform-specific tool that matches what the user is asking about.
Placing an order usually requires a delivery address and cart to already be
set up — if that information is missing, clearly tell the user what's needed
instead of guessing.
"""


async def _build_graph():
    """Build the order sub-agent with Swiggy + Zepto order/payment tools."""
    swiggy_tools = await get_prefixed_swiggy_tools(
        [
            "checkout",
            "get_orders",
            "track_order",
            "get_delivery_status",
            "get_payment_options",
            "check_payment_status",
            "confirm_order",
        ]
    )
    zepto_tools = await get_prefixed_zepto_tools(
        [
            "get_payment_methods",
            "create_order",
            "create_online_payment_order",
            "create_wallet_order",
            "create_upi_reserve_pay_order",
            "check_payment_status",
            "get_order_detail",
            "list_order_history",
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
        messages = [("system", ORDER_SYSTEM_PROMPT)] + state["messages"]
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


async def order_agent_question(user_question: str, history: list = None) -> tuple[AgentResponse, list]:
    """Ask the order agent a question about placing, tracking, or paying
    for orders on Swiggy or Zepto.
    Pass history from previous call to maintain conversation.
    Returns (AgentResponse, updated_history)."""
    global _order_graph

    if _order_graph is None:
        _order_graph = await _build_graph()

    messages = history or []
    messages.append(("user", user_question))

    result = await _order_graph.ainvoke({"messages": messages})
    answer_text = result["messages"][-1].content

    response = await make_agent_response(answer_text)
    return response, result["messages"]


if __name__ == "__main__":
    pass
