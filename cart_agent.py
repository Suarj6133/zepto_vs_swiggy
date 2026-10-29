import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tool_utils import get_prefixed_swiggy_tools, get_prefixed_zepto_tools
from agent_state import AgentState, AgentResponse, make_agent_response

load_dotenv()

# Global graph — built once on first call
_cart_graph = None

CART_SYSTEM_PROMPT = """You are a cart management assistant for Swiggy Instamart
and Zepto. You can view, update, and clear the shopping cart on both platforms.

Tool names are prefixed by platform:
- swiggy_get_cart, swiggy_update_cart, swiggy_clear_cart
- zepto_view_cart, zepto_update_cart

Use the platform-specific tool that matches what the user is asking about.
If the platform isn't specified and context doesn't make it clear, ask the user
which platform (Swiggy or Zepto) they mean.

IMPORTANT — SWIGGY CART OPERATIONS REQUIRE selectedAddressId:
swiggy_update_cart takes a `selectedAddressId` parameter, and Swiggy will
reject cart operations without it (error: "delivery address is required").
The address ID was surfaced earlier in this conversation by the address
step (a string like "d9sboac1d96lknb51ra0__AbHeOgSXoy00_JG9A_nZ8W", often
shown as "(Address ID: ...)" or "(ID: ...)"). You MUST extract that exact
ID from the conversation context and pass it as `selectedAddressId` on
every swiggy_update_cart call. Also pass each item's `spinId` (and `skuId`
if available) and `quantity` from the search results in context.
Do NOT call swiggy_update_cart without selectedAddressId — if no address ID
exists anywhere in context, tell the user their Swiggy address is needed
instead of calling the tool.

If a tool response STILL indicates a delivery address is required after you
passed selectedAddressId, do NOT guess a different address — tell the user.
"""


async def _build_graph():
    """Build the cart sub-agent with Swiggy + Zepto cart tools."""
    swiggy_tools = await get_prefixed_swiggy_tools(
        ["get_cart", "update_cart", "clear_cart"]
    )
    zepto_tools = await get_prefixed_zepto_tools(["view_cart", "update_cart"])
    tools = swiggy_tools + zepto_tools

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: AgentState):
        messages = [("system", CART_SYSTEM_PROMPT)] + state["messages"]
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


async def cart_agent_question(user_question: str, history: list = None) -> tuple[AgentResponse, list]:
    """Ask the cart agent a question about Swiggy or Zepto cart.
    Pass history from previous call to maintain conversation.
    Returns (AgentResponse, updated_history)."""
    global _cart_graph

    if _cart_graph is None:
        _cart_graph = await _build_graph()

    messages = history or []
    messages.append(("user", user_question))

    result = await _cart_graph.ainvoke({"messages": messages})
    answer_text = result["messages"][-1].content

    response = await make_agent_response(answer_text)
    return response, result["messages"]


if __name__ == "__main__":
    pass
