import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from tool_utils import get_prefixed_swiggy_tools, get_prefixed_zepto_tools
from agent_state import AgentState, AgentResponse, make_agent_response

load_dotenv()

# Global graph — built once on first call
_compare_graph = None

COMPARE_SYSTEM_PROMPT = """You are a price comparison assistant for Swiggy
Instamart and Zepto. You compare products by SEARCHING for them on both
platforms (to get the product/variant IDs needed for cart operations),
ADDING them to cart on both platforms, and reading back the real cart
price (which reflects platform-specific discounts, not just raw search
price).

Tools available:
- swiggy_search_products, swiggy_update_cart, swiggy_get_cart, swiggy_clear_cart
- zepto_search_products, zepto_update_cart, zepto_view_cart

For EACH item the user wants compared:
1. Search for the item on Swiggy (swiggy_search_products) to get its
   productVariantId / storeProductId.
2. Add it to cart on Swiggy using those IDs (swiggy_update_cart), then read
   the cart (swiggy_get_cart) to get its real price.
3. Search for the same item on Zepto (zepto_search_products) to get its
   productVariantId.
4. Add it to cart on Zepto using that ID (zepto_update_cart), then read the
   cart (zepto_view_cart) to get its real price.
5. Determine which platform is cheaper for that item.
6. Remove the item from the MORE EXPENSIVE platform's cart (use
   swiggy_clear_cart, or zepto_update_cart with quantity 0), so that only
   the cheaper platform ends up with that item in cart.

If the user lists multiple items, repeat this whole sequence for each item
— each one may end up on a different platform depending on which is
cheaper for it.

After processing all items, tell the user which platform was cheaper for
each item and confirm that the cart has been set up accordingly (so a
follow-up "place order" can proceed directly).

If a search or cart tool response indicates a delivery address or store
context is required before proceeding, do NOT guess an address. Clearly
tell the user an address is needed.
"""


async def _build_graph():
    """Build the compare sub-agent with search + cart tools on both
    platforms — comparison works by searching for the item to get its
    product ID, adding it to cart, reading back the real price, then
    pruning the more expensive platform's cart entry."""
    swiggy_tools = await get_prefixed_swiggy_tools(
        ["search_products", "update_cart", "get_cart", "clear_cart"]
    )
    zepto_tools = await get_prefixed_zepto_tools(
        ["search_products", "update_cart", "view_cart"]
    )
    tools = swiggy_tools + zepto_tools

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: AgentState):
        messages = [("system", COMPARE_SYSTEM_PROMPT)] + state["messages"]
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


async def compare_agent_question(user_question: str, history: list = None) -> tuple[AgentResponse, list]:
    """Ask the compare agent to compare product(s) across Swiggy and Zepto.
    Adds items to cart on both platforms to compare real prices, then keeps
    only the cheaper platform's cart entry — so a follow-up "place order"
    can go straight to order_agent with no ambiguity about what to order.
    Pass history from previous call to maintain conversation.
    Returns (AgentResponse, updated_history)."""
    global _compare_graph

    if _compare_graph is None:
        _compare_graph = await _build_graph()

    messages = history or []
    messages.append(("user", user_question))

    result = await _compare_graph.ainvoke({"messages": messages})
    answer_text = result["messages"][-1].content

    response = await make_agent_response(answer_text)
    return response, result["messages"]


if __name__ == "__main__":
    pass
