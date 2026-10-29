import os
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


class RouteDecision(BaseModel):
    """Structured classification of the user's request into an ORDERED list
    of intents. Using a Pydantic model + Literal makes routing explicit and
    type-validated, instead of relying on the LLM's free-form tool-calling
    judgment.

    Some requests genuinely need more than one sub-agent, in sequence.
    Example: "cost of 5kg oil" with no address set yet in this conversation
    needs ["address", "search"] — address must be resolved before search
    can return a priced result.
    """

    intents: list[Literal["search", "cart", "address", "order", "compare"]] = Field(
        description=(
            "Ordered list of intents needed to fully answer the user's request. "
            "search: finding/browsing products or past order items. "
            "cart: viewing, updating, or clearing the shopping cart. "
            "address: listing, adding, selecting, or deleting delivery addresses. "
            "order: checkout, placing/tracking orders, payments, order history. "
            "compare: comparing a product's price/availability between "
            "Swiggy Instamart and Zepto. "
            "Most requests need only ONE intent. Only include multiple when "
            "the request clearly depends on a prerequisite step, e.g. an "
            "address must be confirmed before search/cart/order can proceed "
            "and no address has been established yet in this conversation."
        )
    )
    reasoning: str = Field(description="One sentence explanation for this classification")


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a shopping assistant
that manages Swiggy Instamart and Zepto. Classify the user's LATEST message
into an ORDERED list of intents from: search, cart, address, order, compare.

Use the conversation history for context. In particular:
- If a sub-agent previously asked a question (e.g. "please pick an address")
  and the user's latest message answers it, classify based on what that
  pending question was about, not the literal words in the reply.

- ADDRESS IS A PREREQUISITE FOR search, cart, order, AND compare. All four
  require a delivery address / store context to be set on the real platforms
  before they can return accurate results (prices, cart contents, order
  placement, or comparison data all depend on which store is serving you).
  If the conversation history does NOT show that an address has already been
  confirmed for this session, prepend "address" to the list for ANY of these
  four intents, e.g. ["address", "search"], ["address", "cart"],
  ["address", "order"], ["address", "compare"].

- SWIGGY AND ZEPTO HAVE COMPLETELY SEPARATE, INDEPENDENT ADDRESS SESSIONS.
  Confirming an address on one platform does NOT confirm it on the other,
  even if the saved address has the identical name/label on both (e.g.
  "Home Test" saved on both Swiggy and Zepto are still two separate
  selections — picking it on Swiggy does not select it on Zepto).
  When deciding whether "address" is needed, check the conversation history
  for confirmation on the SPECIFIC platform relevant to the current request:
    - If the request names a platform (e.g. "on zepto") and history only
      shows an address confirmed for the OTHER platform (or no platform-
      specific confirmation at all), prepend "address" — do NOT assume it
      carries over.
    - If the request doesn't name a platform and could apply to either,
      and NEITHER platform has a confirmed address in history, prepend
      "address".
    - Only skip "address" if the SAME platform relevant to this request
      already had an address explicitly confirmed earlier in this
      conversation.

- If an address was already confirmed earlier in this conversation FOR THE
  SAME PLATFORM as the current request, do NOT ask for it again — just
  classify the single relevant intent (search, cart, order, or compare) on
  its own.

- If the user's LATEST message combines selecting/confirming an address
  AND naming item(s) to add/search/order in the SAME message (e.g. "use
  home test and add milk, bread to my swiggy cart", "select my office
  address and order 2 atta"), you MUST include BOTH "address" and the
  relevant second intent, e.g. ["address", "cart"] or ["address",
  "search"]. Do NOT classify this as ["address"] alone just because
  address is mentioned first — the item names in the same message mean
  the user expects both steps to happen in this turn.

- CART IS A PREREQUISITE FOR order. Only items already in the cart can be
  ordered/checked out. If the user asks to place an order and the
  conversation history does NOT show that the relevant item(s) were already
  added to cart in this session, include "cart" before "order", e.g.
  ["cart", "order"]. If cart already has the item (e.g. the user explicitly
  added it earlier, or a prior turn confirmed "added to cart"), classify
  ["order"] alone.

- If the user's message implies BOTH adding an item AND ordering it in one
  go (e.g. "add the cheaper one and place the order", "order 2 milk"),
  classify ["cart", "order"] — cart always comes first.

- If the user only asks to add something to cart with no mention of
  ordering/checkout, classify ["cart"] alone — do NOT proceed to order
  unless the user explicitly asks for it.

- SEARCH IS A PREREQUISITE FOR ADDING NAMED ITEMS TO CART. The cart agent
  can only view, clear, or modify quantities in the cart — it CANNOT look
  up a product by name. If the user asks to ADD a specific named product
  (e.g. "add Aashirvaad Atta 5kg", "add milk and bread to my cart") and
  that product has NOT already been found via search earlier in this
  conversation, you MUST include "search" before "cart", e.g.
  ["search", "cart"] (or ["address", "search", "cart"] if address is also
  unconfirmed). Only skip "search" if:
    - the request is just viewing/clearing the cart (no item names), or
    - the exact item(s) were already found via a search step earlier in
      this same conversation (so their product ID is already known).

- If a "compare" was already run earlier in this conversation, the compare
  step itself adds items to cart on the cheaper platform (comparison works
  by adding to cart on both platforms and keeping only the cheaper one).
  So if the user says "ok place order" / "go ahead" / "place the order"
  right after a comparison result, classify ["order"] alone — do NOT
  prepend "cart" again, the cart step already happened during compare.

- Most requests need only ONE intent once prerequisites (address, cart) are
  already satisfied. Only return multiple when there's a genuine
  prerequisite dependency as described above.
"""

_router_llm = None


def _get_router_llm():
    """Build (once) a Groq LLM configured to always return a RouteDecision."""
    global _router_llm
    if _router_llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        _router_llm = llm.with_structured_output(RouteDecision)
    return _router_llm


#output from _get_router_llm 
# RouteDecision(
#     intents=["address", "search"],
#     reasoning="The user wants to search for milk, but no delivery address has been confirmed in this session yet."
# )



def classify_intent(messages: list) -> RouteDecision:
    """Classify the latest message into an ordered list of intents, using
    full history for context. Returns a validated RouteDecision."""
    llm = _get_router_llm()
    full_messages = [("system", ROUTER_SYSTEM_PROMPT)] + list(messages)
    return llm.invoke(full_messages)
