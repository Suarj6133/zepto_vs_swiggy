from langchain_core.messages import AIMessage, HumanMessage

from cart_agent import cart_agent_question
from address_agent import address_agent_question
from search_agent import search_agent_question
from order_agent import order_agent_question
from compare_agent import compare_agent_question
from agent_state import agent_state_logs, AgentResponse
from router import classify_intent, RouteDecision

# Each functional sub-agent keeps its own conversation history across calls,
# separate from the orchestrator's history
_cart_history = None
_address_history = None
_search_history = None
_order_history = None
_compare_history = None


def _build_context(
    intent: str, question: str, orchestrator_messages: list, sequence_replies: list[str]
) -> str:
    """Build the question sent to a sub-agent, including relevant context.
    Two sources of context are combined:
    1. orchestrator_messages — the LAST AI reply from a PREVIOUS turn (e.g.
       a compare result the user is now reacting to in a new message).
    2. sequence_replies — replies from EARLIER steps in THIS SAME turn's
       sequence (e.g. address_agent's reply, when "search" runs right
       after "address" within one _run_sequence call). orchestrator_messages
       does NOT yet contain these — they're only appended once the whole
       sequence finishes — so without this, mid-sequence context would be
       lost entirely.

    "cart" needs this to know WHICH item to act on (e.g. after a compare
    result, or a search result earlier in this same turn). "search" needs
    this because Swiggy has no separate select-address step —
    swiggy_search_products requires an addressId parameter on every call,
    so search_agent needs the address/ID that address_agent surfaced,
    whether that happened in a prior turn OR earlier in this same turn's
    sequence. "order" does not need it — order_agent's tools
    (checkout/confirm_order) act on whatever is already in the real cart
    on the platform's servers, not on item names, so no extra context is
    needed there."""
    if intent not in ("cart", "search"):
        return question

    context_parts = []

    # Prior-turn context: last AI message from orchestrator's own history
    for msg in reversed(orchestrator_messages[:-1]):  # skip the just-appended HumanMessage
        if getattr(msg, "type", None) == "ai":
            context_parts.append(msg.content)
            break

    # Same-turn context: replies from earlier steps in this sequence
    context_parts.extend(sequence_replies)

    if not context_parts:
        return question

    context_text = "\n".join(context_parts)
    return (
        f"Context from earlier in this conversation: {context_text}\n"
        f"User's latest message: {question}"
    )


async def _dispatch_one(
    intent: str, question: str, orchestrator_messages: list, sequence_replies: list[str]
) -> AgentResponse:
    """Call the sub-agent matching a single intent, keeping that sub-agent's
    own conversation history across calls. Returns a structured AgentResponse
    (answer text + needs_input flag)."""
    global _cart_history, _address_history, _search_history, _order_history, _compare_history

    dispatch_question = _build_context(intent, question, orchestrator_messages, sequence_replies)

    if intent == "cart":
        response, _cart_history = await cart_agent_question(dispatch_question, _cart_history)
        agent_state_logs({"messages": _cart_history}, label="CART")
        return response

    if intent == "address":
        response, _address_history = await address_agent_question(question, _address_history)
        agent_state_logs({"messages": _address_history}, label="ADDRESS")
        return response

    if intent == "search":
        response, _search_history = await search_agent_question(dispatch_question, _search_history)
        agent_state_logs({"messages": _search_history}, label="SEARCH")
        return response

    if intent == "order":
        response, _order_history = await order_agent_question(question, _order_history)
        agent_state_logs({"messages": _order_history}, label="ORDER")
        return response

    if intent == "compare":
        response, _compare_history = await compare_agent_question(question, _compare_history)
        agent_state_logs({"messages": _compare_history}, label="COMPARE")
        return response

    raise ValueError(f"Unknown intent: {intent}")


async def _run_sequence(intents: list[str], question: str, orchestrator_messages: list) -> str:
    """Dispatch through an ordered list of intents. If any sub-agent signals
    needs_input=True, stop the sequence immediately and return its answer —
    the remaining intents are simply not attempted this turn.
    The same user question is passed to each sub-agent in the sequence;
    each sub-agent's own history lets it build on what earlier steps did,
    and sequence_replies carries forward same-turn context (e.g. an
    addressId from "address" to "search")."""
    last_response: AgentResponse | None = None
    sequence_replies: list[str] = []
    for intent in intents:
        last_response = await _dispatch_one(intent, question, orchestrator_messages, sequence_replies)
        sequence_replies.append(last_response.answer)
        if last_response.needs_input:
            return last_response.answer
    return last_response.answer if last_response else ""


async def orchestrate_question(user_question: str, history: list = None) -> tuple[str, list]:
    """Route a user question through the shopping assistant.

    This is a plain orchestrator, NOT an agent — there is no LLM here
    deciding dynamically what to do next. It:
    1. Classifies the question into an ORDERED list of intents (search/
       cart/address/order/compare) via router.classify_intent, a single
       structured-output LLM call.
    2. Deterministically dispatches to each sub-agent in that fixed order,
       stopping early if a sub-agent needs more input from the user.

    The actual agentic behavior (LLMs dynamically deciding which tools to
    call, looping until done) lives inside each sub-agent
    (cart_agent.py, address_agent.py, search_agent.py, order_agent.py,
    compare_agent.py) — this file only sequences calls to them.

    Pass history from previous call to maintain conversation.
    Returns (answer, updated_history)."""
    messages = history or []
    messages.append(HumanMessage(content=user_question))

    decision: RouteDecision = classify_intent(messages)
    print(f"\n[ROUTER] intents={decision.intents} — {decision.reasoning}\n")

    answer = await _run_sequence(decision.intents, user_question, messages)

    messages.append(AIMessage(content=answer))
    agent_state_logs({"messages": messages}, label="ORCHESTRATOR")

    return answer, messages


if __name__ == "__main__":
    pass
