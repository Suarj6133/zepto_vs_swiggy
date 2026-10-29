#tool_utils.py

from swiggy_mcp_tool import get_swiggy_tools
from zepto_mcp_tool import get_zepto_tools


def _prefix_tool(t, prefix: str):
    """Return a copy of the tool with its name prefixed, so tools with
    the same name across platforms (e.g. 'search_products') don't collide
    when bound together to one LLM."""
    return t.model_copy(update={"name": f"{prefix}_{t.name}"})


async def get_prefixed_swiggy_tools(names: list[str] | None = None):
    """Fetch Swiggy tools, optionally filter by original name, and prefix
    each with 'swiggy_' to avoid name collisions with Zepto tools."""
    tools = await get_swiggy_tools()
    if names:
        tools = [t for t in tools if t.name in names]
    return [_prefix_tool(t, "swiggy") for t in tools]


async def get_prefixed_zepto_tools(names: list[str] | None = None):
    """Fetch Zepto tools, optionally filter by original name, and prefix
    each with 'zepto_' to avoid name collisions with Swiggy tools."""
    tools = await get_zepto_tools()
    if names:
        tools = [t for t in tools if t.name in names]
    return [_prefix_tool(t, "zepto") for t in tools]

