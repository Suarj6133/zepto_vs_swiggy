import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

# Where mcp-remote stores OAuth tokens on disk
TOKEN_DIR = Path.home() / ".mcp-auth"

# Swiggy Instamart MCP server endpoint
MCP_URL = "https://mcp.swiggy.com/im"

# Swiggy's OAuth token refresh endpoint
AUTH_SERVER = "https://mcp.swiggy.com"

# MD5 hash of MCP_URL — used to identify Swiggy's token files
# among other services (Zepto etc.) stored in the same folder
SERVER_HASH = hashlib.md5(MCP_URL.encode()).hexdigest()


def _find_latest(pattern: str) -> Path | None:
    """Search ~/.mcp-auth/ recursively for files matching <SERVER_HASH><pattern>.
    Returns the most recently modified match — handles multiple mcp-remote versions."""
    files = list(TOKEN_DIR.rglob(f"{SERVER_HASH}{pattern}"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def get_access_token() -> str:
    """Get a valid Swiggy access token. Tries cached token first,
    refreshes automatically if expired, raises error if refresh also fails."""

    # Step 1: Find the token file for Swiggy (matched by hash)
    token_file = _find_latest("_tokens.json")
    if not token_file:
        raise ValueError("No tokens found. Start Swiggy MCP in VS Code first.")

    with open(token_file) as f:
        tokens = json.load(f)

    # Step 2: Validate token by making a test request to Swiggy MCP
    # If not 401, token is still valid — use it
    try:
        r = httpx.post(MCP_URL, headers={
            "Authorization": f"Bearer {tokens['access_token']}",
            "Content-Type": "application/json",
        }, json={"jsonrpc": "2.0", "method": "initialize", "id": 0, "params": {}})
        if r.status_code != 401:
            return tokens["access_token"]
    except Exception:
        return tokens["access_token"]

    # Step 3: Token expired (got 401) — try refreshing using refresh_token
    client_file = _find_latest("_client_info.json")
    if not client_file or not tokens.get("refresh_token"):
        raise ValueError("Token expired. Re-auth in VS Code.")

    with open(client_file) as f:
        client_id = json.load(f)["client_id"]

    # Step 4: POST to auth server's /token endpoint with refresh_token
    resp = httpx.post(f"{AUTH_SERVER}/token", data={
        "grant_type": "refresh_token",
        "refresh_token": tokens["refresh_token"],
        "client_id": client_id,
    })
    if resp.status_code != 200:
        raise ValueError("Refresh failed. Re-auth in VS Code.")

    # Step 5: Save new tokens to disk so next run doesn't need refresh
    new_tokens = resp.json()
    with open(token_file, "w") as f:
        json.dump(new_tokens, f, indent=2)

    print("Token refreshed.")
    return new_tokens["access_token"]


async def get_swiggy_tools():
    """Connect to Swiggy Instamart MCP using streamable_http transport with Bearer token.
    Returns list of all available tools (search, cart, order, etc.)."""

    client = MultiServerMCPClient(
        {
            "swiggy-instamart": {
                "url": MCP_URL,
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {get_access_token()}"},
            },
        }
    )

    all_tools = await client.get_tools()

    print(f"\nSwiggy Instamart MCP - Found {len(all_tools)} tools\n")
    for i, tool in enumerate(all_tools, 1):
        print(f"{i}. {tool.name}")
        if tool.description:
            print(f"   {tool.description[:200]}")
        print()

    return all_tools


if __name__ == "__main__":
    asyncio.run(get_swiggy_tools())
