# 🛒 Shopping Assistant (Swiggy Instamart + Zepto)

A conversational, multi-agent shopping assistant that drives **Swiggy Instamart**
and **Zepto** through their official **MCP (Model Context Protocol)** servers.
Ask it in plain language to manage addresses, search products, build a cart,
compare prices across both platforms, and place orders — all from a single
Streamlit chat UI.

The system is built with **LangGraph** sub-agents, a single LLM **router** for
intent classification, and a **deterministic orchestrator** that sequences the
sub-agents. The LLM used throughout is **Google Gemini** via
`langchain-google-genai`.

---

## 🎥 Demo video

▶️ **Watch the full demo on YouTube:** [Shopping Assistant demo](https://youtu.be/YOUR_VIDEO_ID)

> ⚠️ Paste your actual YouTube URL in place of `https://youtu.be/YOUR_VIDEO_ID` above
> (I didn't receive the link — drop it in chat and I'll fill it in).

---

## 🔁 Overall flowchart

```text
                        👤 User message (Streamlit Chat UI)
                                      │
                                      ▼
                        orchestrate_question(question, history)
                                      │
                                      ▼
                        router.classify_intent()
                        (single Gemini call → RouteDecision)
                                      │
                                      ▼
                        Ordered intents list
                        e.g. ["address", "search", "cart"]
                                      │
                                      ▼
            ┌────────────────────────────────────────────────┐
            │   _run_sequence(intents)  — deterministic, no LLM│
            │                                                  │
            │   for each intent, in order:                     │
            └────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
            _build_context()  →  forward addressId / product IDs
                                     │
                                     ▼
            _dispatch_one(intent)  →  pick the matching sub-agent
                                     │
         ┌───────────┬──────────────┼──────────────┬───────────┐
         ▼           ▼              ▼              ▼           ▼
     address      search          cart          compare      order
      agent        agent          agent          agent        agent
         │           │              │              │           │
         └───────────┴──────────────┼──────────────┴───────────┘
                                     ▼
            ┌────────────────────────────────────────────────┐
            │  Inside each sub-agent — LangGraph tool loop     │
            │                                                  │
            │   assistant (LLM bound to swiggy_* / zepto_*)    │
            │        │                                         │
            │        ▼                                         │
            │   tool_calls? ──yes──▶ ToolNode runs MCP tool ──┐│
            │        │                                        ││
            │        │  ◀──────────── loop back ──────────────┘│
            │        no                                        │
            │        ▼                                         │
            │   final text answer                              │
            └────────────────────────┬─────────────────────────┘
                                     │
          (Bearer token, auto-refreshed)  ⇅  Swiggy & Zepto MCP servers
                                     │
                                     ▼
            make_agent_response()  →  AgentResponse(answer, needs_input)
                                     │
                                     ▼
                           needs_input ?
                          ┌──────┴──────┐
                      YES │             │ NO
                          ▼             ▼
                 Stop the sequence   more intents left?
                 ask the user        ┌──────┴──────┐
                          │       YES│             │NO
                          │          ▼             ▼
                          │   back to          Return final answer
                          │   _dispatch_one    + updated history
                          │          │             │
                          └──────────┴──────┬──────┘
                                            ▼
                               🖥️ Answer shown to user
```

> More detailed, block-by-block diagrams (auth / router / orchestration, plus a
> full sequence diagram) are in [`docs/architecture_flow.md`](docs/architecture_flow.md)
> and [`docs/flowchart.md`](docs/flowchart.md).

---

## How it works

The architecture is split into three blocks:

1. **🔐 Authentication** — A one-time OAuth login (via `mcp-remote` in your IDE)
   caches tokens on disk at `~/.mcp-auth/`. At runtime the app reads those
   tokens, validates them, and auto-refreshes if expired — the agent flow never
   triggers an interactive login itself.

2. **🧭 Router** — A single structured-output LLM call (`router.py`) classifies
   each user message into an **ordered list of intents**
   (`search`, `cart`, `address`, `order`, `compare`). Prerequisite rules are
   encoded in the prompt (e.g. address before shopping, search before adding a
   named item, cart before order), so the router can emit multi-step sequences
   like `["address", "search", "cart"]`.

3. **⚙️ Orchestration** — A deterministic (non-LLM) loop (`orchestrator.py`)
   walks the intent list in order, dispatches each intent to its sub-agent,
   forwards context between steps in the same turn (e.g. an `addressId` or
   product IDs), and stops early if any sub-agent needs more input from the user.

The actual *agentic* behavior (an LLM dynamically choosing which tools to call
and looping until done) lives **inside each sub-agent**, never in the
orchestrator.

> 📖 Full diagrams and design notes are in [`docs/architecture_flow.md`](docs/architecture_flow.md),
> with additional references in [`docs/agents.md`](docs/agents.md),
> [`docs/flowchart.md`](docs/flowchart.md), and
> [`docs/mcp_tools_reference.md`](docs/mcp_tools_reference.md).

---

## Project structure

| File | Role |
| --- | --- |
| `app.py` | Streamlit chat UI with tabs for chat, orchestrator logs, and per-platform search/cart logs. |
| `orchestrator.py` | Deterministic sequencer — routes a question through the sub-agents in order. |
| `router.py` | Single LLM call that classifies a message into an ordered list of intents (`RouteDecision`). |
| `agent_state.py` | Shared `AgentState` / `AgentResponse`, `needs_input` classifier, and state-logging helpers. |
| `address_agent.py` | Sub-agent: list / add / select / delete delivery addresses. |
| `search_agent.py` | Sub-agent: search products and look up past-ordered items. |
| `cart_agent.py` | Sub-agent: view / update / clear the cart. |
| `compare_agent.py` | Sub-agent: compare a product's real cart price across both platforms, keep the cheaper one. |
| `order_agent.py` | Sub-agent: checkout, place/track orders, payments, order history. |
| `tool_utils.py` | Fetches MCP tools and prefixes names (`swiggy_*` / `zepto_*`) to avoid collisions. |
| `swiggy_mcp_tool.py` | Connects to the Swiggy Instamart MCP server; token load + auto-refresh. |
| `zepto_mcp_tool2.py` | Connects to the Zepto MCP server; token load + auto-refresh. |
| `.vscode/mcp.json` | MCP server definitions used by `mcp-remote` to perform the OAuth login. |
| `docs/` | Architecture diagrams, agent docs, and MCP tool reference. |

---

## The sub-agents

Each sub-agent is a small **LangGraph** graph (`assistant → tools → assistant`
loop) bound only to the platform-prefixed tools it needs. Every sub-agent keeps
its own conversation history, separate from the orchestrator's.

| Intent | Agent | Swiggy tools | Zepto tools |
| --- | --- | --- | --- |
| `address` | `address_agent` | `get_addresses`, `create_address`, `delete_address` | `list_saved_addresses`, `select_saved_address`, `add_saved_address` |
| `search` | `search_agent` | `search_products`, `your_go_to_items` | `search_products`, `search_multiple_products`, `get_product_details`, `get_past_order_items` |
| `cart` | `cart_agent` | `get_cart`, `update_cart`, `clear_cart` | `view_cart`, `update_cart` |
| `compare` | `compare_agent` | `search_products`, `update_cart`, `get_cart`, `clear_cart` | `search_products`, `update_cart`, `view_cart` |
| `order` | `order_agent` | `checkout`, `get_orders`, `track_order`, `get_delivery_status`, `get_payment_options`, `check_payment_status`, `confirm_order` | `get_payment_methods`, `create_order`, `create_online_payment_order`, `create_wallet_order`, `create_upi_reserve_pay_order`, `check_payment_status`, `get_order_detail`, `list_order_history` |

### Platform quirks baked into the prompts

- **Swiggy requires an address ID on every call.** Unlike Zepto, Swiggy has no
  separate "select address" step — `swiggy_search_products` and
  `swiggy_update_cart` take an `addressId` / `selectedAddressId` parameter
  directly, so the address agent surfaces the exact ID and the orchestrator
  forwards it to later steps.
- **Swiggy and Zepto have fully independent address sessions.** Confirming an
  address on one platform does not confirm it on the other, even if the saved
  label is identical.
- **Compare works by adding to cart on both platforms**, reading back the real
  (discounted) cart price, then pruning the more expensive platform — so a
  follow-up "place order" can proceed directly.

---

## Prerequisites

- **Python 3.11+** (uses `X | None` type syntax and modern typing)
- **Node.js / npx** — required to run `mcp-remote` for the one-time MCP OAuth login
- A **Google Gemini API key**
- Accounts on **Swiggy Instamart** and **Zepto** (you log in via OTP during the
  MCP OAuth flow)

---

## Setup

### 1. Install dependencies

No `requirements.txt` ships with the repo yet. Install the libraries the code
imports:

```bash
pip install streamlit python-dotenv httpx \
  langchain-core langchain-google-genai \
  langgraph langchain-mcp-adapters
```

> Tip: freeze your working environment into a `requirements.txt` with
> `pip freeze > requirements.txt` once everything runs.

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

> `.env` currently also contains a `GROQ_API_KEY` placeholder, but the active
> code paths use Gemini (`gemini-3.5-flash-lite`) via `GEMINI_API_KEY`.
> **Do not commit real API keys.**

### 3. Authenticate with the MCP servers (one-time)

The app reads cached OAuth tokens from `~/.mcp-auth/`. To create them, connect
the MCP servers once using the definitions in [`.vscode/mcp.json`](.vscode/mcp.json):

```json
{
  "servers": {
    "zepto":            { "type": "stdio", "command": "npx", "args": ["mcp-remote@0.1.37", "https://mcp.zepto.co.in/mcp"] },
    "swiggy-instamart": { "type": "stdio", "command": "npx", "args": ["mcp-remote@0.1.37", "https://mcp.swiggy.com/im"] }
  }
}
```

Starting these servers in a compatible IDE (e.g. VS Code / Kiro) launches
`mcp-remote`, which opens a browser for the OAuth login (phone + OTP) on each
platform and caches the resulting tokens on disk. After that, the app refreshes
tokens automatically; you only need to re-authenticate if a refresh fails.

You can sanity-check connectivity directly:

```bash
python swiggy_mcp_tool.py   # lists Swiggy Instamart tools
python zepto_mcp_tool2.py   # lists Zepto tools
```

---

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints. The UI has four tabs:

- **Chat** — talk to the assistant
- **Orchestrator Logs** — full message/state trace per turn
- **Search Agent** — search activity split by Swiggy / Zepto
- **Cart Agent** — cart activity split by Swiggy / Zepto

### Example prompts

- `list my zepto addresses`
- `use my home address on swiggy and search for 5kg aashirvaad atta`
- `compare the price of amul milk 1L on swiggy and zepto`
- `add the cheaper one to cart and place the order`
- `track my last zepto order`

---

## Notes & caveats

- This project automates **real** shopping accounts through the live Swiggy and
  Zepto MCP servers — searches, cart changes, and order placement affect your
  actual accounts. Use with care.
- API/MCP endpoints and tool schemas are controlled by the platforms and may
  change over time.
- Keep credentials and cached tokens (`~/.mcp-auth/`) private; never commit the
  `.env` file.
