"""Shared helper: runs a tool-calling agent loop with a real LLM (Anthropic) if an
API key exists. Demos fall back to their own MOCK planner when no key is set."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def run_tool_loop(system, user_message, tool_schemas, tool_funcs, max_steps=8, log=print):
    """Generic agent loop: LLM decides -> we run the tool -> feed result back -> repeat."""
    import anthropic
    client = anthropic.Anthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    messages = [{"role": "user", "content": user_message}]

    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=model, max_tokens=1000, system=system,
            tools=tool_schemas, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                log(f"[Step {step}] THINK : {b.text.strip()}")
        if not tool_uses:
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for tu in tool_uses:
            log(f"[Step {step}] ACT   : {tu.name}({tu.input})")
            out = tool_funcs[tu.name](**tu.input)
            log(f"[Step {step}] OBSERVE: {out}")
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": str(out)})
        messages.append({"role": "user", "content": results})
    return "Stopped: max steps reached."
