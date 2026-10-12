"""Day 3 Assignment: the same agent, with three guards added."""
import json
from config import client, MODEL, banner
from my_agent import SYSTEM_PROMPT
from my_tools import TOOLS, TOOL_FUNCTIONS

MAX_TOOL_CHARS = 1500
CHAR_BUDGET = 30000

def agent(question, max_steps=6, verbose=True):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    seen_calls = {}
    chars_sent = 0
    for step in range(1, max_steps + 1):
        chars_sent += sum(len(str(m.get("content", ""))) for m in messages)
        if chars_sent > CHAR_BUDGET:
            return f"Stopped: character budget exceeded ({chars_sent} sent)."
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content.strip()
        messages.append({
            "role": "assistant", "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in message.tool_calls]})
        for call in message.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
                function = TOOL_FUNCTIONS.get(name)
                result = str(function(**arguments)) if function else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Argument error: {e}"
            signature = (name, json.dumps(locals().get("arguments", {}), sort_keys=True))
            seen_calls[signature] = seen_calls.get(signature, 0) + 1
            if seen_calls[signature] >= 3:
                return f"Stopped: {name} called 3 times with same arguments. Last result: {result[:200]}"
            if len(result) > MAX_TOOL_CHARS:
                result = result[:MAX_TOOL_CHARS] + " ... [truncated]"
            if verbose:
                print(f"   step {step}: {name}({arguments}) -> {result[:120]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Stopped: maximum steps reached without a final answer."

if __name__ == "__main__":
    banner("MY AGENT (guards on)")
    for question in [
        "Read library.html and tell me the total fine for all of Ranjith's overdue books.",
        "Read missing_records.html and tell me the fine for Clean Code.",   # intentionally missing
        "Read library.html and tell me how many books are overdue.",
    ]:
        print("\nQ:", question)
        print("A:", agent(question))