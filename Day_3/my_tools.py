"""Tools for the library fine agent."""
import ast, operator, os, re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}

def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Unsupported expression")

def calculator(expression: str) -> str:
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}. Use only numbers and + - * / ( )."

TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)
SPACES = re.compile(r"\s+")

def read_webpage(url: str, max_chars: int = 2000) -> str:
    try:
        if os.path.exists(url):
            raw = open(url, encoding="utf-8", errors="ignore").read()
        else:
            return f"Read error: '{url}' is not a URL and no such file exists."
    except Exception as e:
        return f"Read error: {type(e).__name__}: {e}"
    text = SPACES.sub(" ", TAG.sub(" ", raw)).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f" ... [truncated, {len(text)} characters total]"
    return text or "Read error: the page contained no readable text."

TOOL_FUNCTIONS = {"calculator": calculator, "read_webpage": read_webpage}
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression using + - * / and brackets.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "read_webpage",
        "description": "Read a local HTML/text file and return its visible text. Give a file name such as library.html.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
]

if __name__ == "__main__":
    print(calculator("5 * (5 + 2)"))
    print(read_webpage("library.html")[:200])
    print(read_webpage("no_such_file.html"))