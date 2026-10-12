
"""Day 6: Robust travel assistant with defensive tool handling."""

import json
import sys
from pathlib import Path
from collections import Counter


DAY3_PATH = Path(__file__).resolve().parent.parent / "Day_3"

if str(DAY3_PATH) not in sys.path:
    sys.path.insert(0, str(DAY3_PATH))

try:
    from config import client, MODEL
    CONFIG_ERROR = None
except Exception as error:
    client = None
    MODEL = None
    CONFIG_ERROR = f"{type(error).__name__}: {error}"


from tools_v2 import TOOLS, TOOL_FUNCTIONS, SCHEMAS
from validate import validate_arguments


SYSTEM_PROMPT = """
You are a helpful college travel assistant.

Available tools:
- check_bus_fare: checks sample bus fares for supported routes.
- check_weather_advice: gives travel advice for supported weather conditions.

Rules:
1. Use the appropriate tool for fare and weather questions.
2. Never invent a fare, route, bus type, or weather condition.
3. Clearly explain tool errors.
4. Answer general questions directly without calling a tool.
5. You may call multiple tools for a question requiring multiple results.
6. These are sample fares and advice, not live booking or live weather data.
"""

MAX_STEPS = 6
INITIAL_MAX_TOKENS = 500
MAX_MAX_TOKENS = 2000
MAX_LENGTH_RETRIES = 3
REPEAT_LIMIT = 3


def handle_tool_call(call, log=True):
    """Parse, validate, and safely execute one tool call."""

    # Read the tool name and raw arguments.
    try:
        function_info = call.function
        name = function_info.name
        raw_arguments = function_info.arguments
    except (AttributeError, TypeError):
        return "Tool error: malformed tool-call object."

    # Parse arguments.
    try:
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments or "{}")
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            return "Tool error: arguments must be a JSON object."
    except json.JSONDecodeError as error:
        return f"Tool error: invalid JSON arguments: {error.msg}."

    if not isinstance(arguments, dict):
        return "Tool error: arguments must decode to a JSON object."

    # Reject unknown tools.
    if name not in TOOL_FUNCTIONS or name not in SCHEMAS:
        return f"Tool error: unknown tool '{name}'."

    # Validate against the shared schema.
    try:
        problem = validate_arguments(arguments, SCHEMAS[name])
    except Exception as error:
        return f"Tool error: validator failed: {error}"

    if problem:
        return f"Tool error: argument validation failed: {problem}"

    # Execute the function safely.
    try:
        result = str(TOOL_FUNCTIONS[name](**arguments))
    except Exception as error:
        result = f"Tool error: {type(error).__name__}: {error}"

    if log:
        print(f"  {name}({arguments}) -> {result}")

    return result


def serialize_tool_call(call):
    """Convert a tool call into a dictionary for conversation history."""
    arguments = call.function.arguments

    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)

    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": arguments,
        },
    }


def agent(question, max_steps=MAX_STEPS):
    """Run the tool-calling loop with safety limits."""

    if client is None or MODEL is None:
        return (
            "Configuration error: unable to load client and MODEL "
            f"from Day_3/config.py. Details: {CONFIG_ERROR}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    repeated_calls = Counter()
    max_tokens = INITIAL_MAX_TOKENS

    print(f"\nQuestion: {question}")

    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}")

        # Retry truncated responses with a larger token budget.
        choice = None

        for attempt in range(MAX_LENGTH_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=max_tokens,
                )
            except Exception as error:
                return (
                    f"API error ({type(error).__name__}): {error}"
                )

            if not response.choices:
                return "ERROR: The provider returned no response choices."

            choice = response.choices[0]

            if choice.finish_reason != "length":
                break

            if (
                max_tokens >= MAX_MAX_TOKENS
                or attempt >= MAX_LENGTH_RETRIES
            ):
                return (
                    "ERROR: Response remained truncated after "
                    "increasing the token limit."
                )

            max_tokens = min(max_tokens * 2, MAX_MAX_TOKENS)
            print(f"Response truncated; retrying with {max_tokens} tokens.")

        message = choice.message
        tool_calls = message.tool_calls or []

        # A response without tool calls is the final answer.
        if not tool_calls:
            answer = message.content or "(Empty response from the model.)"
            print(f"\nAnswer: {answer}")
            return answer

        # Preserve the assistant's complete tool-call message.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                serialize_tool_call(call) for call in tool_calls
            ],
        })

        # Process every call, including calls returned together.
        for call in tool_calls:
            try:
                signature = (
                    call.function.name,
                    str(call.function.arguments),
                )
            except AttributeError:
                signature = ("malformed", str(call))

            repeated_calls[signature] += 1

            if repeated_calls[signature] >= REPEAT_LIMIT:
                result = (
                    "Tool error: repeated identical call detected. "
                    "Please try a different request."
                )
            else:
                result = handle_tool_call(call)

            # Each tool call must have its own matching result message.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return (
        f"Stopped after {max_steps} steps to prevent an infinite loop."
    )


if __name__ == "__main__":
    if client is None or MODEL is None:
        print(
            "Configuration could not be loaded.\n"
            f"Details: {CONFIG_ERROR}\n"
            "Check Day_3/config.py and its required dependencies."
        )
    else:
        print(f"Using model: {MODEL}")

        test_questions = [
            "What is the express bus fare from Chennai to Madurai?",
            "What is the AC bus fare from Chennai to Coimbatore?",
            "Give travel advice for rainy weather in Chennai.",
            "Hello! What can you help me with?",
            (
                "Find the ordinary bus fare from Chennai to Bengaluru "
                "and give sunny-weather advice for Bengaluru."
            ),
        ]

        for question in test_questions:
            agent(question)
