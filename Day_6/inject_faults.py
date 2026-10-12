
"""Day 6: deterministic fault-injection tests; no model required."""

import json
from types import SimpleNamespace

from robust_agent import handle_tool_call


def fake_call(name, arguments):
    return SimpleNamespace(
        id="test-call",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


FAULTS = [
    ("valid call", "check_bus_fare",
     '{"route":"CHENNAI-MADURAI","bus_type":"express"}'),
    ("invalid JSON", "check_bus_fare",
     '{"route":"CHENNAI-MADURAI"'),
    ("unknown tool", "send_email", '{"to":"test@example.com"}'),
    ("missing argument", "check_bus_fare",
     '{"route":"CHENNAI-MADURAI"}'),
    ("wrong type", "check_bus_fare",
     '{"route":123,"bus_type":"express"}'),
    ("invalid enum", "check_bus_fare",
     '{"route":"CHENNAI-MADURAI","bus_type":"luxury"}'),
    ("extra argument", "check_bus_fare",
     '{"route":"CHENNAI-MADURAI","bus_type":"ac","year":2026}'),
    ("wrong tool capitalization", "Check_Bus_Fare",
     '{"route":"CHENNAI-MADURAI","bus_type":"ac"}'),
    ("JSON array instead of object", "check_bus_fare",
     json.dumps(["CHENNAI-MADURAI", "express"])),
]


if __name__ == "__main__":
    print("=== DAY 6 FAULT INJECTION ===")

    for label, name, arguments in FAULTS:
        result = handle_tool_call(fake_call(name, arguments), log=False)
        print(f"\n{label}\n  -> {result}")

    print("\nFinished: every test returned a result without an uncaught error.")
