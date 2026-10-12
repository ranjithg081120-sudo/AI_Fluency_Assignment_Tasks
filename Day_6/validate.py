
"""Validate tool arguments using the same schemas sent to the model."""

from tools_v2 import SCHEMAS


def validate_arguments(arguments, schema):
    if not isinstance(arguments, dict):
        return "Arguments must be a JSON object."

    properties = schema["properties"]
    required = schema.get("required", [])

    missing = [key for key in required if key not in arguments]
    if missing:
        return (
            f"Missing required argument(s): {', '.join(missing)}. "
            f"Expected: {', '.join(properties)}."
        )

    extra = [key for key in arguments if key not in properties]
    if extra:
        return (
            f"Unexpected argument(s): {', '.join(extra)}. "
            f"Allowed: {', '.join(properties)}."
        )

    for name, value in arguments.items():
        rule = properties[name]
        expected = rule["type"]

        if expected == "string" and not isinstance(value, str):
            return (
                f"Argument '{name}' must be a string, "
                f"but got {type(value).__name__}: {value!r}."
            )

        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return f"Argument '{name}' must be an integer."

        if expected == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return f"Argument '{name}' must be a number."

        if expected == "boolean" and not isinstance(value, bool):
            return f"Argument '{name}' must be a boolean."

        if "enum" in rule and value not in rule["enum"]:
            return (
                f"Argument '{name}' must be one of {rule['enum']}; "
                f"got {value!r}."
            )

    return None


if __name__ == "__main__":
    examples = [
        ("Valid", "check_bus_fare",
         {"route": "CHENNAI-MADURAI", "bus_type": "express"}),
        ("Missing argument", "check_bus_fare", {"route": "CHENNAI-MADURAI"}),
        ("Wrong type", "check_bus_fare",
         {"route": 123, "bus_type": "express"}),
        ("Invalid enum", "check_bus_fare",
         {"route": "CHENNAI-MADURAI", "bus_type": "luxury"}),
        ("Extra argument", "check_bus_fare",
         {"route": "CHENNAI-MADURAI", "bus_type": "ac", "year": 2026}),
    ]

    for label, tool_name, arguments in examples:
        result = validate_arguments(arguments, SCHEMAS[tool_name])
        print(f"{label:<20} -> {result or 'OK'}")
