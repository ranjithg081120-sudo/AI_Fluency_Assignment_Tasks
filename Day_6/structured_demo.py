
"""Day 6: Compare free text, JSON mode, and JSON Schema mode."""

import json
import sys
from pathlib import Path

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


EXTRACTION_TEXT = """
Extract the travel details from this sentence:

"I need an AC bus from Chennai to Madurai on 15 November 2026.
There are 2 passengers and my budget is 1200 rupees per person."

Return:
- origin: departure city
- destination: arrival city
- bus_type: ordinary, express, or ac
- travel_date: YYYY-MM-DD
- passengers: integer
- budget_per_person: numeric amount in rupees
"""


TRIP_SCHEMA = {
    "type": "object",
    "properties": {
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "bus_type": {
            "type": "string",
            "enum": ["ordinary", "express", "ac"],
        },
        "travel_date": {"type": "string"},
        "passengers": {"type": "integer"},
        "budget_per_person": {"type": "number"},
    },
    "required": [
        "origin",
        "destination",
        "bus_type",
        "travel_date",
        "passengers",
        "budget_per_person",
    ],
    "additionalProperties": False,
}


def show_result(mode, raw_reply):
    """Display the original response and any parsed JSON."""
    print("\n" + "=" * 65)
    print(f"MODE: {mode}")
    print("-" * 65)
    print("RAW MODEL REPLY:")
    print(raw_reply)

    try:
        parsed = json.loads(raw_reply)
    except (json.JSONDecodeError, TypeError):
        print("\nPARSED RESULT:")
        print("The response is not valid standalone JSON.")
        return

    print("\nPARSED RESULT:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    # Basic checks for the extracted fields.
    if isinstance(parsed, dict):
        missing = [
            key for key in TRIP_SCHEMA["required"]
            if key not in parsed
        ]
        extra = [
            key for key in parsed
            if key not in TRIP_SCHEMA["properties"]
        ]

        if missing:
            print(f"\nValidation note: missing fields: {missing}")
        if extra:
            print(f"Validation note: unexpected fields: {extra}")
        if not missing and not extra:
            print("\nRequired-field check: passed.")


def run_mode(mode, response_format=None):
    """Run one experiment and continue if the provider rejects the mode."""
    print(f"\nRunning: {mode}")

    if client is None or MODEL is None:
        print(
            "ERROR: Configuration is unavailable. "
            f"Details: {CONFIG_ERROR}"
        )
        return

    messages = [
        {
            "role": "system",
            "content": (
                "Extract travel details accurately. "
                "Do not invent values. Follow the requested output format."
            ),
        },
        {"role": "user", "content": EXTRACTION_TEXT},
    ]

    request = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 500,
    }

    if response_format is not None:
        request["response_format"] = response_format

    try:
        response = client.chat.completions.create(**request)

        if not response.choices:
            print("ERROR: Provider returned no response choices.")
            return

        message = response.choices[0].message
        raw_reply = message.content or ""
        show_result(mode, raw_reply)

    except Exception as error:
        print(f"MODE FAILED: {mode}")
        print(f"{type(error).__name__}: {error}")
        print(
            "This provider/model may not support the requested "
            "output format. The remaining tests will continue."
        )


def main():
    print("DAY 6: STRUCTURED OUTPUT COMPARISON")
    print(f"Model: {MODEL if MODEL else 'Not configured'}")
    print("Using the same extraction question in all modes.")

    # Experiment 1: Natural-language output.
    run_mode("Free text")

    # Experiment 2: JSON object output.
    run_mode(
        "JSON mode",
        response_format={"type": "json_object"},
    )

    # Experiment 3: JSON Schema-constrained output.
    run_mode(
        "JSON Schema mode",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "trip_details",
                "strict": True,
                "schema": TRIP_SCHEMA,
            },
        },
    )

    print("\n" + "=" * 65)
    print("Structured-output comparison finished.")
    print("Record actual successes and errors in analysis.md.")


if __name__ == "__main__":
    main()
