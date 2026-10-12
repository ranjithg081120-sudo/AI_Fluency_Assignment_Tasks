
"""Day 6: travel assistant tools and shared JSON schemas."""

ROUTES = {
    "CHENNAI-MADURAI": 450,
    "CHENNAI-COIMBATORE": 350,
    "CHENNAI-BENGALURU": 500,
}

FARE_MULTIPLIERS = {
    "ordinary": 1.0,
    "express": 1.5,
    "ac": 2.0,
}


def check_bus_fare(route: str, bus_type: str) -> str:
    """Look up the estimated fare for a route and bus type."""
    route = route.strip().upper()
    bus_type = bus_type.lower()

    if route not in ROUTES:
        return f"Unknown route: {route}. Valid routes: {', '.join(ROUTES)}"

    if bus_type not in FARE_MULTIPLIERS:
        return "Invalid bus type. Choose ordinary, express, or ac."

    fare = ROUTES[route] * FARE_MULTIPLIERS[bus_type]
    return f"Estimated fare: Rs. {fare:.0f}"


def check_weather_advice(city: str, condition: str) -> str:
    """Give basic travel advice for a supplied weather condition."""
    condition = condition.lower()

    if condition == "sunny":
        advice = "Carry water and sun protection."
    elif condition == "rainy":
        advice = "Carry an umbrella and allow extra travel time."
    else:
        advice = "Check local weather updates before departure."

    return f"{city.title()}: {advice}"


TOOL_FUNCTIONS = {
    "check_bus_fare": check_bus_fare,
    "check_weather_advice": check_weather_advice,
}

SCHEMAS = {
    "check_bus_fare": {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": list(ROUTES.keys()),
            },
            "bus_type": {
                "type": "string",
                "enum": list(FARE_MULTIPLIERS.keys()),
            },
        },
        "required": ["route", "bus_type"],
        "additionalProperties": False,
    },
    "check_weather_advice": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "condition": {
                "type": "string",
                "enum": ["sunny", "rainy", "cloudy"],
            },
        },
        "required": ["city", "condition"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Look up the estimated bus fare for a route and bus type."
                if name == "check_bus_fare"
                else "Give basic travel advice for a city and weather condition."
            ),
            "parameters": schema,
        },
    }
    for name, schema in SCHEMAS.items()
]
