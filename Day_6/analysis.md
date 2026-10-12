# Day 6 Task: Reliable Tool Calling

**Scenario:** College Travel Assistant

## 1. Explanation of Concepts

### Chat Completions format

Chat Completions sends a model a list of messages with roles such as system and user. The request may include a model name, messages, tools, temperature and maximum output tokens. The response contains choices, a message, and a finish reason. A `stop` finish reason means the answer is complete; `length` means the output was truncated; and `tool_calls` means the model requested a tool. The assistant's message content can be empty when it requests tools because the tool request is carried separately in `tool_calls`.

### OpenAI-compatible servers

An OpenAI-compatible server implements an API interface that the OpenAI Python client can call. The same client can connect to different providers by changing its base URL, API key and model configuration. Compatibility does not guarantee that every provider supports every feature, including strict JSON Schema outputs or identical tool-calling behaviour.

### Streaming and API styles

Streaming returns parts of a response as they are generated, reducing perceived waiting time. Inside an agent loop, non-streaming is simpler because the program can inspect the complete response and all tool calls before acting. The Responses API uses a different interaction format from Chat Completions. This task uses Chat Completions to match the lab's existing code and provider setup.

### The tool-calling cycle

The model receives the conversation and tool definitions, requests a tool with arguments, and sends the request to the application. The application parses the arguments, checks the tool and schema, executes the function and sends its result back with the matching tool-call ID. The model then produces the final answer. The model does not execute Python functions itself; the application does. Generated arguments must therefore be validated before execution.

### Tool schemas and tool choice

Descriptions explain when to use a tool. Properties specify argument names and types. Required fields prevent omissions, enums restrict values, and `additionalProperties: false` rejects invented arguments. Strict schema mode requests closer adherence to the declared structure, where supported. `auto` lets the model decide whether to call a tool, `none` prevents tool calls, `required` requests a tool call, and a named function selects a particular tool. For the travel assistant, `auto` is appropriate because some questions need tools and others do not.

### Structured outputs

JSON mode aims to produce valid JSON but does not necessarily enforce a particular set of fields. JSON Schema mode specifies the required structure and permitted values. Tool calling is appropriate when the application must look up a fare or perform an action. Structured outputs are appropriate when extracting trip details into fields. Even perfectly shaped JSON can contain incorrect facts.

### Parallel tool calls

A model can request more than one tool in the same response. The agent must process every call and append exactly one result for each `tool_call_id`. Leaving a tool call unanswered can break the next API request. The final multi-part travel question is intended to test fare lookup and weather advice in one conversation, although the model may choose whether to call them in parallel.

### Failure handling and repair

The agent handles invalid JSON, unknown tools, missing arguments, incorrect types, invalid enum values, invented arguments, truncated responses and repeated identical calls. Validation errors are returned as readable strings. Truncated responses are retried with a larger token limit, while malformed tool calls receive an error result so the model can attempt to recover. Repeated calls are stopped after the configured limit, and the loop also has a maximum step count.

### Fault injection

Fault injection deliberately supplies malformed tool calls to the handler. It is repeatable and does not require a live model because it tests the application's defensive code directly. A schema can reject a missing field or an invalid enum, but it cannot guarantee that a syntactically valid request is factually correct.

## 2. Comparison: Tool Calling and Structured Outputs

| Basis | Tool calling | Structured outputs |
|---|---|---|
| Model's task | Request an application function | Return data in a specified shape |
| Who acts | Application executes the function | Model returns the data; application consumes it |
| Shape control | Tool schema and application validation | JSON mode or JSON Schema |
| Remaining risks | Incorrect values, bad choices, runtime failures | Correct shape but incorrect facts |
| Main guard | Validate arguments before execution | Parse and validate returned data |
| Provider support | Varies by model and server | Varies by model and server |
| Fetching a fare | Preferred; it invokes a lookup function | Not sufficient by itself |
| Extracting trip details | Possible, but unnecessary if no action is required | Preferred for predictable fields |

## 3. Implementation

The implementation uses `tools_v2.py` for tool functions and shared schemas, `validate.py` for argument validation, `robust_agent.py` for the agent loop, `inject_faults.py` for deliberate fault tests, and `structured_demo.py` for the three output modes.

The travel scenario uses a small fixed route-fare table and basic weather advice. The fare values are sample values for demonstrating tool calling, not live ticket prices.

## 4. Observations

### 4.1 Fault injection

| Fault | Expected result | Actual result | Continued without uncaught error? |
|---|---|---|---|
| Valid call | Fare result | Fill after run | Fill after run |
| Invalid JSON | Parse error | Fill after run | Fill after run |
| Unknown tool | Available tools listed | Fill after run | Fill after run |
| Missing argument | Missing field named | Fill after run | Fill after run |
| Wrong type | Expected type named | Fill after run | Fill after run |
| Invalid enum | Allowed values listed | Fill after run | Fill after run |
| Extra argument | Extra field rejected | Fill after run | Fill after run |
| Wrong tool capitalization | Unknown tool | Fill after run | Fill after run |
| JSON array instead of object | Object required | Fill after run | Fill after run |

### 4.2 Agent behaviour

Run the five questions in `robust_agent.py`. Record the number of steps, tools requested, final answer, whether multiple tools arrived together, and whether a truncated response occurred. Do not assume the model used parallel calls merely because the question mentioned two tasks.

### 4.3 Structured outputs

Record the raw response and whether each mode returned valid JSON, the intended fields, or an unsupported-feature error. Results may vary by provider and model.

## 5. Suitability and Conclusion

Tool calling is best when the application must retrieve information or execute a function. Structured outputs are best when an application needs predictable fields from a sentence. Both require application-side checks: schemas reduce malformed inputs but do not establish factual correctness. Deliberate fault injection tests error-handling paths consistently, while real-model runs reveal additional issues such as skipped tools or unexpected output.

**Before submission:** Replace all observation placeholders with actual results from your runs. Do not claim a test passed unless the output supports it.