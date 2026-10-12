
# Day 8 — Agentic RAG Task Results

## Part A — Ingestion and retrieval

- Documents loaded: [actual output; expected 6]
- Chunks stored: [actual output]
- Embedding provider: [actual provider]
- Vector length: [actual output]
- Placement query top result: [actual filename]
- Placement query score: [actual score]
- France query score: [actual score]
- MAX_DISTANCE selected: [measured threshold]

## Part B — Exam eligibility tool

| Attendance | Expected result | Actual result | Passed? |
|---:|---|---|---|
| 82% | ELIGIBLE | [fill] | [Y/N] |
| 70% | CONDONATION, Rs. 500 per course | [fill] | [Y/N] |
| 50% | NOT ELIGIBLE | [fill] | [Y/N] |
| 120% | Error message | [fill] | [Y/N] |

## Part C and D — Agent test results

| # | Tools called (in order) | Agent's answer (short) | Correct? (Y/N) | If N, why? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

## Model and embedding details

- Model used: [actual model]
- Embedding model: [actual embedding model]
- MAX_DISTANCE: [actual value]

## Observations

- The placement policy was ingested and retrieved from placement_policy.md.
- The eligibility tool was tested with 82%, 70%, 50%, and 120%.
- The relevance guard was tested using placement and France questions.
- Questions 1–5 were run in the same task-run thread.
- Record any incorrect answers or tool-selection failures honestly.
