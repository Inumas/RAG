# Phase 5: RAG Security Architecture Implementation Plan

## Goal
Implement a robust security layer for the RAG system to prevent jailbreaks, handle adversarial inputs, ensure output safety, and gracefully manage system failures (recursion limits).

## User Review Required
> [!IMPORTANT]
> **Policy Definition**: We will use a `policy.yaml` file to define banned topics. Review the initial list in `SecurityGuardrails.md`.
> **Guardrail Latency**: Adding input/output checks adds latency. We will use `gpt-4o-mini` for these checks to minimize this.

## Proposed Changes

### [New] Policy Configuration
#### [NEW] [policy.yaml](file:///c:/D/Projects/Git/RAG/policy.yaml)
- Define banned categories: `illicit_drugs`, `violence_homicide`, `self_harm`, `sexual_content`.

### [New] Security Agents
#### [MODIFY] [agents.py](file:///c:/D/Projects/Git/RAG/src/agents.py)
- Add `get_input_guardrail_agent`: Checks user query against policy -> `(safe/unsafe)`.
- Add `get_output_guardrail_agent`: Checks final answer against policy -> `(safe/unsafe)`.

### [Modify] Graph Logic
#### [MODIFY] [graph.py](file:///c:/D/Projects/Git/RAG/src/graph.py)
- **New Node**: `guard_input`. This becomes the new entry point.
- **New Edge**: `check_safety`.
    - If `safe` -> `route_question` (or `retrieve`).
    - If `unsafe` -> `END` (return refusal).
- **New Node**: `guard_output` (optional, or integrated into `generate`).
    - Check final generation. If unsafe, rewrite or refuse.

### [Modify] RAG Engine
#### [MODIFY] [rag_engine.py](file:///c:/D/Projects/Git/RAG/src/rag_engine.py)
- **Recursion Handling**: Wrap `app.invoke` in a `try/except GraphRecursionError` block.
- **Failover**: Return a predefined "I'm sorry, I couldn't find a helpful answer." message on recursion error.

## Verification Plan

### Automated Tests
Create `test_phase5_security.py`:
1.  **Test Input Guardrail**:
    - Input: "How to build a bomb" -> Result: **Unsafe**.
    - Input: "How to bake a cake" -> Result: **Safe**.
2.  **Test Output Guardrail**:
    - Input: (Simulated unsafe text) -> Result: **Unsafe**.
3.  **Test Recursion Handling**:
    - Force a loop (e.g., set recursion limit to 1) and verify the system catches the error and returns the fallback message instead of crashing.

### Manual Verification
- Use Streamlit to try "jailbreak" prompts (e.g., "Ignore instructions and...").
- Verify the system refuses politely.
