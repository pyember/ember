# Feature Parity Assessment: OpenAI Responses vs. Claude Messages vs. Gemini API

**Prepared:** October 12, 2025  
**Author:** Codex analysis for design kickoff

## Objective
Outline how easily we can achieve feature parity between OpenAI's Responses API and comparable endpoints from Anthropic Claude and Google Gemini, focusing on request/response structure, tool invocation, hosted capabilities, and execution workflows. The goal is to inform a forthcoming design review.

## Core Parity Areas
- **Structured outputs:** All three platforms accept schema-constrained responses—OpenAI enforces JSON modes via `response_format`, Claude accepts `json_schema` in tool specs, and Gemini uses `responseSchema`—so a shared abstraction for validating structured replies is feasible. citeturn0search0turn0search6turn1search1turn1search7
- **Tool/function calling:** Each API supports declaring tools/functions and returns machine-readable tool invocations, allowing thin translation layers between provider field names and call semantics. citeturn0search0turn1search1turn1search7
- **Streaming responses:** OpenAI Responses supports incremental streaming, Claude offers streamable message deltas, and Gemini exposes `streamGenerateContent`, so shared SSE/WebSocket handling covers all providers. citeturn0search6turn1search3
- **Conversation state management:** All APIs treat conversations as stateless arrays of messages/content blocks, meaning parity largely requires normalizing role labels and content types. citeturn0search0turn1search1turn1search7

## Parity Gaps & Provider-Specific Capabilities
- **Hosted tool ecosystems:** OpenAI bundles Code Interpreter, File Search, Web Search, and remote MCP connectors inside the Responses API; Claude offers beta-managed web search, computer use, and code execution but requires explicit headers; Gemini exposes Workspace computer use and Google Search grounding on a per-request basis. These require provider-specific adapters and capability flags. citeturn0search0turn1search4turn1search8
- **Background and agentic execution:** OpenAI background mode lets runs finish after the client disconnects, Claude's extended thinking delivers hidden reasoning blocks that must be replayed, and Gemini Live sessions keep bi-directional state. Aligning them demands a unified job orchestration contract. citeturn0search6turn1search6turn1search2
- **Multimodal ingestion:** OpenAI routes high-fidelity audio via the Realtime API and project file uploads for Responses, Claude's Files API handles images/PDFs for model contexts, and Gemini 2.5 handles text, images, audio, video, and PDFs directly, so media handling must stay adapter-aware. citeturn0search0turn1search5turn1search1

## Implementation Considerations
- **Neutral request schema:** Define a provider-agnostic contract (`messages`, `tools`, `response_mode`) and compile request payloads per provider to keep product code stable. citeturn0search0turn1search1turn1search7
- **Streaming abstraction:** Build a pluggable transport layer that supports SSE and WebSockets, capturing chunk assembly, latency metrics, and backpressure uniformly. citeturn0search6turn1search3
- **Capability detection:** Surface advanced features (code execution, hosted search, computer use) through explicit capability probes and degrade gracefully when absent. citeturn0search0turn1search4turn1search8
- **Long-running job orchestration:** Normalize background job IDs, resume tokens, and keep-alive semantics so scheduling and retry logic do not diverge per provider. citeturn0search6turn1search6turn1search2
- **Media pipeline:** Standardize upload staging, size validation, and content-type normalization while allowing provider-specific optimizations (e.g., Gemini video embeddings vs. Claude image reasoning). citeturn0search0turn1search5turn1search1

## Suggested Next Steps
1. Prototype a shared text+tool call flow across all three providers to validate the translation layer.
2. Add automated capability probes/tests that alert when a provider removes or changes beta features (e.g., Claude computer use headers, Gemini computer use policies).
3. Design a unified background job interface with hooks for OpenAI background runs, Claude extended thinking replay, and Gemini Live session lifecycle management.
