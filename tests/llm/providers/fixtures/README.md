# Provider wire fixtures

Every file here comes from a provider's official API reference, fetched on
2026-11-07. "Verbatim" files are the bytes between the example's code fences,
unchanged. "Assembled" files were written field by field from the reference's
schema for that object, because the page shows the schema but no example
payload; their values are illustrative and their field names, nesting and
enum values are the documented ones. No file is recorded traffic.

Tests that need a variant (a different finish reason, a missing usage block)
load one of these files and change the one documented field the test is
about, so each variant's provenance is the file it starts from.

## OpenAI

| File | Kind | Source |
|---|---|---|
| `openai/chat_text.json` | verbatim | https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create (Example, response) |
| `openai/chat_tool_call.json` | verbatim | same page (Functions, response) |
| `openai/chat_stream_text.sse` | verbatim | same page (Streaming, response); keeps the page's `:` comment line |
| `openai/chat_stream_tool_calls.sse` | assembled | `chat.completion.chunk` schema at https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events and `stream_options.include_usage` on the create page |
| `openai/responses_function_call.json` | verbatim | https://developers.openai.com/api/reference/resources/responses/methods/create (Functions, response) |
| `openai/responses_reasoning.json` | verbatim | same page (Reasoning, response) |
| `openai/responses_stream_text.sse` | verbatim | same page (Streaming, response); keeps the page's `:` comment line |
| `openai/responses_error_event.json` | verbatim | https://developers.openai.com/api/reference/resources/responses/streaming-events (`error`) |
| `openai/responses_stream_function_call.sse` | assembled | event schemas on the streaming-events page (`response.output_item.added`, `response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`/`.done`, `response.completed`) and the `Reasoning` output item on the create page |

## Anthropic

| File | Kind | Source |
|---|---|---|
| `anthropic/message_reference.json` | verbatim | https://docs.claude.com/en/api/messages (Create a Message, Response 200) |
| `anthropic/message_end_turn.json` | verbatim | https://docs.claude.com/en/docs/build-with-claude/handling-stop-reasons (Example response) |
| `anthropic/message_mixed_tool_use.json` | verbatim | same page (A mixed tool_use response); the example has no `usage` |
| `anthropic/message_thinking_tool_use.json` | assembled | `Message`, `ThinkingBlock`, `RedactedThinkingBlock`, `ToolUseBlock` and `Usage` (with `output_tokens_details.thinking_tokens`) schemas on https://docs.claude.com/en/api/messages; signature values are copied from the streaming page |
| `anthropic/error_not_found.json` | verbatim | https://docs.claude.com/en/api/errors (Error shapes) |
| `anthropic/stream_text.sse` | verbatim | https://docs.claude.com/en/docs/build-with-claude/streaming (Basic streaming request, response) |
| `anthropic/stream_tool_use.sse` | verbatim | same page (Streaming request with tool use, response); its last event has no trailing blank line |
| `anthropic/stream_thinking.sse` | verbatim | same page (Streaming request with thinking, response); the example carries no `usage` |
| `anthropic/stream_error_overloaded.sse` | verbatim | same page (Error events) |

## Gemini

The Gemini reference documents `GenerateContentResponse`, `Candidate`, `Part`
and `UsageMetadata` as schemas with no example payloads, so every Gemini file
is assembled from https://ai.google.dev/api/generate-content and
https://ai.google.dev/api/generate-content/stream-generate-content (one
`GenerateContentResponse` per `data:` line). The error shape (`code`,
`message`, `status`) and the `RESOURCE_EXHAUSTED` status are from
https://ai.google.dev/gemini-api/docs/troubleshooting.

| File | Covers |
|---|---|
| `gemini/text.json` | text parts, a `thoughtSignature` on a text part, cached and thought token counts |
| `gemini/function_call.json` | a `thought: true` summary part, function calls with and without `id`, signatures on parts, tool-use prompt tokens |
| `gemini/prompt_blocked.json` | `promptFeedback.blockReason` with no candidates |
| `gemini/error_resource_exhausted.json` | the Google error body |
| `gemini/stream_text.sse` | three chunks; the last carries `finishReason` and the final usage |
