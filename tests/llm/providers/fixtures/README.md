# Provider wire fixtures

Every file here comes from a provider's official API reference, fetched on
2026-11-07. Every call ember makes streams, so the tests decode event streams;
a JSON object here is either a stream payload (a Gemini chunk, a Responses
terminal event's response) or the documented content a stream must
reproduce. "Verbatim" files are the bytes between the example's code fences,
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
| `openai/chat_stream_text.sse` | verbatim | https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create (Streaming, response); keeps the page's `:` comment line |
| `openai/chat_stream_tool_calls.sse` | assembled | `chat.completion.chunk` schema at https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events and `stream_options.include_usage` on the create page |
| `openai/responses_function_call.json` | verbatim | https://developers.openai.com/api/reference/resources/responses/methods/create (Functions, response); tests send it as a `response.completed` event's `response` |
| `openai/responses_reasoning.json` | verbatim | same page (Reasoning, response); used the same way |
| `openai/responses_stream_text.sse` | verbatim | same page (Streaming, response); keeps the page's `:` comment line |
| `openai/responses_error_event.json` | verbatim | https://developers.openai.com/api/reference/resources/responses/streaming-events (`error`) |
| `openai/responses_stream_function_call.sse` | assembled | event schemas on the streaming-events page (`response.output_item.added`, `response.reasoning_summary_text.delta`, `response.function_call_arguments.delta`/`.done`, `response.completed`) and the `Reasoning` output item on the create page |

## Anthropic

| File | Kind | Source |
|---|---|---|
| `anthropic/message_reference.json` | verbatim | https://docs.claude.com/en/api/messages (Create a Message, Response 200); its `usage` block is put into a `message_start` event |
| `anthropic/message_thinking_tool_use.json` | assembled | `Message`, `ThinkingBlock`, `RedactedThinkingBlock`, `ToolUseBlock` and `Usage` schemas on https://docs.claude.com/en/api/messages; its `content` is what a replayed turn must send back, byte for byte |
| `anthropic/stream_thinking_tool_use.sse` | assembled | the stream of `message_thinking_tool_use.json`, built with the event and delta types on https://docs.claude.com/en/docs/build-with-claude/streaming (`thinking_delta`, `signature_delta`, `input_json_delta`; usage split between `message_start` and the cumulative `message_delta`) |
| `anthropic/stream_mixed_tool_use.sse` | assembled | the stream of the "A mixed tool_use response" example on https://docs.claude.com/en/docs/build-with-claude/handling-stop-reasons (a `server_tool_use` block and a client `tool_use` block); like the example, it carries no `usage` |
| `anthropic/error_not_found.json` | verbatim | https://docs.claude.com/en/api/errors (Error shapes) |
| `anthropic/stream_text.sse` | verbatim | https://docs.claude.com/en/docs/build-with-claude/streaming (Basic streaming request, response) |
| `anthropic/stream_tool_use.sse` | verbatim | same page (Streaming request with tool use, response); its last event has no trailing blank line |
| `anthropic/stream_thinking.sse` | verbatim | same page (Streaming request with thinking, response); the example carries no `usage` |
| `anthropic/stream_error_overloaded.sse` | verbatim | same page (Error events) |

## Gemini

The Gemini reference documents `GenerateContentResponse`, `Candidate`, `Part`
and `UsageMetadata` as schemas with no example payloads, so every Gemini file
except `error_api_key_invalid.json` is assembled from https://ai.google.dev/api/generate-content and
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
| `gemini/error_api_key_invalid.json` | verbatim, read 2026-11-08: the "Error response format" example on https://ai.google.dev/gemini-api/docs/generate-content/api-errors, an invalid key reported as HTTP 400 `INVALID_ARGUMENT` with a `google.rpc.ErrorInfo` detail whose reason is `API_KEY_INVALID` |
| `gemini/stream_text.sse` | three chunks; the last carries `finishReason` and the final usage |

A `GenerateContentResponse` is also one stream chunk, so the JSON files are
decoded as single-chunk streams. `RetryInfo` in the error test follows
`google/rpc/error_details.proto` in https://github.com/googleapis/googleapis,
which https://google.aip.dev/193 names for error details. The credential
reasons treated as authentication failures, `API_KEY_INVALID` ("the provided
API key is invalid. It may be in a bad format, cannot be found, or has been
expired") and `CREDENTIALS_MISSING`, are defined for the `googleapis.com`
domain in `google/api/error_reason.proto` in the same repository
(https://github.com/googleapis/googleapis/blob/master/google/api/error_reason.proto);
an expired key has no reason of its own and reports `API_KEY_INVALID`.

## Usage fields not on the reference pages' examples

`cache_write_tokens` under `prompt_tokens_details` (Chat) and
`input_tokens_details` (Responses) is documented on the create pages above and
in openai-python's generated types,
https://github.com/openai/openai-python/blob/main/src/openai/types/completion_usage.py
and
https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_usage.py.
