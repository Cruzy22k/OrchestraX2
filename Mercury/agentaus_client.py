"""Minimal Python client for the Agentaus chat-completions API.

Wire format reconstructed from the reference implementation in
agentaus_connector_for_claude_code (agentaus_api_doc.md, config.py, server.py).
The API is OpenAI-compatible chat completions.

Endpoint:  POST {base_url}{path}   e.g. https://agentaus.com.au/api/v1/chat/completions
Auth:      Authorization: Bearer <key>
Body:      {"messages": [...], "stream": true|false, ...}
Reply:     OpenAI-style {"choices": [...], "usage": {...}}

Key notes learned from the bridge:
  * Send "Accept: */*". An explicit "Accept: text/event-stream" gets a 406; the
    `stream` body flag alone decides the response format.
  * To replace Agentaus' built-in persona, include a system-role message AND
    "system_prompt_overwrite": true. Without the flag it prepends its own persona.
  * Streaming returns SSE lines ("data: {...}", ending "data: [DONE]"); buffered
    responses are a single JSON body.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

DEFAULT_BASE_URL = os.environ.get("AGENTAUS_BASE_URL", "https://agentaus.com.au").rstrip("/")
DEFAULT_PATH = os.environ.get("AGENTAUS_PATH", "/api/v1/chat/completions")
DEFAULT_API_KEY = os.environ.get("AGENTAUS_API_KEY", "")


@dataclass
class AgentausError(Exception):
    """Raised for non-2xx responses and network failures."""

    status: int | None = None
    message: str = ""


def chat_url(base_url: str = DEFAULT_BASE_URL, path: str = DEFAULT_PATH) -> str:
    return f"{base_url}{path}"


class AgentausClient:
    """A small, stateless client for one chat-completions call at a time."""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        path: str = DEFAULT_PATH,
        *,
        timeout: float = 120.0,
        system_prompt_overwrite: bool = True,
    ) -> None:
        api_key = input("Enter your Agentaus API key: ") if not api_key else api_key
        self.api_key = api_key
        self.base_url = base_url
        self.path = path
        self.timeout = timeout
        self.system_prompt_overwrite = system_prompt_overwrite

    # -- construction helpers -------------------------------------------------

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Agentaus answers 406 to "Accept: text/event-stream"; let the
            # `stream` body flag alone decide the format.
            "Accept": "*/*",
        }

    def system(self, text: str) -> list[dict]:
        """System-role messages, with the flag that replaces Agentaus' persona."""
        return [{"role": "system", "content": text}]

    def tool(self, name: str, description: str, parameters: dict) -> dict:
        """One OpenAI-style function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    # -- the call -------------------------------------------------------------

    def _payload(self, messages: list[dict], *, tools: list | None = None,
                 stream: bool = False, **extra: Any) -> dict:
        body: dict = {"messages": messages, "stream": stream}
        if tools:
            body["tools"] = tools
            body.setdefault("tool_choice", "auto")
        # Replace the default persona whenever the caller supplies a system message.
        if any(m.get("role") == "system" for m in messages) and self.system_prompt_overwrite:
            body["system_prompt_overwrite"] = True
        body.update(extra)
        return body

    def chat(self, messages: list[dict], *, tools: list | None = None,
             **extra: Any) -> dict:
        """One buffered (non-streaming) call. Returns the full response dict."""
        body = self._payload(messages, tools=tools, stream=False, **extra)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(chat_url(self.base_url, self.path),
                               headers=self.headers(), json=body)
        if resp.status_code != 200:
            raise AgentausError(status=resp.status_code,
                                message=resp.text[:300])
        return resp.json()

    def reply_text(self, messages: list[dict], *, tools: list | None = None,
                   **extra: Any) -> str:
        """Convenience: return just the assistant's text content."""
        data = self.chat(messages, tools=tools, **extra)
        choice = (data.get("choices") or [{}])[0]
        return (choice.get("message") or {}).get("content") or ""

    def tool_calls(self, data: dict) -> list[dict]:
        """Pull `tool_calls` out of a response: [{id, name, arguments(dict)}]."""
        choice = (data.get("choices") or [{}])[0]
        calls = ((choice.get("message") or {}).get("tool_calls")) or []
        parsed = []
        for call in calls:
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                args = {"__raw_arguments__": raw}
            parsed.append({
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            })
        return parsed

    def stream(self, messages: list[dict], *, tools: list | None = None,
               **extra: Any) -> Iterable[dict]:
        """Yield each SSE chunk dict from a streamed call.

        A chunk is the parsed body of one `data:` line, e.g.
        {"choices": [{"delta": {"content": "..."}}]}. Callers accumulate
        `delta.content` (and `delta.tool_calls`) themselves.
        """
        body = self._payload(messages, tools=tools, stream=True, **extra)
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", chat_url(self.base_url, self.path),
                               headers=self.headers(), json=body) as resp:
                if resp.status_code != 200:
                    raise AgentausError(status=resp.status_code,
                                        message=resp.text[:300])
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict):
                        yield chunk
