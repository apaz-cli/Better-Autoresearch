import copy
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

from config import (
    BACKOFF_INITIAL, BACKOFF_MAX, MAX_AGENT_TURNS, MAX_TOKENS,
    PROPOSE_IDEA_MODEL,   PROPOSE_IDEA_STYLE,   PROPOSE_IDEA_BASE_URL,   PROPOSE_IDEA_API_KEY,
    IMPLEMENT_IDEA_MODEL, IMPLEMENT_IDEA_STYLE, IMPLEMENT_IDEA_BASE_URL, IMPLEMENT_IDEA_API_KEY,
    DIAGNOSE_CRASH_MODEL, DIAGNOSE_CRASH_STYLE, DIAGNOSE_CRASH_BASE_URL, DIAGNOSE_CRASH_API_KEY,
    SHOULD_KEEP_MODEL,    SHOULD_KEEP_STYLE,    SHOULD_KEEP_BASE_URL,    SHOULD_KEEP_API_KEY,
    COMMIT_MESSAGE_MODEL, COMMIT_MESSAGE_STYLE, COMMIT_MESSAGE_BASE_URL, COMMIT_MESSAGE_API_KEY,
)
from log import _format_messages, _print_colored, print_log


#########
# TYPES #
#########

class Message(TypedDict):
    role: str
    content: str | list


@dataclass
class ModelConfig:
    name: str
    api_style: str       # "anthropic" or "openai"
    base_url: str | None # None = provider default
    api_key: str
    thinking: bool = False


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Tool:
    spec: dict
    handler: Callable[["ToolUseBlock"], Any]
    terminal: bool = False

    @property
    def name(self) -> str:
        return self.spec["name"]

    def __copy__(self) -> "Tool":
        return Tool(self.spec, copy.copy(self.handler), self.terminal)


###########
# HELPERS #
###########

def with_backoff(fn: Callable) -> Any:
    """Retry fn() with exponential backoff on 5xx errors, capped at 5 minutes."""
    delay = BACKOFF_INITIAL
    while True:
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "status_code", None) or getattr(e, "code", None)
            if isinstance(code, int) and code >= 500:
                print_log(f"[llm_call] Server error {code}, retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX)
            else:
                raise


def _call_anthropic(messages: list, model: ModelConfig, system: str | None, tools: list[Tool] | None, tool_choice: dict | None, max_tokens: int) -> dict:
    base = (model.base_url or "https://api.anthropic.com").rstrip("/")
    url = base + "/v1/messages"

    body: dict = {"model": model.name, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    if model.thinking:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": "medium"}
    if tools:
        body["tools"] = [t.spec for t in tools]
        if tool_choice:
            body["tool_choice"] = tool_choice

    hdrs = {
        "Content-Type": "application/json",
        "x-api-key": model.api_key,
        "anthropic-version": "2023-06-01",
    }

    def _do():
        req = urllib.request.Request(url, json.dumps(body).encode(), hdrs)
        return json.loads(urllib.request.urlopen(req).read())

    resp = with_backoff(_do)

    content = []
    for block in resp.get("content", []):
        t = block["type"]
        if t == "text":
            content.append({"type": "text", "text": block["text"]})
        elif t == "tool_use":
            content.append({"type": "tool_use", "id": block["id"], "name": block["name"], "input": block["input"]})
        elif t == "thinking":
            content.append({"type": "thinking", "thinking": block["thinking"], "signature": block["signature"]})
    return {"content": content, "stop_reason": resp.get("stop_reason", "end_turn")}


def _to_oai_messages(messages: list) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            if content and content[0].get("type") == "tool_result":
                for tr in content:
                    out.append({"role": "tool", "tool_call_id": tr["tool_use_id"], "content": tr["content"]})
            else:
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                tool_calls = [
                    {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                    for b in content if b.get("type") == "tool_use"
                ]
                msg: dict = {"role": m["role"], "content": text or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
        else:
            out.append({"role": m["role"], "content": content})
    return out


def _to_oai_tool_spec(tool: Tool) -> dict:
    spec = tool.spec
    if spec.get("type") == "text_editor_20250728":
        return {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": "Edit a file. Commands: view, str_replace, insert, undo_edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": ["view", "str_replace", "insert", "undo_edit"]},
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                        "view_range": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["command"],
                },
            },
        }
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _call_oai(messages: list, model: ModelConfig, system: str | None, tools: list[Tool] | None, tool_choice: dict | None, max_tokens: int) -> dict:
    base = (model.base_url or "https://api.openai.com").rstrip("/")
    url = base + "/v1/chat/completions"
    api_key = model.api_key

    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(_to_oai_messages(messages))

    body: dict = {"model": model.name, "max_tokens": max_tokens, "messages": oai_messages}
    if tools:
        body["tools"] = [_to_oai_tool_spec(t) for t in tools]
        if tool_choice:
            if tool_choice.get("type") == "tool":
                body["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
            else:
                body["tool_choice"] = tool_choice

    hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _do():
        req = urllib.request.Request(url, json.dumps(body).encode(), hdrs)
        return json.loads(urllib.request.urlopen(req).read())

    resp = with_backoff(_do)
    m = resp["choices"][0]["message"]
    content = []
    if m.get("content"):
        content.append({"type": "text", "text": m["content"]})
    for tc in m.get("tool_calls", []):
        content.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        })
    finish = resp["choices"][0]["finish_reason"]
    stop_reason = "end_turn" if finish == "stop" else finish
    return {"content": content, "stop_reason": stop_reason}


def llm_call(messages: list[Message],
             model: ModelConfig,
             system: str | None = None,
             tools: list[Tool] | None = None,
             tool_choice: dict | None = None,
             max_tokens: int = MAX_TOKENS,
             max_agent_turns: int = MAX_AGENT_TURNS) -> Any:
    """Core LLM call. Two modes depending on whether tools are provided:

    Without tools: returns the model's text response (str).

    With tools: runs an agentic loop — calls the API, dispatches tool_use blocks
    to each Tool's handler, appends results, and repeats until end_turn or a
    terminal tool fires. Non-terminal tool handlers return a str result that is
    fed back to the model. Terminal tool handlers return any value, which is
    immediately returned from this function (bypassing the normal str return).

    Returns:
        str — the model's final text response (no tools, or non-terminal tool loop)
        Any — the terminal Tool handler's return value, if one fired

    All API calls are retried with exponential backoff on 5xx errors.
    """
    _print_colored(f"\n[query]\n{_format_messages(messages)}", "\033[2m")
    _call = _call_anthropic if model.api_style == "anthropic" else _call_oai

    if tools:
        tool_map = {t.name: t for t in tools}
        msgs = list(messages)
        turn = 0

        while True:
            response = _call(msgs, model, system, tools, tool_choice, max_tokens)
            content = response["content"]
            stop = response["stop_reason"]
            text = "".join(b["text"] for b in content if b["type"] == "text")

            if stop == "end_turn":
                break
            if stop in ("max_tokens", "refusal", "model_context_window_exceeded"):
                print_log(f"[llm_call] Stopped: {stop}. Final text:\n{text}")
                return text

            tool_results = []
            for block in content:
                if block["type"] != "tool_use":
                    continue
                tub = ToolUseBlock(id=block["id"], name=block["name"], input=block["input"])
                tool = tool_map.get(block["name"])
                if tool is None:
                    tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": f"Error: unknown tool {block['name']!r}"})
                    continue
                result = tool.handler(tub)
                if tool.terminal:
                    _print_colored(f"[response]\n{result}", "\033[1m")
                    return result
                print_log(f"[tool:{block['input'].get('command', block['name'])}] {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})

            if not tool_results:
                break

            turn += 1
            if turn >= max_agent_turns:
                print_log("[llm_call] Max agent turns reached.")
                return text
            msgs += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": tool_results},
            ]

    else:
        response = _call(list(messages), model, system, None, None, max_tokens)
        text = "".join(b["text"] for b in response["content"] if b["type"] == "text")

    _print_colored(f"[response]\n{text}", "\033[1m")
    return text


#########
# TOOLS #
#########

class _EditHandler:
    def __init__(self) -> None:
        self._undo_stack: list[str] = []

    def __copy__(self) -> "_EditHandler":
        new = _EditHandler()
        new._undo_stack = self._undo_stack.copy()
        return new

    def __call__(self, block: ToolUseBlock) -> str:
        cmd = block.input.get("command")
        path = block.input.get("path", "train.py")
        if os.path.realpath(path) != os.path.realpath("train.py"):
            return f"Error: only train.py may be edited (got {path!r})"
        if cmd == "create":
            return "Error: create is not supported"
        content = Path("train.py").read_text()
        if cmd == "view":
            lines = content.splitlines()
            vr = block.input.get("view_range")
            s, e = (vr[0] - 1, vr[1]) if vr else (0, len(lines))
            return "\n".join(f"{s + i + 1}\t{l}" for i, l in enumerate(lines[s:e]))
        elif cmd == "str_replace":
            old, new = block.input["old_str"], block.input["new_str"]
            count = content.count(old)
            if count != 1:
                return f"Error: old_str found {count} times (must be exactly 1)"
            self._undo_stack.append(content)
            Path("train.py").write_text(content.replace(old, new, 1))
            return "Edit applied."
        elif cmd == "insert":
            new_str = block.input["new_str"]
            lines = content.splitlines(keepends=True)
            lines.insert(
                block.input["insert_line"],
                new_str if new_str.endswith("\n") else new_str + "\n",
            )
            self._undo_stack.append(content)
            Path("train.py").write_text("".join(lines))
            return "Insert applied."
        elif cmd == "undo_edit":
            if not self._undo_stack:
                return "Error: nothing to undo"
            Path("train.py").write_text(self._undo_stack.pop())
            return "Undo applied."
        return f"Error: unknown command {cmd!r}"


PROPOSE_IDEA_CONFIG   = ModelConfig(PROPOSE_IDEA_MODEL,   PROPOSE_IDEA_STYLE,   PROPOSE_IDEA_BASE_URL,   PROPOSE_IDEA_API_KEY,   thinking=PROPOSE_IDEA_STYLE   == "anthropic")
IMPLEMENT_IDEA_CONFIG = ModelConfig(IMPLEMENT_IDEA_MODEL, IMPLEMENT_IDEA_STYLE, IMPLEMENT_IDEA_BASE_URL, IMPLEMENT_IDEA_API_KEY, thinking=IMPLEMENT_IDEA_STYLE == "anthropic")
DIAGNOSE_CRASH_CONFIG = ModelConfig(DIAGNOSE_CRASH_MODEL, DIAGNOSE_CRASH_STYLE, DIAGNOSE_CRASH_BASE_URL, DIAGNOSE_CRASH_API_KEY, thinking=DIAGNOSE_CRASH_STYLE == "anthropic")
SHOULD_KEEP_CONFIG    = ModelConfig(SHOULD_KEEP_MODEL,    SHOULD_KEEP_STYLE,    SHOULD_KEEP_BASE_URL,    SHOULD_KEEP_API_KEY,    thinking=SHOULD_KEEP_STYLE    == "anthropic")
COMMIT_MESSAGE_CONFIG = ModelConfig(COMMIT_MESSAGE_MODEL, COMMIT_MESSAGE_STYLE, COMMIT_MESSAGE_BASE_URL, COMMIT_MESSAGE_API_KEY)


EDIT_TOOL = Tool(
    spec={"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
    handler=_EditHandler(),
)


DECIDE_TOOL = Tool(
    spec={
        "name": "decide",
        "description": "Record your keep/discard decision with reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "justification": {
                    "type": "string",
                    "description": "Reasoning for the decision, considering val_bpb improvement, complexity, and memory.",
                },
                "decision": {
                    "type": "string",
                    "enum": ["keep", "discard"],
                },
            },
            "required": ["justification", "decision"],
        },
    },
    handler=lambda block: block.input,
    terminal=True,
)
