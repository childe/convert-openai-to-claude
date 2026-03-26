import json
import uuid
import logging

import yaml
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI()

STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


# ── Request conversion: Claude → OpenAI ──────────────────────────────


def _convert_content_block(block):
    """Convert a single Claude content block to OpenAI format."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    t = block.get("type")
    if t == "text":
        return {"type": "text", "text": block["text"]}
    if t == "image":
        src = block["source"]
        url = (
            f"data:{src['media_type']};base64,{src['data']}"
            if src.get("type") == "base64"
            else src.get("url", "")
        )
        return {"type": "image_url", "image_url": {"url": url}}
    # tool_use / tool_result handled at message level
    return None


def _convert_messages(claude_messages: list) -> list:
    """Convert Claude messages array to OpenAI messages array."""
    out: list[dict] = []
    for msg in claude_messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # Classify blocks
        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        others = [b for b in content if not (isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"))]

        # Assistant with tool_use → OpenAI assistant with tool_calls
        if role == "assistant" and tool_uses:
            texts = []
            calls = []
            for b in content:
                if isinstance(b, str):
                    texts.append(b)
                elif b.get("type") == "text":
                    texts.append(b["text"])
                elif b.get("type") == "tool_use":
                    calls.append({
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]) if isinstance(b["input"], dict) else str(b["input"]),
                        },
                    })
            m = {"role": "assistant", "content": "\n".join(texts) or None, "tool_calls": calls}
            out.append(m)
            continue

        # User with tool_result → OpenAI tool messages
        if tool_results:
            # Emit non-tool content first
            parts = [p for b in others if (p := _convert_content_block(b))]
            if parts:
                out.append({"role": "user", "content": parts if len(parts) > 1 else parts[0].get("text", parts)})
            for tr in tool_results:
                c = tr.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(b.get("text", str(b)) for b in c)
                out.append({"role": "tool", "tool_call_id": tr["tool_use_id"], "content": c or ""})
            continue

        # Regular content blocks
        parts = [p for b in content if (p := _convert_content_block(b))]
        if len(parts) == 1 and parts[0].get("type") == "text":
            out.append({"role": role, "content": parts[0]["text"]})
        elif parts:
            out.append({"role": role, "content": parts})

    return out


def _convert_tools(claude_tools: list | None) -> list | None:
    if not claude_tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in claude_tools
    ]


def _convert_tool_choice(tc) -> str | dict | None:
    if not tc:
        return None
    if isinstance(tc, str):
        return tc
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "tool":
        return {"type": "function", "function": {"name": tc["name"]}}
    return None


def _build_openai_request(body: dict) -> dict:
    messages = []
    system = body.get("system")
    if system:
        text = "\n".join(b["text"] if isinstance(b, dict) else str(b) for b in system) if isinstance(system, list) else system
        messages.append({"role": "system", "content": text})
    messages.extend(_convert_messages(body["messages"]))

    req = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for src, dst in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")]:
        if src in body:
            req[dst] = body[src]
    if "stop_sequences" in body:
        req["stop"] = body["stop_sequences"]

    tools = _convert_tools(body.get("tools"))
    if tools:
        req["tools"] = tools
    tc = _convert_tool_choice(body.get("tool_choice"))
    if tc:
        req["tool_choice"] = tc

    return req


# ── Response conversion: OpenAI streaming → Claude SSE ───────────────


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_response(body: dict):
    openai_req = _build_openai_request(body)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = body.get("model", CONFIG["model"])
    url = f"{CONFIG['base_url'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {CONFIG['api_key']}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        try:
            async with client.stream("POST", url, json=openai_req, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    logger.error("Backend %s: %s", resp.status_code, err.decode(errors="replace"))
                    yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": f"Backend returned {resp.status_code}"}})
                    return

                block_idx = 0
                in_text = False
                tool_states: dict[int, dict] = {}  # openai tool index → state
                tool_block_indices: dict[int, int] = {}  # openai tool index → claude block index
                input_tokens = output_tokens = 0
                stop_reason = None
                started = False

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # message_start on first chunk
                    if not started:
                        started = True
                        yield _sse("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": msg_id, "type": "message", "role": "assistant",
                                "content": [], "model": model,
                                "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        })
                        yield _sse("ping", {"type": "ping"})

                    if chunk.get("usage"):
                        input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)
                        output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish = choices[0].get("finish_reason")

                    # ── text delta ──
                    text = delta.get("content")
                    if text is not None:
                        if not in_text:
                            in_text = True
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": block_idx,
                                "content_block": {"type": "text", "text": ""},
                            })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta", "index": block_idx,
                            "delta": {"type": "text_delta", "text": text},
                        })

                    # ── tool_calls delta ──
                    for tc in delta.get("tool_calls", []):
                        ti = tc["index"]
                        if ti not in tool_states:
                            # close text block if open
                            if in_text:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                                block_idx += 1
                                in_text = False
                            tool_states[ti] = {"id": tc.get("id", ""), "name": tc.get("function", {}).get("name", ""), "args": ""}
                            tool_block_indices[ti] = block_idx
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": block_idx,
                                "content_block": {"type": "tool_use", "id": tool_states[ti]["id"], "name": tool_states[ti]["name"], "input": {}},
                            })
                        args_part = tc.get("function", {}).get("arguments", "")
                        if args_part:
                            tool_states[ti]["args"] += args_part
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": tool_block_indices[ti],
                                "delta": {"type": "input_json_delta", "partial_json": args_part},
                            })

                    # ── finish ──
                    if finish:
                        stop_reason = STOP_REASON_MAP.get(finish, "end_turn")
                        if in_text:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
                            block_idx += 1
                            in_text = False
                        for ti in sorted(tool_states):
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": tool_block_indices[ti]})
                            block_idx += 1

                yield _sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                })
                yield _sse("message_stop", {"type": "message_stop"})

        except httpx.ConnectError as e:
            logger.error("Connection failed: %s", e)
            yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": f"Connection failed: {e}"}})
        except httpx.ReadTimeout:
            logger.error("Backend read timeout")
            yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": "Backend read timeout"}})


# ── Endpoint ─────────────────────────────────────────────────────────


def _error(status: int, msg: str):
    return JSONResponse(status_code=status, content={"type": "error", "error": {"type": "invalid_request_error", "message": msg}})


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Invalid JSON body")
    if "messages" not in body:
        return _error(400, "messages is required")
    if "max_tokens" not in body:
        return _error(400, "max_tokens is required")

    logger.info("Request: model=%s msgs=%d", body.get("model"), len(body["messages"]))
    return StreamingResponse(
        _stream_response(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Proxy %s:%s → %s (model=%s)", CONFIG["host"], CONFIG["port"], CONFIG["base_url"], CONFIG["model"])
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
