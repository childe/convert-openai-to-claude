# Claude-to-OpenAI API Proxy

If you already have an OpenAI-compatible API endpoint and want to use it with Claude clients, this proxy bridges the gap. It accepts Claude Messages API requests, translates them to OpenAI Chat Completions format, forwards them to your backend, and converts the streaming response back to Claude SSE format.

## Features

- Streaming responses (SSE)
- System prompt
- Multi-turn conversations
- Image inputs (base64 / URL)
- Tool use (function calling)
- Tool result passback
- Parameter mapping: max_tokens, temperature, top_p, stop_sequences

## Setup

```bash
uv add fastapi uvicorn httpx pyyaml
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your backend details:

```yaml
base_url: "https://api.openai.com/v1"
api_key: "your-api-key-here"
model: "gpt-4o"
host: "0.0.0.0"
port: 8080
```

## Usage

```bash
python main.py
```

Then point your Claude client to `http://localhost:8080` as the API base URL:

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```
