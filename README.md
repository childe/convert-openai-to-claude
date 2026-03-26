# Claude-to-OpenAI API Proxy

接收 Claude Messages API (`/v1/messages`) 格式的流式请求，转换为 OpenAI Chat Completions API 格式转发到已配置的后端，再将响应转回 Claude SSE 格式。

## 安装

```bash
uv add fastapi uvicorn httpx pyyaml
```

## 配置

编辑 `config.yaml`：

```yaml
base_url: "https://api.openai.com/v1"
api_key: "your-api-key-here"
model: "gpt-4o"
host: "0.0.0.0"
port: 8080
```

## 运行

```bash
python main.py
```

## 使用

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

## 支持的功能

- 流式响应 (SSE)
- System prompt
- 多轮对话
- 图片输入 (base64 / URL)
- Tool use (函数调用)
- Tool result (工具结果回传)
- 参数映射: max_tokens, temperature, top_p, stop_sequences
