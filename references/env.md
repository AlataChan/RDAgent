# RD-Agent `.env` templates (minimal)

RD-Agent loads `.env` from the **current working directory** (it calls `load_dotenv(".env")`).

Keep secrets out of git. Use placeholders below and fill on the Linux host.

## OpenAI-compatible (LiteLLM default backend)

```bash
cat <<'EOF' > .env
# Models (must be supported by your LiteLLM backend)
CHAT_MODEL=<chat-model>
EMBEDDING_MODEL=<embedding-model>

# OpenAI-compatible API
OPENAI_API_BASE=<https://your-api-base/v1>
OPENAI_API_KEY=<your-api-key>
EOF
```

## DeepSeek chat + separate embedding provider (example)

```bash
cat <<'EOF' > .env
# Chat model via DeepSeek
CHAT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=<your-deepseek-api-key>

# Embedding via a separate provider supported by LiteLLM
EMBEDDING_MODEL=litellm_proxy/<embedding-model>
LITELLM_PROXY_API_BASE=<https://your-embedding-api-base/v1>
LITELLM_PROXY_API_KEY=<your-embedding-api-key>
EOF
```

## Health check

```bash
rdagent health_check
```

