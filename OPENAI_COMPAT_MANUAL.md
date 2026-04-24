# Gemini OpenAI-Compatible API Manual

This document describes the local OpenAI-compatible API wrapper deployed for `Gemini-API` on this machine.

## Overview

The original repository is a Python client and CLI for the Gemini web app. It does not expose an OpenAI-style HTTP API by itself.

A local compatibility server has been added in:

- [src/gemini_webapi/openai_compat.py](/home/nixos/Gemini-API/src/gemini_webapi/openai_compat.py)

It provides these endpoints:

- `GET /healthz`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`

The server is deployed as a user-level `systemd` service and currently listens on:

- `http://127.0.0.1:8001`

Port `8000` was already in use on this machine, so the service was configured to use `8001`.

## Files

Runtime files:

- Repo: [Gemini-API](/home/nixos/Gemini-API)
- Cookie config: [~/.config/gemini-webapi/cookies.json](/home/nixos/.config/gemini-webapi/cookies.json)
- API env config: [~/.config/gemini-webapi/openai.env](/home/nixos/.config/gemini-webapi/openai.env)
- Launcher: [~/.local/bin/gemini-openai-api](/home/nixos/.local/bin/gemini-openai-api)
- Service unit: [~/.config/systemd/user/gemini-openai-api.service](/home/nixos/.config/systemd/user/gemini-openai-api.service)

Code changes:

- Server adapter: [src/gemini_webapi/openai_compat.py](/home/nixos/Gemini-API/src/gemini_webapi/openai_compat.py)
- Tests: [tests/test_openai_compat.py](/home/nixos/Gemini-API/tests/test_openai_compat.py)
- Optional dependencies: [pyproject.toml](/home/nixos/Gemini-API/pyproject.toml)

## First-Time Setup

### 1. Set Gemini cookies

Edit:

- [~/.config/gemini-webapi/cookies.json](/home/nixos/.config/gemini-webapi/cookies.json)

Template:

```json
{
  "__Secure-1PSID": "paste-your-value-here",
  "__Secure-1PSIDTS": ""
}
```

Notes:

- `__Secure-1PSID` is required.
- `__Secure-1PSIDTS` is optional for some accounts.
- The server will reject requests until `__Secure-1PSID` is replaced with a real value.

### 2. Set the API key and runtime options

Edit:

- [~/.config/gemini-webapi/openai.env](/home/nixos/.config/gemini-webapi/openai.env)

Current contents:

```dotenv
OPENAI_COMPAT_HOST=127.0.0.1
OPENAI_COMPAT_PORT=8001
OPENAI_COMPAT_API_KEY=change-me
OPENAI_COMPAT_MODEL=gemini-3-flash-thinking
OPENAI_COMPAT_TIMEOUT=300
OPENAI_COMPAT_SKIP_VERIFY=false
GEMINI_WEBAPI_COOKIES=/home/nixos/.config/gemini-webapi/cookies.json
GEMINI_COOKIE_PATH=/home/nixos/.local/state/gemini-webapi/cookies
```

At minimum, change:

- `OPENAI_COMPAT_API_KEY`

### 3. Restart the service

```bash
systemctl --user restart gemini-openai-api.service
```

## Service Management

Check status:

```bash
systemctl --user status gemini-openai-api.service --no-pager
```

Restart:

```bash
systemctl --user restart gemini-openai-api.service
```

Stop:

```bash
systemctl --user stop gemini-openai-api.service
```

Start:

```bash
systemctl --user start gemini-openai-api.service
```

View logs:

```bash
journalctl --user -u gemini-openai-api.service -f
```

## Health Check

No API key is required for `healthz`.

```bash
curl http://127.0.0.1:8001/healthz
```

Expected shape:

```json
{
  "status": "ok",
  "cookies_path": "/home/nixos/.config/gemini-webapi/cookies.json",
  "default_model": "gemini-3-flash"
}
```

## Authentication

All `/v1/*` endpoints require:

```http
Authorization: Bearer YOUR_API_KEY
```

If the header is missing or wrong, the server returns `401`.

## Endpoints

### `GET /v1/models`

Example:

```bash
curl -H 'Authorization: Bearer YOUR_API_KEY' \
  http://127.0.0.1:8001/v1/models
```

This asks Gemini for the list of models available to the configured account.

### `GET /v1/models/{model_id}`

Example:

```bash
curl -H 'Authorization: Bearer YOUR_API_KEY' \
  http://127.0.0.1:8001/v1/models/gemini-3-flash
```

### `POST /v1/chat/completions`

Non-streaming example:

```bash
curl -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8001/v1/chat/completions \
  -d '{
    "model": "gemini-3-flash",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Explain TCP in one paragraph."}
    ]
  }'
```

Streaming example:

```bash
curl -N -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8001/v1/chat/completions \
  -d '{
    "model": "gemini-3-flash",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Write a haiku about caching."}
    ]
  }'
```

## OpenAI SDK Example

Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="gemini-3-flash",
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello."},
    ],
)

print(response.choices[0].message.content)
```

Streaming example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="YOUR_API_KEY",
)

stream = client.chat.completions.create(
    model="gemini-3-flash",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

## Message Mapping

The compatibility layer maps OpenAI chat messages into a single prompt sent to Gemini.

Behavior:

- `system` and `developer` messages are merged into a system-instructions block.
- Remaining messages are serialized as a transcript.
- Gemini then replies to the latest user-visible message using that transcript as context.

This works well for ordinary chat use, but it is not a perfect wire-compatible reproduction of the OpenAI backend.

## Image Input Support

The server accepts message content arrays with text and image parts.

Supported patterns:

- `{"type":"text","text":"..."}`
- `{"type":"input_text","text":"..."}`
- `{"type":"image_url","image_url":{"url":"..."}}`
- `{"type":"input_image","image_url":"..."}`

For image URLs:

- Remote URLs are downloaded temporarily before being uploaded to Gemini.
- `data:` URLs are also accepted.

## Limitations

Current limitations of this compatibility layer:

- Only chat-completions style requests are implemented.
- No `responses` API.
- No embeddings endpoint.
- No audio/transcription endpoints.
- No tool calling or function calling schema.
- No exact OpenAI token accounting. Usage fields are currently placeholders.
- No persistent chat-session mapping across separate HTTP calls.
- Output images and other Gemini-specific rich media are not exposed in OpenAI-native response fields yet.

## Troubleshooting

### `401 Invalid or missing API key`

Cause:

- Missing or wrong `Authorization: Bearer ...` header.

Fix:

- Use the key from [openai.env](/home/nixos/.config/gemini-webapi/openai.env).

### `auth_error` about `__Secure-1PSID`

Cause:

- Placeholder cookie still present.
- Cookie expired.
- Wrong Gemini account/session.

Fix:

- Replace the cookie in [cookies.json](/home/nixos/.config/gemini-webapi/cookies.json).
- Restart the service.

### Service will not start

Check:

```bash
systemctl --user status gemini-openai-api.service --no-pager
journalctl --user -u gemini-openai-api.service -n 50 --no-pager
```

Common causes:

- Port already in use.
- Invalid env file.
- Broken cookie config.

### Port conflict

Edit:

- [openai.env](/home/nixos/.config/gemini-webapi/openai.env)

Change:

```dotenv
OPENAI_COMPAT_PORT=8001
```

to another free port, then restart:

```bash
systemctl --user restart gemini-openai-api.service
```

### SSL verification issues

If you are behind a proxy or unusual local CA setup, you can set:

```dotenv
OPENAI_COMPAT_SKIP_VERIFY=true
```

Then restart the service.

Use this only if you understand the TLS tradeoff.

## Validation Checklist

After configuration, these checks should pass:

1. `curl http://127.0.0.1:8001/healthz`
2. `curl -H 'Authorization: Bearer YOUR_API_KEY' http://127.0.0.1:8001/v1/models`
3. `curl -H 'Authorization: Bearer YOUR_API_KEY' -H 'Content-Type: application/json' http://127.0.0.1:8001/v1/chat/completions -d '{"model":"gemini-3-flash","messages":[{"role":"user","content":"Hello"}]}'`

## Maintenance

If you update the repo or the adapter code:

```bash
cd /home/nixos/Gemini-API
/home/nixos/Gemini-API/.venv/bin/pip install -e '.[browser,server]'
systemctl --user restart gemini-openai-api.service
```

If you want to disable the service at login:

```bash
systemctl --user disable --now gemini-openai-api.service
```
