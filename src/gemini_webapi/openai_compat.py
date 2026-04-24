from __future__ import annotations

import argparse
import asyncio
import base64
import mimetypes
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import GeminiClient
from .constants import Model
from .exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ModelInvalid,
    TemporarilyBlocked,
    TimeoutError,
    UsageLimitExceeded,
)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None


@dataclass(slots=True)
class Settings:
    cookies_path: Path
    cookie_cache_dir: Path
    host: str
    port: int
    api_key: str | None
    default_model: str
    proxy: str | None
    request_timeout: float
    skip_verify: bool


def load_settings() -> Settings:
    home = Path.home()
    return Settings(
        cookies_path=Path(
            os.getenv(
                "GEMINI_WEBAPI_COOKIES",
                home / ".config/gemini-webapi/cookies.json",
            )
        ),
        cookie_cache_dir=Path(
            os.getenv(
                "GEMINI_COOKIE_PATH",
                home / ".local/state/gemini-webapi/cookies",
            )
        ),
        host=os.getenv("OPENAI_COMPAT_HOST", "127.0.0.1"),
        port=int(os.getenv("OPENAI_COMPAT_PORT", "8000")),
        api_key=os.getenv("OPENAI_COMPAT_API_KEY") or None,
        default_model=os.getenv("OPENAI_COMPAT_MODEL", Model.BASIC_FLASH.model_name),
        proxy=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None,
        request_timeout=float(os.getenv("OPENAI_COMPAT_TIMEOUT", "300")),
        skip_verify=os.getenv("OPENAI_COMPAT_SKIP_VERIFY", "").lower()
        in {"1", "true", "yes", "on"},
    )


def _load_cookie_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"Cookie config not found: {path}")

    data = orjson.loads(path.read_bytes())
    if isinstance(data, dict):
        if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            return dict(data)
        inner = data.get("cookies")
        if isinstance(inner, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in inner.items()
        ):
            return dict(inner)
        if isinstance(inner, list):
            result = {}
            for item in inner:
                if isinstance(item, dict):
                    name = item.get("name")
                    value = item.get("value")
                    if isinstance(name, str) and isinstance(value, str):
                        result[name] = value
            if result:
                return result
    if isinstance(data, list):
        result = {}
        for item in data:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                if isinstance(name, str) and isinstance(value, str):
                    result[name] = value
        if result:
            return result

    raise RuntimeError(f"Unsupported cookie config format: {path}")


def _guess_suffix(content_type: str | None, fallback: str = ".bin") -> str:
    if not content_type:
        return fallback
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed or fallback


def _download_url_bytes(url: str) -> tuple[bytes, str | None]:
    with urlopen(url, timeout=30) as response:
        return response.read(), response.headers.get_content_type()


async def _materialize_image_url(url: str, tmp_dir: Path) -> Path:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if not data:
            raise ValueError("Invalid data URL for image input.")
        meta = header[5:] if header.startswith("data:") else ""
        content_type = meta.split(";", 1)[0] or "application/octet-stream"
        if ";base64" in meta:
            raw = base64.b64decode(data)
        else:
            raw = data.encode("utf-8")
    else:
        try:
            raw, content_type = await asyncio.to_thread(_download_url_bytes, url)
        except URLError as exc:
            raise ValueError(f"Failed to download image URL: {url}") from exc

    suffix = _guess_suffix(content_type, ".img")
    fd, tmp_path = tempfile.mkstemp(prefix="openai-image-", suffix=suffix, dir=tmp_dir)
    os.close(fd)
    path = Path(tmp_path)
    path.write_bytes(raw)
    return path


async def _extract_part_text(
    part: dict[str, Any], files: list[Path], tmp_dir: Path
) -> str:
    part_type = part.get("type")
    if part_type in {"text", "input_text"}:
        text = part.get("text", "")
        return text if isinstance(text, str) else ""

    if part_type in {"image_url", "input_image"}:
        image_url = part.get("image_url")
        url: str | None = None
        if isinstance(image_url, dict):
            value = image_url.get("url")
            if isinstance(value, str):
                url = value
        elif isinstance(image_url, str):
            url = image_url
        if not url:
            raise ValueError("image_url part is missing a valid URL.")
        files.append(await _materialize_image_url(url, tmp_dir))
        return "[User attached an image]"

    return f"[Unsupported content part type: {part_type}]"


async def build_prompt_and_files(
    messages: list[ChatMessage], tmp_dir: Path
) -> tuple[str, list[Path]]:
    if not messages:
        raise ValueError("messages must not be empty.")

    system_chunks: list[str] = []
    transcript: list[str] = []
    files: list[Path] = []

    for message in messages:
        role = message.role.lower()
        if message.content is None:
            content_text = ""
        elif isinstance(message.content, str):
            content_text = message.content
        elif isinstance(message.content, list):
            parts = []
            for part in message.content:
                if not isinstance(part, dict):
                    continue
                parts.append(await _extract_part_text(part, files, tmp_dir))
            content_text = "\n".join(filter(None, parts))
        else:
            raise ValueError(f"Unsupported content type for role '{message.role}'.")

        if role in {"system", "developer"}:
            if content_text:
                system_chunks.append(content_text)
            continue

        display_role = role.upper()
        if message.name:
            display_role = f"{display_role} ({message.name})"
        transcript.append(f"{display_role}:\n{content_text}".strip())

    if not transcript:
        raise ValueError("At least one non-system message is required.")

    prompt_parts = []
    if system_chunks:
        prompt_parts.append("System instructions:\n" + "\n\n".join(system_chunks))
    prompt_parts.append("Conversation:\n" + "\n\n".join(transcript))
    prompt_parts.append(
        "Reply as the assistant to the latest user-visible message. "
        "Use the conversation history above as context."
    )
    return "\n\n".join(prompt_parts), files


def _usage_stub() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _openai_error(
    message: str,
    *,
    code: str | None = None,
    error_type: str = "invalid_request_error",
    status_code: int = status.HTTP_400_BAD_REQUEST,
) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


def _map_exception(exc: Exception) -> JSONResponse:
    if isinstance(exc, AuthError):
        return _openai_error(
            str(exc) or "Gemini authentication failed. Check your cookies.",
            code="auth_error",
            error_type="authentication_error",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    if isinstance(exc, (ModelInvalid, ValueError)):
        return _openai_error(str(exc), code="invalid_model")
    if isinstance(exc, UsageLimitExceeded):
        return _openai_error(
            str(exc),
            code="rate_limit_exceeded",
            error_type="rate_limit_error",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    if isinstance(exc, TemporarilyBlocked):
        return _openai_error(
            str(exc),
            code="temporarily_blocked",
            error_type="server_error",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    if isinstance(exc, TimeoutError):
        return _openai_error(
            str(exc),
            code="timeout",
            error_type="timeout_error",
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    if isinstance(exc, (APIError, GeminiError, RuntimeError)):
        return _openai_error(
            str(exc),
            code="gemini_error",
            error_type="server_error",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    return _openai_error(
        str(exc) or exc.__class__.__name__,
        code="internal_error",
        error_type="server_error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


class GeminiOpenAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: GeminiClient | None = None
        self._lock = asyncio.Lock()

    async def ensure_client(self) -> GeminiClient:
        if self._client and self._client._running:
            return self._client

        async with self._lock:
            if self._client and self._client._running:
                return self._client

            cookies = _load_cookie_map(self.settings.cookies_path)
            secure_1psid = cookies.get("__Secure-1PSID")
            secure_1psidts = cookies.get("__Secure-1PSIDTS", "")
            if not secure_1psid or secure_1psid == "paste-your-value-here":
                raise AuthError(
                    f"Set a real __Secure-1PSID value in {self.settings.cookies_path}."
                )

            os.environ["GEMINI_COOKIE_PATH"] = str(self.settings.cookie_cache_dir)

            client = GeminiClient(
                secure_1psid=secure_1psid,
                secure_1psidts=secure_1psidts,
                proxy=self.settings.proxy,
                verify=not self.settings.skip_verify,
            )

            extra_cookies = {
                key: value
                for key, value in cookies.items()
                if key not in {"__Secure-1PSID", "__Secure-1PSIDTS"} and value
            }
            if extra_cookies:
                client.cookies = extra_cookies

            try:
                await client.init(
                    timeout=self.settings.request_timeout,
                    auto_close=False,
                    auto_refresh=True,
                    verbose=False,
                )
            except Exception:
                await client.close()
                raise

            self._client = client
            return client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def list_models(self) -> list[dict[str, Any]]:
        client = await self.ensure_client()
        models = client.list_models() or []
        return [
            {
                "id": model.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "google-gemini-web",
            }
            for model in models
            if getattr(model, "model_name", None)
        ]

    async def create_completion_payload(
        self, request: ChatCompletionRequest
    ) -> tuple[dict[str, Any], Any]:
        client = await self.ensure_client()
        model_name = request.model or self.settings.default_model
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        with tempfile.TemporaryDirectory(prefix="gemini-openai-") as tmp_name:
            prompt, files = await build_prompt_and_files(
                request.messages, Path(tmp_name)
            )
            output = await client.generate_content(
                prompt=prompt,
                files=[str(path) for path in files] or None,
                model=model_name,
            )

        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output.text or "",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage_stub(),
        }
        return payload, output

    async def stream_completion(self, request: ChatCompletionRequest):
        client = await self.ensure_client()
        model_name = request.model or self.settings.default_model
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        tmp_ctx = tempfile.TemporaryDirectory(prefix="gemini-openai-")
        tmp_dir = Path(tmp_ctx.name)
        try:
            prompt, files = await build_prompt_and_files(request.messages, tmp_dir)
        except Exception:
            tmp_ctx.cleanup()
            raise

        async def iterator():
            try:
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {orjson.dumps(first_chunk).decode('utf-8')}\n\n"

                async for chunk in client.generate_content_stream(
                    prompt=prompt,
                    files=[str(path) for path in files] or None,
                    model=model_name,
                ):
                    if not chunk.text_delta:
                        continue
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {orjson.dumps(payload).decode('utf-8')}\n\n"

                final_payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {orjson.dumps(final_payload).decode('utf-8')}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                tmp_ctx.cleanup()

        return iterator()


def create_app(service: GeminiOpenAIService | None = None) -> FastAPI:
    settings = service.settings if service else load_settings()
    app_service = service or GeminiOpenAIService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.service = app_service
        try:
            yield
        finally:
            await app.state.service.close()

    app = FastAPI(
        title="Gemini OpenAI-Compatible API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = app_service

    def require_api_key(request: Request) -> None:
        api_key = app.state.settings.api_key
        if not api_key:
            return
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {api_key}":
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return _openai_error(
            str(exc.detail),
            code="http_error",
            error_type="authentication_error"
            if exc.status_code == status.HTTP_401_UNAUTHORIZED
            else "invalid_request_error",
            status_code=exc.status_code,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "cookies_path": str(app.state.settings.cookies_path),
            "default_model": app.state.settings.default_model,
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        require_api_key(request)
        try:
            return {"object": "list", "data": await app.state.service.list_models()}
        except Exception as exc:
            return _map_exception(exc)

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, request: Request):
        require_api_key(request)
        try:
            models = await app.state.service.list_models()
            for model in models:
                if model["id"] == model_id:
                    return model
            return _openai_error(
                f"Model '{model_id}' was not found.",
                code="model_not_found",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        except Exception as exc:
            return _map_exception(exc)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
    ):
        require_api_key(request)
        try:
            if body.stream:
                stream = await app.state.service.stream_completion(body)
                return StreamingResponse(stream, media_type="text/event-stream")
            payload, _ = await app.state.service.create_completion_payload(body)
            return payload
        except Exception as exc:
            return _map_exception(exc)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API wrapper for gemini-webapi"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    uvicorn.run(
        create_app(GeminiOpenAIService(settings)),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
