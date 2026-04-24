import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from gemini_webapi.openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    GeminiOpenAIService,
    Settings,
    build_prompt_and_files,
    create_app,
)


class FakeModel:
    def __init__(self, model_name):
        self.model_name = model_name


class FakeChunk:
    def __init__(self, text_delta):
        self.text_delta = text_delta


class FakeClient:
    async def generate_content(self, prompt, files=None, model=None):
        class Output:
            text = f"echo:{model}:{prompt.splitlines()[-1]}"

        return Output()

    async def generate_content_stream(self, prompt, files=None, model=None):
        yield FakeChunk("hel")
        yield FakeChunk("lo")


class FakeService(GeminiOpenAIService):
    def __init__(self):
        super().__init__(
            Settings(
                cookies_path=Path("/tmp/cookies.json"),
                cookie_cache_dir=Path("/tmp/gemini-cache"),
                host="127.0.0.1",
                port=8000,
                api_key="secret",
                default_model="gemini-3-flash",
                proxy=None,
                request_timeout=300,
                skip_verify=False,
            )
        )
        self.fake_client = FakeClient()

    async def ensure_client(self):
        return self.fake_client

    async def list_models(self):
        return [{"id": "gemini-3-flash", "object": "model", "created": 0, "owned_by": "google-gemini-web"}]


class OpenAICompatTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_prompt_and_files_text_only(self):
        with TemporaryDirectory() as tmp_dir:
            prompt, files = await build_prompt_and_files(
                [
                    ChatMessage(role="system", content="Be concise."),
                    ChatMessage(role="user", content="Hello"),
                    ChatMessage(role="assistant", content="Hi"),
                    ChatMessage(role="user", content="How are you?"),
                ],
                Path(tmp_dir),
            )
        self.assertIn("System instructions:\nBe concise.", prompt)
        self.assertIn("USER:\nHello", prompt)
        self.assertIn("ASSISTANT:\nHi", prompt)
        self.assertIn("USER:\nHow are you?", prompt)
        self.assertEqual(files, [])

    async def test_request_model_defaults(self):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Ping")])
        self.assertFalse(req.stream)
        self.assertIsNone(req.model)


class OpenAICompatRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(FakeService()))

    def test_healthz(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_models_requires_api_key(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 401)

    def test_models_success(self):
        response = self.client.get(
            "/v1/models", headers={"Authorization": "Bearer secret"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "gemini-3-flash")

    def test_chat_completion(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "chat.completion")
        self.assertEqual(response.json()["choices"][0]["message"]["role"], "assistant")

    def test_chat_completion_stream(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": "gemini-3-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ) as response:
            body = "".join(response.iter_text())
        self.assertEqual(response.status_code, 200)
        self.assertIn("data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()
