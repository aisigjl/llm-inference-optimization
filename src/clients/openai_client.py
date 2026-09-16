from __future__ import annotations

import json
import time
from typing import Any

import aiohttp


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        endpoint: str = "chat",
        api_key: str = "EMPTY",
        timeout_sec: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def generate(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = True,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions" if self.endpoint == "chat" else f"{self.base_url}/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.endpoint == "chat":
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
            }
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
            }

        start = time.perf_counter()
        first_token_time = None
        output_text_parts: list[str] = []
        async with session.post(url, headers=headers, json=payload, timeout=self.timeout) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")

            if not stream:
                data = await resp.json()
                latency = time.perf_counter() - start
                if self.endpoint == "chat":
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = data["choices"][0]["text"]
                return {"text": content, "latency_sec": latency, "ttft_sec": None}

            async for raw in resp.content:
                chunk = raw.decode("utf-8", errors="ignore")
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    parsed = json.loads(data)
                    choice = parsed["choices"][0]
                    if self.endpoint == "chat":
                        piece = choice.get("delta", {}).get("content", "")
                    else:
                        piece = choice.get("text", "")
                    if piece and first_token_time is None:
                        first_token_time = time.perf_counter()
                    if piece:
                        output_text_parts.append(piece)

        latency = time.perf_counter() - start
        return {
            "text": "".join(output_text_parts),
            "latency_sec": latency,
            "ttft_sec": None if first_token_time is None else first_token_time - start,
        }
