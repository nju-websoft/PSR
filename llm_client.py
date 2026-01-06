import json
import os
import requests
import time
from typing import Any, Optional
from openai import OpenAI

LLM_STATS = {
    "total_calls": 0,
    "per_model": {},
    "token_usage": {},
    "time_usage_sec": {},
}


class LLM:
    def __init__(self, local_llm_url: Optional[str] = None, default_engine: Optional[str] = None):
        self._API_KEY_1 = os.getenv("LLM_API_KEY_1")
        self._API_BASE_1 = os.getenv("LLM_API_BASE_1")
        self._API_KEY_2 = os.getenv("LLM_API_KEY_2")
        self._API_BASE_2 = os.getenv("LLM_API_BASE_2")
        self._API_KEY_3 = os.getenv("LLM_API_KEY_3")
        self._openai_compatible_client = OpenAI(
            api_key=self._API_KEY_3,
            base_url=os.getenv("LLM_API_BASE_3")
        )

        self.local_url = local_llm_url or os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
        self.default_engine = default_engine or os.getenv("DEFAULT_ENGINE", "llama2:7b")

        self._default_timeout = 600
        self._default_retries = 4
        self._backoff_factor = 0.4

    def _wrap_prompt(self, prompt: str) -> str:
        return f"[INST] {prompt} [/INST]"

    def _parse_response(self, resp: Any) -> str:
        if hasattr(resp, "choices"):
            ch = resp.choices[0]
            if hasattr(ch, "message"):
                if hasattr(ch.message, "content") and isinstance(ch.message.content, str):
                    return ch.message.content.strip()

        if isinstance(resp, dict):
            if "choices" in resp:
                c = resp["choices"][0]
                if "message" in c and "content" in c["message"]:
                    return c["message"]["content"].strip()

                for k in ["text", "content", "response"]:
                    v = c.get(k)
                    if isinstance(v, str):
                        return v.strip()

            if "response" in resp and isinstance(resp["response"], str):
                return resp["response"].strip()

        if isinstance(resp, str):
            return resp.strip()

        return ""

    def _post_with_retry(self, url, json_payload, headers=None, timeout=None, retries=None):
        t = timeout or self._default_timeout
        rts = retries or self._default_retries

        for attempt in range(1, rts + 1):
            try:
                r = requests.post(url, json=json_payload, headers=headers, timeout=t)
                if r.status_code < 400:
                    return r.json()

                if attempt == rts:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

            except Exception as e:
                if attempt == rts:
                    raise RuntimeError(f"Request failed after retries: {e}")

            time.sleep(self._backoff_factor * (2 ** attempt))

    def _post_stream(self, url, payload, headers):
        r = requests.post(url, json=payload, headers=headers, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        chunks = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str in ("", "[DONE]", "null"):
                continue
            try:
                obj = json.loads(data_str)
            except Exception:
                continue
            try:
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
            except Exception:
                continue

        return "".join(chunks)

    def _call_api_provider_1(self, prompt, max_tokens, temperature, model, stream=False):
        headers = {"Authorization": f"Bearer {self._API_KEY_1}",
                   "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self._wrap_prompt(prompt)}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            return self._post_stream(self._API_BASE_1 + "/chat/completions", payload, headers)
        else:
            return self._post_with_retry(self._API_BASE_1 + "/chat/completions", payload, headers)

    def _call_api_provider_2(self, prompt, max_tokens, temperature, model):
        headers = {
            "Authorization": f"Bearer {self._API_KEY_2}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self._wrap_prompt(prompt)}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        return self._post_with_retry(self._API_BASE_2, payload, headers)

    def _call_api_provider_3(self, prompt, max_tokens, temperature, model):
        return self._openai_compatible_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )

    def _call_local_llama(self, prompt, max_tokens, temperature, model):
        chat_url = f"{self.local_url}/api/chat"
        payload_chat = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        try:
            r = requests.post(chat_url, json=payload_chat, timeout=30)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        gen_url = f"{self.local_url}/api/generate"
        payload_gen = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        return self._post_with_retry(gen_url, payload_gen)

    def generate(self, model: str, prompt: str,
                 max_tokens: int = 200,
                 temperature: float = 0.2,
                 timeout: int = 600,
                 retries: int = None) -> str:

        start = time.time()
        model_lower = model.lower()

        if model_lower.startswith("gpt"):
            resp = self._call_api_provider_2(prompt, max_tokens, temperature, model)
        elif any(model_lower.startswith(p) for p in ["claude", "llama", "mistral"]):
            resp = self._call_api_provider_3(prompt, max_tokens, temperature, model)
        else:
            engine = model or self.default_engine
            resp = self._call_local_llama(prompt, max_tokens, temperature, engine)

        elapsed = time.time() - start
        self._update_stats(model, resp, elapsed)

        return self._parse_response(resp)

    def _update_stats(self, model: str, resp: Any, elapsed: float):
        LLM_STATS["total_calls"] += 1
        LLM_STATS["per_model"][model] = LLM_STATS["per_model"].get(model, 0) + 1
        LLM_STATS["time_usage_sec"][model] = LLM_STATS["time_usage_sec"].get(model, 0.0) + elapsed

        parsed = self._parse_response(resp)
        tokens = len(parsed.split())
        LLM_STATS["token_usage"][model] = LLM_STATS["token_usage"].get(model, 0) + tokens


_global_llm = None


def _get_global_llm():
    global _global_llm
    if _global_llm is None:
        _global_llm = LLM()
    return _global_llm


def call_llm(model: str, prompt: str,
             max_tokens: int = 200,
             temperature: float = 0.2,
             timeout: int = 600) -> str:
    return _get_global_llm().generate(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )