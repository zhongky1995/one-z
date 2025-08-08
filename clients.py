import requests
from typing import Dict, Any

# Mapping providers to header keys and default paths
PROVIDERS = {
    "openai": {
        "auth": "Authorization",
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings"
    },
    "azure": {
        "auth": "api-key",
        # For azure, base_url should include deployment e.g. https://{endpoint}/openai/deployments/{model}
        "chat_path": "/chat/completions?api-version=2024-02-15-preview",
        "embed_path": "/embeddings?api-version=2024-02-15-preview"
    },
    "qwen": {  # 通义千问
        "auth": "Authorization",
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings"
    },
    "deepseek": {
        "auth": "Authorization",
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings"
    },
    "kimi": {
        "auth": "Authorization",
        "chat_path": "/v1/chat/completions",
        "embed_path": "/v1/embeddings"
    },
}


def _request(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def generate_text(prompt: str, cfg: Dict[str, str]) -> str:
    provider = cfg["provider"]
    base = cfg["base_url"]
    model = cfg["model"]
    info = PROVIDERS[provider]
    headers = {info["auth"]: f"Bearer {cfg['api_key']}" if info["auth"] == "Authorization" else cfg['api_key']}
    url = base + info["chat_path"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    data = _request(url, headers, payload)
    return data["choices"][0]["message"]["content"].strip()


def get_embedding(text: str, cfg: Dict[str, str]) -> Any:
    provider = cfg["provider"]
    base = cfg["base_url"]
    model = cfg["embedding_model"]
    info = PROVIDERS[provider]
    headers = {info["auth"]: f"Bearer {cfg['api_key']}" if info["auth"] == "Authorization" else cfg['api_key']}
    url = base + info["embed_path"]
    payload = {"model": model, "input": text}
    data = _request(url, headers, payload)
    return data["data"][0]["embedding"]
