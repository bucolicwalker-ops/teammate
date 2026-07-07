"""teammate · W4 —— Embedding HTTP 服务。
把 MLX 本地 Qwen3-Embedding 暴露成 OpenAI 兼容的 /v1/embeddings 端点。
跑法：.venv/bin/python -m src.embed_server  （默认 :8765）

W3 的 VectorMemory 是进程内加载模型；W4 起服务，避免每个脚本重复加载。
"""
import mlx.core as mx
from mlx_lm import load as mlx_load
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

app = FastAPI(title="teammate embedding service")

print(f"Loading {MODEL_ID}...")
_model, _tokenizer = mlx_load(MODEL_ID)
print(f"Model loaded. dim=1024")


MAX_INPUT_CHARS = 32000  # Qwen3-Embedding 32K context，保守字符上限


class EmbedRequest(BaseModel):
    model: str = MODEL_ID
    input: str | list[str]


@app.post("/v1/embeddings")
def create_embeddings(req: EmbedRequest):
    from fastapi import HTTPException
    texts = [req.input] if isinstance(req.input, str) else req.input
    for i, text in enumerate(texts):
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(413, f"Input {i} exceeds {MAX_INPUT_CHARS} chars")
    data = []
    for i, text in enumerate(texts):
        vec = _embed(text)
        data.append({"object": "embedding", "embedding": vec, "index": i})
    return {
        "object": "list",
        "data": data,
        "model": req.model,
    }


def _embed(text: str) -> list[float]:
    tokens = _tokenizer.encode(text)
    input_ids = mx.array([tokens])
    hidden = _model.model(input_ids)
    pooled = hidden[0, -1, :]
    norm = mx.sqrt(mx.sum(pooled * pooled))
    return [float(x) for x in (pooled / norm)]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dim": 1024}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
