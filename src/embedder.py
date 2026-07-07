"""teammate · W4 —— Embedder：embedding 后端抽象。
HTTP 优先，本地 MLX fallback。支持单条 + batch。
VectorMemory 和 KnowledgeBase 共用，改模型只改一处。
"""
import requests

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


class Embedder:
    """embedding 后端：HTTP 服务优先，本地 MLX fallback。

    用法：embedder = Embedder("http://127.0.0.1:8765")
         vec = embedder("text")           # 单条
         vecs = embedder.batch(["t1","t2"])  # 批量
    """

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint
        self._model = None
        self._tokenizer = None

    def __call__(self, text: str) -> list[float]:
        """单条 embed：HTTP 优先 → fallback 本地。"""
        if self.endpoint:
            try:
                resp = requests.post(
                    f"{self.endpoint}/v1/embeddings",
                    json={"model": MODEL_ID, "input": text},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except Exception as e:
                if _HAS_MLX:
                    print(f"  ⚠️ HTTP embedding 失败，回退本地: {e}")
                else:
                    raise
        return self._embed_local(text)

    def batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embed：HTTP 一次请求，本地逐条。"""
        if self.endpoint:
            try:
                resp = requests.post(
                    f"{self.endpoint}/v1/embeddings",
                    json={"model": MODEL_ID, "input": texts},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
            except Exception as e:
                if _HAS_MLX:
                    print(f"  ⚠️ HTTP batch 失败，回退本地逐条: {e}")
                else:
                    raise
        return [self(t) for t in texts]

    def _embed_local(self, text: str) -> list[float]:
        if self._model is None:
            if not _HAS_MLX:
                raise RuntimeError("MLX not installed and no embed_endpoint set")
            print("  🧠 加载 embedding 模型（首次约 3 秒）...")
            self._model, self._tokenizer = mlx_load(MODEL_ID)
        tokens = self._tokenizer.encode(text)
        input_ids = mx.array([tokens])
        hidden = self._model.model(input_ids)
        pooled = hidden[0, -1, :]
        norm = mx.sqrt(mx.sum(pooled * pooled))
        return [float(x) for x in (pooled / norm)]
