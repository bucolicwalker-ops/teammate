"""teammate · L1 W3 —— 长期记忆：embedding + 向量存储 + 余弦相似度召回。
用 MLX 本地跑 Qwen3-Embedding-0.6B（不装 torch，Apple Silicon 原生）。
"""
import json
import os
import time
import mlx.core as mx
from mlx_lm import load as mlx_load


class VectorMemory:
    """长期记忆：跨 session 持久化对话 + 按语义相关性召回。

    pipeline: embed(text) → L2 归一化 → 存 JSON → 检索时 embed(query)
    → 和所有存储向量算余弦相似度 → top-K。

    归一化优化：存的时候预归一化（|v|=1），检索时余弦=点积，省一次模长计算。
    """

    MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    def __init__(self, storage_path: str = "data/memory.json"):
        self.storage_path = storage_path
        self._model = None
        self._tokenizer = None
        self.memories: list[dict] = []
        self._load()

    def _ensure_model(self):
        if self._model is None:
            print("  🧠 加载 embedding 模型（首次约 3 秒）...")
            self._model, self._tokenizer = mlx_load(self.MODEL_ID)

    def embed(self, text: str) -> list[float]:
        """文本 → 1024 维向量 → L2 归一化 → 返回 list[float]。

        last_token_pool：取最后一个 token 的 hidden state（decoder-only 模型做
        embedding 的特点，和 BGE 的 CLS pooling 不同）。
        """
        self._ensure_model()
        tokens = self._tokenizer.encode(text)
        input_ids = mx.array([tokens])
        hidden = self._model.model(input_ids)      # (1, seq_len, 1024)
        pooled = hidden[0, -1, :]                    # last_token_pool → (1024,)
        norm = mx.sqrt(mx.sum(pooled * pooled))
        normalized = pooled / norm                   # L2 归一化
        return [float(x) for x in normalized]       # bfloat16 → float for JSON

    def add(self, text: str, metadata: dict | None = None):
        """存一条记忆：embed → 归一化向量 + 原文 → 持久化到 JSON。"""
        vec = self.embed(text)
        entry = {
            "text": text,
            "vector": vec,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.memories.append(entry)
        self._save()

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.5) -> list[dict]:
        """召回：embed(query) → 和所有存储向量算余弦 → 过滤 → top-K。

        threshold：相似度低于此值的记忆不返回（防噪声）。
        归一化后余弦=点积，直接算 dot product。
        """
        if not self.memories:
            return []
        q_vec = self.embed(query)
        scored = []
        for mem in self.memories:
            score = self._cosine_similarity(q_vec, mem["vector"])
            if score >= threshold:
                scored.append({**mem, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度 = A·B / (|A| × |B|)。

        归一化优化：embed() 已对向量做 L2 归一化（|v|=1），
        所以余弦 = 点积，直接算 dot product 省 ~2x 计算。

        通用版（不假设归一化）：
          dot = sum(x*y for x,y in zip(a,b))
          return dot / (sum(x*x for x in a)**0.5 * sum(x*x for x in b)**0.5)
        """
        return sum(x * y for x, y in zip(a, b))

    def format_context(self, memories: list[dict], max_chars: int = 200) -> str:
        """把召回的记忆格式化成 system prompt 注入文本。

        每条截断到 max_chars 防止 token 膨胀（K=3 × 200 字 = 600 字上限）。
        """
        if not memories:
            return ""
        lines = ["\n[过往记忆（按相关性排序）]"]
        for m in memories:
            text = m["text"][:max_chars]
            score = m.get("score", 0)
            lines.append(f"- {text}（相关度 {score:.2f}）")
        lines.append("[/过往记忆]\n")
        return "\n".join(lines)

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️ 记忆文件损坏，从空开始: {e}")
                self.memories = []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        tmp_path = self.storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False)
        os.rename(tmp_path, self.storage_path)
