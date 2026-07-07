"""teammate · L1 W3 → W4 —— 长期记忆：向量存储 + 余弦相似度召回。
embed 逻辑抽到 Embedder（src/embedder.py），和 KnowledgeBase 共用。
"""
import json
import os
import time
from src.embedder import Embedder


class VectorMemory:
    """长期记忆：跨 session 持久化对话 + 按语义相关性召回。"""

    def __init__(self, storage_path: str = "data/memory.json",
                 embed_endpoint: str | None = None):
        self.storage_path = storage_path
        self._embedder = Embedder(embed_endpoint)
        self.memories: list[dict] = []
        self._load()

    def embed(self, text: str) -> list[float]:
        """文本 → 1024 维 L2 归一化向量（委托 Embedder）。"""
        return self._embedder(text)

    def add(self, text: str, metadata: dict | None = None):
        """存一条记忆：embed → 向量 + 原文 → 持久化到 JSON。"""
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
        """召回：embed(query) → 和所有存储向量算余弦 → 过滤 → top-K。"""
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
        """归一化向量余弦 = 点积。通用版见注释：
        dot / (sum(x*x)**0.5 * sum(y*y)**0.5)"""
        return sum(x * y for x, y in zip(a, b))

    def format_context(self, memories: list[dict], max_chars: int = 200) -> str:
        """格式化召回记忆注入 system prompt。每条截断到 max_chars。"""
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
