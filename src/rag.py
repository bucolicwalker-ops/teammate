"""teammate · L2 W4 —— RAG 知识库：文档加载→切块→embedding→向量存储→检索。
pipeline: load(file) → chunk(text, mode) → batch embed(chunks) → store → retrieve(query, top_k)

切块模式：
- "fixed"：固定大小 + overlap（通用，任何文档）
- "semantic"：按 markdown 标题切（#### 级别，保持问题/段落完整，不断义）
"""
import json
import os
import re
from src.embedder import Embedder


class KnowledgeBase:
    """RAG 知识库：文档 → chunk → 向量 → 检索 → 注入 prompt。"""

    def __init__(self, storage_path: str = "data/knowledge.json",
                 embed_endpoint: str | None = None):
        self.storage_path = storage_path
        self._embedder = Embedder(embed_endpoint)
        self.chunks: list[dict] = []
        self._load()

    def load_document(self, file_path: str, chunk_size: int = 500,
                      overlap: int = 50, mode: str = "semantic") -> int:
        """加载文档 → 切块 → batch embed → 存。返回 chunk 数。

        mode="semantic"：按 #### 标题切（保持问题/段落完整）
        mode="fixed"：固定大小 + overlap（通用）
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if mode == "semantic":
            text_chunks = self._chunk_by_headers(text)
        else:
            text_chunks = self._chunk_text(text, chunk_size, overlap)

        print(f"  📄 {os.path.basename(file_path)}: {len(text_chunks)} chunks (mode={mode})")

        # batch embed：一次 HTTP 请求 embed 所有 chunk（P2-B 优化）
        vectors = self._embedder.batch(text_chunks)

        source = os.path.basename(file_path)
        for i, (chunk_text, vec) in enumerate(zip(text_chunks, vectors)):
            self.chunks.append({
                "text": chunk_text,
                "vector": vec,
                "source": source,
                "chunk_idx": i,
            })
        self._save()
        return len(text_chunks)

    def retrieve(self, query: str, top_k: int = 3,
                 threshold: float = 0.3) -> list[dict]:
        """检索相关 chunk：embed query → 余弦 → 过阈值 → top-K。"""
        if not self.chunks:
            return []
        q_vec = self._embedder(query)
        scored = []
        for chunk in self.chunks:
            score = self._cosine(q_vec, chunk["vector"])
            if score >= threshold:
                scored.append({**chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_context(self, chunks: list[dict], max_chars: int = 500) -> str:
        """格式化检索结果注入 prompt。带来源标注（grounding/溯源）。"""
        if not chunks:
            return ""
        lines = ["\n[知识库检索结果（按相关性排序，回答须基于此内容并标注来源）]"]
        for c in chunks:
            text = c["text"][:max_chars]
            lines.append(f"- [来源:{c['source']}] {text}（相关度 {c['score']:.2f}）")
        lines.append("[/知识库检索结果]\n")
        return "\n".join(lines)

    @staticmethod
    def _chunk_by_headers(text: str, level: str = "####") -> list[str]:
        """语义切块：按 markdown 标题切，保持每个问题/段落完整。

        比 fixed-size 好在哪：固定切块可能把一个问题的要点和关联卡片切到两个 chunk，
        检索时只召回半个问题。按标题切保证每个 chunk 是完整的"问题+要点+关联"单元。
        """
        lines = text.split("\n")
        chunks = []
        current = []

        for line in lines:
            if line.strip().startswith(level):
                if current:
                    chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            chunks.append("\n".join(current))

        # 过滤掉空 chunk（文件头部的 header 之前可能有内容）
        return [c for c in chunks if c.strip()]

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """固定大小切块 + overlap。深挖版：语义切块（见 _chunk_by_headers）。"""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """归一化向量余弦 = 点积。"""
        return sum(x * y for x, y in zip(a, b))

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️ 知识库文件损坏，从空开始: {e}")
                self.chunks = []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        tmp_path = self.storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        os.rename(tmp_path, self.storage_path)
