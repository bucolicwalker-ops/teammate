"""teammate · L1 W3 → W4 —— 长期记忆：向量存储 + 余弦相似度召回。
embed 逻辑抽到 Embedder（src/embedder.py），和 KnowledgeBase 共用。
持久化：SQLite（原子写入、并发安全、可按 session_id 索引查询）。
"""
import json
import os
import sqlite3
import time
from src.embedder import Embedder


class VectorMemory:
    """长期记忆：跨 session 持久化对话 + 按语义相关性召回。"""

    def __init__(self, storage_path: str = "data/memory.db",
                 embed_endpoint: str | None = None):
        self.storage_path = storage_path
        self._embedder = Embedder(embed_endpoint)
        self.memories: list[dict] = []
        self._db = None
        self._init_db()
        self._migrate_from_json()
        self._load()

    def _init_db(self):
        """初始化 SQLite 表结构。"""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        self._db = sqlite3.connect(self.storage_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                metadata TEXT
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)"
        )
        self._db.commit()

    def _migrate_from_json(self):
        """如果旧 JSON 文件存在且 DB 为空，导入旧数据。"""
        json_path = self.storage_path.replace('.db', '.json')
        if not os.path.exists(json_path):
            return
        count = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > 0:
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_memories = json.load(f)
            for mem in old_memories:
                self._db.execute(
                    "INSERT INTO memories (text, vector, timestamp, session_id, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (mem.get("text", ""), json.dumps(mem.get("vector", [])),
                     mem.get("timestamp", time.time()),
                     mem.get("session_id"),
                     json.dumps(mem.get("metadata", {})))
                )
            self._db.commit()
            print(f"  📦 从旧 JSON 迁移了 {len(old_memories)} 条记忆")
        except Exception as e:
            print(f"  ⚠️ JSON 迁移失败（不影响使用）: {e}")

    def embed(self, text: str) -> list[float]:
        """文本 → 1024 维 L2 归一化向量（委托 Embedder）。"""
        return self._embedder(text)

    def add(self, text: str, metadata: dict | None = None, session_id: str | None = None):
        """存一条记忆：embed → 向量 + 原文 → SQLite 持久化。

        session_id：可选，标记记忆所属会话。retrieve 时可按 session 过滤。
        """
        vec = self.embed(text)
        entry = {
            "text": text,
            "vector": vec,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "session_id": session_id,
        }
        self._db.execute(
            "INSERT INTO memories (text, vector, timestamp, session_id, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, json.dumps(vec), entry["timestamp"], session_id,
             json.dumps(metadata or {}))
        )
        self._db.commit()
        self.memories.append(entry)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.5,
                 session_id: str | None = None) -> list[dict]:
        """召回：embed(query) → 和所有存储向量算余弦 → 过滤 → top-K。

        session_id：可选，只召回指定会话的记忆。None = 不限（全局召回）。
        """
        if not self.memories:
            return []
        q_vec = self.embed(query)
        scored = []
        for mem in self.memories:
            if session_id is not None and mem.get("session_id") != session_id:
                continue
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
        """从 SQLite 加载全部记忆到内存（供余弦检索扫描）。"""
        rows = self._db.execute(
            "SELECT text, vector, timestamp, session_id, metadata FROM memories"
        ).fetchall()
        self.memories = [
            {
                "text": r[0],
                "vector": json.loads(r[1]),
                "timestamp": r[2],
                "session_id": r[3],
                "metadata": json.loads(r[4] or "{}"),
            }
            for r in rows
        ]

    def close(self):
        """关闭数据库连接。"""
        if self._db:
            self._db.close()
            self._db = None
