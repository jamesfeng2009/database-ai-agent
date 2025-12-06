"""知识库与简单 RAG 能力实现.

该模块提供基础的 KnowledgeEntry 模型与 KnowledgeService 服务,
使用已有的 BaseDatabaseConnector 将知识条目持久化到业务数据库中,
并通过 LIKE / 简单全文检索提供检索能力。

后续如需接入向量检索, 可在此模块上扩展 EmbeddingBackend 等接口。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...database.connector_base import BaseDatabaseConnector

logger = logging.getLogger(__name__)


class KnowledgeEntry(BaseModel):
    """知识条目模型.

    用于存储历史案例、规范、最佳实践等, 以增强优化与安全建议。
    """

    entry_id: str = Field(..., description="知识条目 ID")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="正文内容")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    source: str = Field(..., description="来源标识, 如 incident/spec/internal_doc 等")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")


@dataclass
class KnowledgeSearchResult:
    """知识检索结果."""

    entry: KnowledgeEntry
    score: float


class KnowledgeService:
    """基于业务数据库的简单知识库 / RAG 服务.

    当前实现使用一张表持久化 KnowledgeEntry, 并通过 LIKE 提供检索能力。
    如需更强大的全文/向量能力, 可在此基础上扩展。
    """

    def __init__(self, connector: BaseDatabaseConnector, table_name: str = "knowledge_entries") -> None:
        self.connector = connector
        self.table_name = table_name

    async def init_schema(self) -> None:
        """创建知识库表 (如果不存在)."""

        db_type = self.connector.database_type.lower()

        if db_type == "mysql":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
              entry_id    VARCHAR(64) PRIMARY KEY,
              title       TEXT        NOT NULL,
              content     LONGTEXT    NOT NULL,
              tags        TEXT        NULL,
              source      VARCHAR(64) NOT NULL,
              created_at  DATETIME    NOT NULL,
              updated_at  DATETIME    NOT NULL
            )
            """
        elif db_type == "postgresql":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
              entry_id   VARCHAR(64) PRIMARY KEY,
              title      TEXT        NOT NULL,
              content    TEXT        NOT NULL,
              tags       TEXT        NULL,
              source     VARCHAR(64) NOT NULL,
              created_at TIMESTAMP   NOT NULL,
              updated_at TIMESTAMP   NOT NULL
            )
            """
        else:
            logger.warning("KnowledgeService.init_schema: unsupported db type: %s", db_type)
            return

        try:
            await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to init knowledge table: %s", exc)

    async def add_entry(self, entry: KnowledgeEntry) -> None:
        """新增知识条目 (存在则覆盖)."""

        await self.init_schema()

        tags_json = json.dumps(entry.tags, ensure_ascii=False).replace("'", "''")
        content_sql = entry.content.replace("'", "''")
        title_sql = entry.title.replace("'", "''")

        db_type = self.connector.database_type.lower()
        if db_type in {"mysql", "postgresql"}:
            sql = f"""
            INSERT INTO {self.table_name} (entry_id, title, content, tags, source, created_at, updated_at)
            VALUES (
              '{entry.entry_id}',
              '{title_sql}',
              '{content_sql}',
              '{tags_json}',
              '{entry.source}',
              '{entry.created_at.isoformat()}',
              '{entry.updated_at.isoformat()}'
            )
            ON CONFLICT (entry_id) DO UPDATE SET
              title = EXCLUDED.title,
              content = EXCLUDED.content,
              tags = EXCLUDED.tags,
              source = EXCLUDED.source,
              created_at = EXCLUDED.created_at,
              updated_at = EXCLUDED.updated_at
            """ if db_type == "postgresql" else f"""
            REPLACE INTO {self.table_name} (entry_id, title, content, tags, source, created_at, updated_at)
            VALUES (
              '{entry.entry_id}',
              '{title_sql}',
              '{content_sql}',
              '{tags_json}',
              '{entry.source}',
              '{entry.created_at.isoformat()}',
              '{entry.updated_at.isoformat()}'
            )
            """
        else:
            logger.warning("KnowledgeService.add_entry: unsupported db type: %s", db_type)
            return

        try:
            await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to add knowledge entry %s: %s", entry.entry_id, exc)

    async def search(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """按关键字检索知识条目 (基于 LIKE 的简单实现)."""

        await self.init_schema()

        q = query.replace("'", "''")
        db_type = self.connector.database_type.lower()

        if db_type in {"mysql", "postgresql"}:
            sql = f"""
            SELECT entry_id, title, content, tags, source, created_at, updated_at
            FROM {self.table_name}
            WHERE title   LIKE '%{q}%'
               OR content LIKE '%{q}%'
               OR tags    LIKE '%{q}%'
            ORDER BY updated_at DESC
            LIMIT {int(limit)}
            """
        else:
            logger.warning("KnowledgeService.search: unsupported db type: %s", db_type)
            return []

        try:
            rows = await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to search knowledge entries: %s", exc)
            return []

        results: List[KnowledgeEntry] = []
        for row in rows:
            try:
                tags_raw = row.get("tags") or "[]"
                tags = json.loads(tags_raw)
            except Exception:
                tags = []

            created = row.get("created_at")
            updated = row.get("updated_at")
            try:
                created_dt = datetime.fromisoformat(created) if isinstance(created, str) else datetime.utcnow()
                updated_dt = datetime.fromisoformat(updated) if isinstance(updated, str) else created_dt
            except Exception:
                created_dt = datetime.utcnow()
                updated_dt = created_dt

            results.append(
                KnowledgeEntry(
                    entry_id=row.get("entry_id"),
                    title=row.get("title", ""),
                    content=row.get("content", ""),
                    tags=tags,
                    source=row.get("source", "unknown"),
                    created_at=created_dt,
                    updated_at=updated_dt,
                )
            )

        return results

    async def related_to_sql(self, sql: str, context: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[KnowledgeEntry]:
        """根据 SQL 与上下文查找相关知识条目.

        当前实现为: 将 SQL 文本与上下文关键信息拼接, 再调用 search。
        后续可替换为向量检索等更智能的实现。
        """

        ctx_parts: List[str] = []
        if context:
            for k, v in context.items():
                try:
                    ctx_parts.append(f"{k}={v}")
                except Exception:
                    continue

        query = sql
        if ctx_parts:
            query += " " + " ".join(ctx_parts)

        return await self.search(query=query, limit=limit)
