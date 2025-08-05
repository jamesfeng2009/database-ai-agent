"""SQLite 数据库连接和查询执行模块."""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiosqlite
import sqlparse

from .connector_base import BaseDatabaseConnector, BaseSlowQueryReader
from .models import ExplainResult, SQLiteConfig, SlowQueryConfig, SlowQueryEntry

logger = logging.getLogger(__name__)


class SQLiteConnector(BaseDatabaseConnector):
    """SQLite 数据库连接器."""
    
    def __init__(self, config: SQLiteConfig) -> None:
        """初始化 SQLite 连接器."""
        super().__init__(config)
        self.connection: Optional[aiosqlite.Connection] = None
    
    async def connect(self) -> None:
        """建立数据库连接."""
        try:
            self.connection = await aiosqlite.connect(self.config.database_path)
            # 启用外键约束
            await self.connection.execute("PRAGMA foreign_keys = ON")
            await self.connection.commit()
            logger.info(f"已连接到 SQLite 数据库: {self.config.database_path}")
        except Exception as e:
            logger.error(f"连接 SQLite 数据库失败: {e}")
            raise
    
    async def disconnect(self) -> None:
        """关闭数据库连接."""
        if self.connection:
            await self.connection.close()
            logger.info("SQLite 连接已关闭")
    
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Union[tuple, list, dict]] = None
    ) -> List[Dict[str, Any]]:
        """执行查询并返回结果."""
        if not self.connection:
            raise RuntimeError("数据库连接未初始化，请先调用 connect() 方法")
        
        async with self.connection.execute(sql, params or ()) as cursor:
            rows = await cursor.fetchall()
            # 获取列名
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append(dict(zip(columns, row)))
            return result
    
    async def execute_explain(self, sql: str) -> List[ExplainResult]:
        """执行 EXPLAIN 查询并返回结构化结果."""
        try:
            cleaned_sql = self._clean_and_validate_sql(sql)
            explain_sql = f"EXPLAIN QUERY PLAN {cleaned_sql}"
            
            raw_results = await self.execute_query(explain_sql)
            
            explain_results = []
            for row in raw_results:
                explain_result = ExplainResult(
                    id=row.get('id'),
                    select_type=row.get('detail'),
                    table=self._extract_table_name(row.get('detail', '')),
                    type=self._determine_scan_type(row.get('detail', '')),
                    rows=0,  # SQLite EXPLAIN不提供行数估计
                    extra=row.get('detail')
                )
                explain_results.append(explain_result)
            
            return explain_results
            
        except Exception as e:
            logger.error(f"执行 EXPLAIN 失败: {e}, SQL: {sql[:100]}...")
            raise
    
    def _extract_table_name(self, detail: str) -> str:
        """从EXPLAIN详情中提取表名."""
        if not detail:
            return ""
        
        # SQLite EXPLAIN QUERY PLAN格式: "SCAN TABLE table_name" 或 "SEARCH TABLE table_name"
        parts = detail.split()
        for i, part in enumerate(parts):
            if part.upper() == "TABLE" and i + 1 < len(parts):
                return parts[i + 1]
        
        return ""
    
    def _determine_scan_type(self, detail: str) -> str:
        """根据EXPLAIN详情确定扫描类型."""
        if not detail:
            return "unknown"
        
        detail_upper = detail.upper()
        if "SCAN TABLE" in detail_upper:
            return "ALL"  # 全表扫描
        elif "SEARCH TABLE" in detail_upper or "INDEX" in detail_upper:
            return "index"  # 索引扫描
        else:
            return "unknown"
    
    def _clean_and_validate_sql(self, sql: str) -> str:
        """清理和验证 SQL 语句，防止注入攻击."""
        if not sql or not sql.strip():
            raise ValueError("SQL语句不能为空")
        
        # 长度限制防止过大的SQL语句
        if len(sql) > 100000:
            raise ValueError("SQL语句过长，可能存在安全风险")
        
        # 检查多语句执行（防止; injection）
        statements = sqlparse.split(sql)
        if len(statements) > 1:
            non_empty_statements = [s.strip() for s in statements if s.strip()]
            if len(non_empty_statements) > 1:
                logger.warning(f"检测到多语句SQL: {len(non_empty_statements)} 条语句")
                raise ValueError("不允许执行多条SQL语句")
        
        # 移除注释和多余的空白
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                raise ValueError("无效的SQL语句")
        except Exception as e:
            raise ValueError(f"SQL解析失败: {e}")
        
        cleaned_sql = sqlparse.format(
            str(parsed[0]), 
            strip_comments=True, 
            strip_whitespace=True
        ).strip()
        
        # 基本的安全检查
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
            'TRUNCATE', 'ATTACH', 'DETACH', 'PRAGMA'
        ]
        
        upper_sql = cleaned_sql.upper()
        
        # 检查危险关键字
        for keyword in dangerous_keywords:
            if keyword in upper_sql:
                logger.warning(f"检测到危险关键字: {keyword}")
                if not upper_sql.strip().startswith('SELECT'):
                    raise ValueError(f"EXPLAIN只支持SELECT语句，检测到危险关键字: {keyword}")
        
        # 检查是否为SELECT语句
        if not upper_sql.strip().startswith('SELECT'):
            raise ValueError("EXPLAIN只支持SELECT语句")
        
        return cleaned_sql


class SQLiteSlowQueryReader(BaseSlowQueryReader):
    """SQLite 慢查询日志读取器."""
    
    async def get_slow_queries(self) -> List[SlowQueryEntry]:
        """获取慢查询日志条目."""
        # SQLite没有内置的慢查询日志，返回空列表
        logger.info("SQLite不支持慢查询日志功能")
        return []
    
    async def _get_from_performance_schema(self) -> List[SlowQueryEntry]:
        """从性能模式获取慢查询."""
        # SQLite没有性能模式
        return []
    
    async def _get_from_processlist(self) -> List[SlowQueryEntry]:
        """从进程列表获取当前运行的查询."""
        # SQLite是嵌入式数据库，没有进程列表
        return []