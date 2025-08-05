"""PostgreSQL 数据库连接和查询执行模块."""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import asyncpg
import sqlparse

from .connector_base import BaseDatabaseConnector, BaseSlowQueryReader
from .models import ExplainResult, PostgreSQLConfig, SlowQueryConfig, SlowQueryEntry

# 过滤掉asyncpg的警告信息
warnings.filterwarnings("ignore", module="asyncpg")

logger = logging.getLogger(__name__)


def _safe_float_convert(value: Any, default: float = 0.0) -> float:
    """安全地将值转换为浮点数."""
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"无法将值 {value} 转换为浮点数，使用默认值 {default}: {e}")
        return default


def _safe_int_convert(value: Any, default: int = 0) -> int:
    """安全地将值转换为整数."""
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"无法将值 {value} 转换为整数，使用默认值 {default}: {e}")
        return default


def _safe_str_convert(value: Any, default: str = "") -> str:
    """安全地将值转换为字符串."""
    if value is None:
        return default
    
    try:
        return str(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"无法将值 {value} 转换为字符串，使用默认值 {default}: {e}")
        return default


def _validate_and_convert_timestamp(timestamp_value: Any) -> datetime:
    """验证并转换时间戳为 datetime 对象."""
    if timestamp_value is None:
        logger.debug("时间戳为空，使用当前时间")
        return datetime.now()
    
    try:
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        elif isinstance(timestamp_value, (int, float)):
            return datetime.fromtimestamp(timestamp_value)
        else:
            return datetime.now()
        
    except (OSError, OverflowError, ValueError) as e:
        logger.warning(f"时间戳转换失败: {timestamp_value}, 错误: {e}，使用当前时间")
        return datetime.now()


class PostgreSQLConnector(BaseDatabaseConnector):
    """PostgreSQL 数据库连接器."""
    
    def __init__(self, config: PostgreSQLConfig) -> None:
        """初始化 PostgreSQL 连接器."""
        super().__init__(config)
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """建立数据库连接池."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                command_timeout=self.config.connect_timeout,
                min_size=1,
                max_size=self.config.max_connections,
                ssl=self.config.ssl_mode if self.config.ssl_mode != "disable" else None,
                application_name=self.config.application_name
            )
            logger.info(f"已连接到 PostgreSQL 数据库: {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.error(f"连接 PostgreSQL 数据库失败: {e}")
            raise
    
    async def disconnect(self) -> None:
        """关闭数据库连接池."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL 连接池已关闭")
    
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Union[tuple, list, dict]] = None
    ) -> List[Dict[str, Any]]:
        """执行查询并返回结果."""
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化，请先调用 connect() 方法")
        
        async with self.pool.acquire() as conn:
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)
            
            # 转换为字典列表
            result = []
            for row in rows:
                result.append(dict(row))
            return result
    
    async def execute_explain(self, sql: str) -> List[ExplainResult]:
        """执行 EXPLAIN 查询并返回结构化结果."""
        try:
            cleaned_sql = self._clean_and_validate_sql(sql)
            explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {cleaned_sql}"
            
            raw_results = await self.execute_query(explain_sql)
            
            explain_results = []
            for row in raw_results:
                # PostgreSQL的EXPLAIN返回JSON格式，需要解析
                plan_data = row.get('QUERY PLAN', [])
                if isinstance(plan_data, list) and plan_data:
                    # 解析JSON格式的执行计划
                    explain_results.extend(self._parse_postgresql_explain(plan_data))
                else:
                    # 如果返回的是文本格式，尝试解析
                    logger.warning("PostgreSQL EXPLAIN 返回格式不支持，使用基础解析")
            
            return explain_results
            
        except Exception as e:
            logger.error(f"执行 EXPLAIN 失败: {e}, SQL: {sql[:100]}...")
            # 回退到简单的EXPLAIN
            try:
                cleaned_sql = self._clean_and_validate_sql(sql)
                explain_sql = f"EXPLAIN {cleaned_sql}"
                raw_results = await self.execute_query(explain_sql)
                
                explain_results = []
                for row in raw_results:
                    explain_result = ExplainResult(
                        table=row.get('QUERY PLAN', ''),
                        type='unknown',
                        rows=0,
                        extra=row.get('QUERY PLAN', '')
                    )
                    explain_results.append(explain_result)
                
                return explain_results
            except Exception as fallback_error:
                logger.error(f"回退 EXPLAIN 也失败: {fallback_error}")
                raise
    
    def _parse_postgresql_explain(self, plan_data: List[Dict]) -> List[ExplainResult]:
        """解析 PostgreSQL 的 JSON 格式执行计划."""
        explain_results = []
        
        def parse_node(node: Dict, node_id: int = 0) -> ExplainResult:
            """递归解析执行计划节点."""
            node_type = node.get('Node Type', 'Unknown')
            
            # 提取表名
            table_name = node.get('Relation Name', '')
            if not table_name and 'Scan' in node_type:
                # 对于扫描操作，尝试从其他字段获取表名
                table_name = node.get('Index Name', '') or node.get('CTE Name', '')
            
            # 提取行数信息
            rows = node.get('Plan Rows', 0)
            actual_rows = node.get('Actual Rows', 0)
            
            # 提取成本信息
            startup_cost = node.get('Startup Cost', 0.0)
            total_cost = node.get('Total Cost', 0.0)
            
            # 提取额外信息
            extra_info = []
            if node.get('Parallel Aware'):
                extra_info.append("Parallel Aware")
            if node.get('Async Capable'):
                extra_info.append("Async Capable")
            if node.get('Workers Planned'):
                extra_info.append(f"Workers Planned: {node.get('Workers Planned')}")
            
            # 提取索引信息
            index_name = node.get('Index Name', '')
            scan_direction = node.get('Scan Direction', '')
            
            # 构建连接类型
            connection_type = node_type
            if 'Index' in node_type:
                connection_type = 'index'
            elif 'Seq' in node_type:
                connection_type = 'ALL'  # 对应MySQL的全表扫描
            
            return ExplainResult(
                id=node_id,
                select_type=node_type,
                table=table_name,
                type=connection_type,
                key=index_name,
                rows=rows,
                startup_cost=startup_cost,
                total_cost=total_cost,
                actual_rows=actual_rows,
                extra='; '.join(extra_info) if extra_info else None
            )
        
        # 递归解析执行计划树
        def parse_plan_tree(plan: Dict, node_id: int = 0) -> List[ExplainResult]:
            results = []
            
            # 解析当前节点
            current_node = parse_node(plan, node_id)
            results.append(current_node)
            
            # 递归解析子节点
            if 'Plans' in plan:
                for i, child_plan in enumerate(plan['Plans']):
                    child_results = parse_plan_tree(child_plan, node_id + i + 1)
                    results.extend(child_results)
            
            return results
        
        # 解析根节点
        if plan_data:
            explain_results = parse_plan_tree(plan_data[0])
        
        return explain_results
    
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
            'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE',
            'COPY', 'VACUUM', 'ANALYZE', 'REINDEX'
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


class PostgreSQLSlowQueryReader(BaseSlowQueryReader):
    """PostgreSQL 慢查询日志读取器."""
    
    async def get_slow_queries(self) -> List[SlowQueryEntry]:
        """获取慢查询日志条目."""
        if self.config.use_performance_schema:
            return await self._get_from_performance_schema()
        else:
            return await self._get_from_log_file()
    
    async def _get_from_performance_schema(self) -> List[SlowQueryEntry]:
        """从 pg_stat_statements 获取慢查询."""
        sql = """
        SELECT 
            mean_exec_time / 1000.0 as query_time,
            0.0 as lock_time,
            calls as rows_sent,
            rows as rows_examined,
            query as sql_statement,
            EXTRACT(EPOCH FROM NOW()) as timestamp_value,
            usename as user,
            client_addr::text as host,
            current_database() as database_name
        FROM pg_stat_statements
        JOIN pg_user ON pg_stat_statements.userid = pg_user.usesysid
        WHERE mean_exec_time / 1000.0 >= $1
        AND rows >= $2
        AND query NOT LIKE $3
        AND query NOT LIKE $4
        AND current_database() = $5
        ORDER BY mean_exec_time DESC
        LIMIT $6
        """
        
        try:
            params = (
                self.config.query_time_threshold,
                self.config.rows_examined_threshold,
                'SHOW%',
                'EXPLAIN%',
                self.connector.config.database,
                self.config.limit
            )
            
            logger.info(f"从 pg_stat_statements 查询最近 {self.config.time_range_hours} 小时内的慢查询...")
            results = await self.connector.execute_query(sql, params)
            
            if len(results) > 0:
                logger.info(f"找到 {len(results)} 条符合条件的慢查询")
            else:
                logger.info("未找到任何符合条件的慢查询记录")
            
            slow_queries = []
            
            for row in results:
                timestamp = _validate_and_convert_timestamp(row.get('timestamp_value'))
                
                slow_query = SlowQueryEntry(
                    query_time=_safe_float_convert(row.get('query_time'), 0.0),
                    lock_time=_safe_float_convert(row.get('lock_time'), 0.0),
                    rows_sent=_safe_int_convert(row.get('rows_sent'), 0),
                    rows_examined=_safe_int_convert(row.get('rows_examined'), 0),
                    sql_statement=_safe_str_convert(row.get('sql_statement'), ""),
                    timestamp=timestamp,
                    user=row.get('user'),
                    host=row.get('host'),
                    database=row.get('database_name')
                )
                slow_queries.append(slow_query)
                logger.debug(f"获取到慢查询: {slow_query.sql_statement[:100]}... (时间: {slow_query.query_time}s)")
            
            logger.info(f"从 pg_stat_statements 获取到 {len(slow_queries)} 条慢查询")
            return slow_queries
            
        except Exception as e:
            logger.error(f"从 pg_stat_statements 读取慢查询失败: {e}")
            return await self._get_from_processlist()
    
    async def _get_from_processlist(self) -> List[SlowQueryEntry]:
        """从 pg_stat_activity 获取当前运行的查询."""
        sql = """
        SELECT 
            pid,
            usename,
            client_addr::text as host,
            datname,
            state,
            EXTRACT(EPOCH FROM (NOW() - query_start)) as query_time,
            query
        FROM pg_stat_activity
        WHERE state = $1
        AND EXTRACT(EPOCH FROM (NOW() - query_start)) >= $2
        AND query IS NOT NULL
        AND query NOT LIKE $3
        AND datname = $4
        ORDER BY query_start DESC
        LIMIT $5
        """
        
        try:
            params = (
                'active',
                self.config.query_time_threshold, 
                'SHOW%',
                self.connector.config.database,
                self.config.limit
            )
            results = await self.connector.execute_query(sql, params)
            
            slow_queries = []
            current_time = datetime.now()
            
            for row in results:
                slow_query = SlowQueryEntry(
                    query_time=_safe_float_convert(row.get('query_time'), 0.0),
                    lock_time=0.0,
                    rows_sent=0,
                    rows_examined=0,
                    sql_statement=_safe_str_convert(row.get('query'), ""),
                    timestamp=current_time,
                    user=row.get('usename'),
                    host=row.get('host'),
                    database=row.get('datname')
                )
                slow_queries.append(slow_query)
            
            logger.info(f"从 pg_stat_activity 获取到 {len(slow_queries)} 条当前运行的查询")
            return slow_queries
            
        except Exception as e:
            logger.error(f"从 pg_stat_activity 读取查询失败: {e}")
            return [] 