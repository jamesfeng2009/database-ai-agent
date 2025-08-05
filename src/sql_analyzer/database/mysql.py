"""MySQL 数据库连接和查询执行模块."""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiomysql
import sqlparse

from .connector_base import BaseDatabaseConnector, BaseSlowQueryReader
from .models import ExplainResult, MySQLConfig, SlowQueryConfig, SlowQueryEntry

# 过滤掉aiomysql的MySQL警告信息
warnings.filterwarnings("ignore", category=aiomysql.Warning)
warnings.filterwarnings("ignore", module="aiomysql")
warnings.filterwarnings("ignore", module="pymysql")
warnings.filterwarnings("ignore", message=".*select#.*")

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


def _validate_and_convert_timestamp(timestamp_micro: Any) -> datetime:
    """验证并转换时间戳为 datetime 对象."""
    if timestamp_micro is None:
        logger.debug("时间戳为空，使用当前时间")
        return datetime.now()
    
    try:
        timestamp_value = _safe_float_convert(timestamp_micro, 0.0)
        
        if timestamp_value <= 0:
            logger.debug("时间戳无效，使用当前时间")
            return datetime.now()
        
        # 检查时间戳是否在合理范围内（1970年到2100年）
        if timestamp_value < 0 or timestamp_value > 4102444800000000:
            logger.warning(f"时间戳超出合理范围: {timestamp_value}，使用当前时间")
            return datetime.now()
        
        # 转换微秒时间戳为秒
        timestamp_seconds = timestamp_value / 1000000
        
        return datetime.fromtimestamp(timestamp_seconds)
        
    except (OSError, OverflowError, ValueError) as e:
        logger.warning(f"时间戳转换失败: {timestamp_micro}, 错误: {e}，使用当前时间")
        return datetime.now()


class MySQLConnector(BaseDatabaseConnector):
    """MySQL 数据库连接器."""
    
    def __init__(self, config: MySQLConfig) -> None:
        """初始化 MySQL 连接器."""
        super().__init__(config)
        self.pool: Optional[aiomysql.Pool] = None
    
    async def connect(self) -> None:
        """建立数据库连接池."""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db=self.config.database,
                charset=self.config.charset,
                minsize=1,
                maxsize=self.config.max_connections,
                connect_timeout=self.config.connect_timeout,
                autocommit=True
            )
            logger.info(f"已连接到 MySQL 数据库: {self.config.host}:{self.config.port}/{self.config.database}")
        except Exception as e:
            logger.error(f"连接 MySQL 数据库失败: {e}")
            raise
    
    async def disconnect(self) -> None:
        """关闭数据库连接池."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("MySQL 连接池已关闭")
    
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Union[tuple, list, dict]] = None
    ) -> List[Dict[str, Any]]:
        """执行查询并返回结果."""
        if not self.pool:
            raise RuntimeError("数据库连接池未初始化，请先调用 connect() 方法")
        
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, params)
                result = await cursor.fetchall()
                return list(result)
    
    async def execute_explain(self, sql: str) -> List[ExplainResult]:
        """执行 EXPLAIN 查询并返回结构化结果."""
        try:
            cleaned_sql = self._clean_and_validate_sql(sql)
            explain_sql = f"EXPLAIN {cleaned_sql}"
            
            raw_results = await self.execute_query(explain_sql)
            
            explain_results = []
            for row in raw_results:
                explain_result = ExplainResult(
                    id=row.get('id'),
                    select_type=row.get('select_type'),
                    table=row.get('table'),
                    partitions=row.get('partitions'),
                    type=row.get('type'),
                    possible_keys=row.get('possible_keys'),
                    key=row.get('key'),
                    key_len=row.get('key_len'),
                    ref=row.get('ref'),
                    rows=row.get('rows'),
                    filtered=row.get('filtered'),
                    extra=row.get('Extra')
                )
                explain_results.append(explain_result)
            
            return explain_results
            
        except Exception as e:
            logger.error(f"执行 EXPLAIN 失败: {e}, SQL: {sql[:100]}...")
            raise
    
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
            'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE',
            'SCRIPT', 'SYSTEM', 'SHUTDOWN'
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


class MySQLSlowQueryReader(BaseSlowQueryReader):
    """MySQL 慢查询日志读取器."""
    
    async def get_slow_queries(self) -> List[SlowQueryEntry]:
        """获取慢查询日志条目."""
        if self.config.use_performance_schema:
            return await self._get_from_performance_schema()
        else:
            return await self._get_from_log_file()
    
    async def _get_from_performance_schema(self) -> List[SlowQueryEntry]:
        """从 performance_schema 获取慢查询."""
        sql = """
        SELECT 
            TIMER_WAIT / 1000000000000 as query_time,
            LOCK_TIME / 1000000000000 as lock_time,
            ROWS_SENT as rows_sent,
            ROWS_EXAMINED as rows_examined,
            SQL_TEXT as sql_statement,
            TIMER_START / 1000000000000 as timestamp_micro,
            SUBSTRING_INDEX(USER(), '@', 1) as user,
            SUBSTRING_INDEX(USER(), '@', -1) as host,
            CURRENT_SCHEMA as database_name
        FROM performance_schema.events_statements_history_long
        WHERE TIMER_WAIT / 1000000000000 >= %s
        AND ROWS_EXAMINED >= %s
        AND SQL_TEXT IS NOT NULL
        AND SQL_TEXT NOT LIKE %s
        AND SQL_TEXT NOT LIKE %s
        AND CURRENT_SCHEMA = %s
        ORDER BY TIMER_WAIT DESC
        LIMIT %s
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
            
            logger.info(f"从 performance_schema 查询最近 {self.config.time_range_hours} 小时内的慢查询...")
            results = await self.connector.execute_query(sql, params)
            
            if len(results) > 0:
                logger.info(f"找到 {len(results)} 条符合条件的慢查询")
            else:
                logger.info("未找到任何符合条件的慢查询记录")
            
            slow_queries = []
            
            for row in results:
                timestamp = _validate_and_convert_timestamp(row.get('timestamp_micro'))
                
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
            
            logger.info(f"从 performance_schema 获取到 {len(slow_queries)} 条慢查询")
            return slow_queries
            
        except Exception as e:
            logger.error(f"从 performance_schema 读取慢查询失败: {e}")
            return await self._get_from_processlist()
    
    async def _get_from_processlist(self) -> List[SlowQueryEntry]:
        """从 SHOW PROCESSLIST 获取当前运行的查询."""
        sql = """
        SELECT 
            ID,
            USER,
            HOST,
            DB,
            COMMAND,
            TIME,
            STATE,
            INFO
        FROM INFORMATION_SCHEMA.PROCESSLIST
        WHERE COMMAND = %s
        AND TIME >= %s
        AND INFO IS NOT NULL
        AND INFO NOT LIKE %s
        AND DB = %s
        ORDER BY TIME DESC
        LIMIT %s
        """
        
        try:
            params = (
                'Query',
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
                    query_time=_safe_float_convert(row.get('TIME'), 0.0),
                    lock_time=0.0,
                    rows_sent=0,
                    rows_examined=0,
                    sql_statement=_safe_str_convert(row.get('INFO'), ""),
                    timestamp=current_time,
                    user=row.get('USER'),
                    host=row.get('HOST'),
                    database=row.get('DB')
                )
                slow_queries.append(slow_query)
            
            logger.info(f"从 PROCESSLIST 获取到 {len(slow_queries)} 条当前运行的查询")
            return slow_queries
            
        except Exception as e:
            logger.error(f"从 PROCESSLIST 读取查询失败: {e}")
            return [] 