"""数据库抽象层基类."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .models import DatabaseConfig, ExplainResult, SlowQueryConfig, SlowQueryEntry

logger = logging.getLogger(__name__)


class BaseDatabaseConnector(ABC):
    """数据库连接器抽象基类.
    
    定义了所有数据库连接器必须实现的公共接口。
    """
    
    def __init__(self, config: DatabaseConfig) -> None:
        """初始化数据库连接器.
        
        Args:
            config: 数据库连接配置
        """
        self.config = config
        self.database_type = config.get_database_type()
    
    @abstractmethod
    async def connect(self) -> None:
        """建立数据库连接池."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """关闭数据库连接池."""
        pass
    
    @abstractmethod
    async def execute_query(
        self, 
        sql: str, 
        params: Optional[Union[tuple, list, dict]] = None
    ) -> List[Dict[str, Any]]:
        """执行查询并返回结果.
        
        Args:
            sql: SQL 查询语句
            params: 查询参数，支持参数化查询防止SQL注入
            
        Returns:
            查询结果列表
        """
        pass
    
    @abstractmethod
    async def execute_explain(self, sql: str) -> List[ExplainResult]:
        """执行 EXPLAIN 查询并返回结构化结果.
        
        Args:
            sql: 要分析的 SQL 语句
            
        Returns:
            EXPLAIN 结果列表
        """
        pass
    
    @abstractmethod
    def _clean_and_validate_sql(self, sql: str) -> str:
        """清理和验证 SQL 语句，防止注入攻击.
        
        Args:
            sql: 原始 SQL 语句
            
        Returns:
            清理后的 SQL 语句
        """
        pass
    
    async def test_connection(self) -> bool:
        """测试数据库连接.
        
        Returns:
            连接是否成功
        """
        try:
            await self.execute_query("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False


class BaseSlowQueryReader(ABC):
    """慢查询日志读取器抽象基类.
    
    定义了所有慢查询读取器必须实现的公共接口。
    """
    
    def __init__(self, connector: BaseDatabaseConnector, config: SlowQueryConfig) -> None:
        """初始化慢查询读取器.
        
        Args:
            connector: 数据库连接器
            config: 慢查询配置
        """
        self.connector = connector
        self.config = config
        self.database_type = connector.database_type
    
    @abstractmethod
    async def get_slow_queries(self) -> List[SlowQueryEntry]:
        """获取慢查询日志条目.
        
        Returns:
            慢查询条目列表
        """
        pass
    
    @abstractmethod
    async def _get_from_performance_schema(self) -> List[SlowQueryEntry]:
        """从性能模式获取慢查询.
        
        Returns:
            慢查询条目列表
        """
        pass
    
    @abstractmethod
    async def _get_from_processlist(self) -> List[SlowQueryEntry]:
        """从进程列表获取当前运行的查询.
        
        Returns:
            当前查询条目列表
        """
        pass
    
    async def _get_from_log_file(self) -> List[SlowQueryEntry]:
        """从慢查询日志文件读取.
        
        Returns:
            慢查询条目列表
            
        Raises:
            NotImplementedError: 文件读取功能暂未实现
        """
        logger.warning("从文件读取慢查询日志功能暂未实现")
        raise NotImplementedError("从文件读取慢查询日志功能暂未实现") 