"""数据库工厂模块."""

import logging
from typing import Optional

from .connector_base import BaseDatabaseConnector, BaseSlowQueryReader
from .models import (
    DatabaseConfig, 
    DatabaseType, 
    SlowQueryConfig,
    TiDBConfig,
    MariaDBConfig,
    OracleConfig,
    SQLServerConfig,
    SQLiteConfig
)

logger = logging.getLogger(__name__)


def create_database_connector(config: DatabaseConfig) -> BaseDatabaseConnector:
    """创建数据库连接器.
    
    Args:
        config: 数据库配置
        
    Returns:
        数据库连接器实例
        
    Raises:
        ValueError: 当数据库类型不支持时
    """
    database_type = config.get_database_type()
    
    if database_type == DatabaseType.MYSQL:
        from .mysql import MySQLConnector
        return MySQLConnector(config)
    elif database_type == DatabaseType.POSTGRESQL:
        from .postgresql import PostgreSQLConnector
        return PostgreSQLConnector(config)
    elif database_type == DatabaseType.TIDB:
        # TiDB使用MySQL协议，复用MySQL连接器
        from .mysql import MySQLConnector
        return MySQLConnector(config)
    elif database_type == DatabaseType.MARIADB:
        # MariaDB使用MySQL协议，复用MySQL连接器
        from .mysql import MySQLConnector
        return MySQLConnector(config)
    elif database_type == DatabaseType.SQLITE:
        from .sqlite import SQLiteConnector
        return SQLiteConnector(config)
    elif database_type == DatabaseType.ORACLE:
        from .oracle import OracleConnector
        return OracleConnector(config)
    elif database_type == DatabaseType.SQLSERVER:
        from .sqlserver import SQLServerConnector
        return SQLServerConnector(config)
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")


def create_slow_query_reader(
    connector: BaseDatabaseConnector, 
    config: SlowQueryConfig
) -> BaseSlowQueryReader:
    """创建慢查询读取器.
    
    Args:
        connector: 数据库连接器
        config: 慢查询配置
        
    Returns:
        慢查询读取器实例
        
    Raises:
        ValueError: 当数据库类型不支持时
    """
    database_type = connector.database_type
    
    if database_type == DatabaseType.MYSQL:
        from .mysql import MySQLSlowQueryReader
        return MySQLSlowQueryReader(connector, config)
    elif database_type == DatabaseType.POSTGRESQL:
        from .postgresql import PostgreSQLSlowQueryReader
        return PostgreSQLSlowQueryReader(connector, config)
    elif database_type in [DatabaseType.TIDB, DatabaseType.MARIADB]:
        # TiDB和MariaDB使用MySQL协议，复用MySQL慢查询读取器
        from .mysql import MySQLSlowQueryReader
        return MySQLSlowQueryReader(connector, config)
    elif database_type == DatabaseType.SQLITE:
        from .sqlite import SQLiteSlowQueryReader
        return SQLiteSlowQueryReader(connector, config)
    elif database_type == DatabaseType.ORACLE:
        from .oracle import OracleSlowQueryReader
        return OracleSlowQueryReader(connector, config)
    elif database_type == DatabaseType.SQLSERVER:
        from .sqlserver import SQLServerSlowQueryReader
        return SQLServerSlowQueryReader(connector, config)
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")


async def create_and_connect_database_connector(config: DatabaseConfig) -> BaseDatabaseConnector:
    """创建并连接数据库连接器.
    
    Args:
        config: 数据库配置
        
    Returns:
        已连接的数据库连接器实例
    """
    connector = create_database_connector(config)
    await connector.connect()
    return connector 