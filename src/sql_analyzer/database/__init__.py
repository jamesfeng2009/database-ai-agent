"""数据库抽象层包."""

from .connector_base import BaseDatabaseConnector, BaseSlowQueryReader
from .factory import create_database_connector, create_slow_query_reader, create_and_connect_database_connector
from .models import (
    DatabaseConfig, 
    DatabaseType, 
    SlowQueryConfig,
    MySQLConfig,
    PostgreSQLConfig,
    TiDBConfig,
    MariaDBConfig,
    OracleConfig,
    SQLServerConfig,
    SQLiteConfig,
    ConnectionStatus,
    DatabaseHealthCheck,
    LoadBalancingStrategy,
    DatabaseConnectionPool,
    DatabaseCluster
)
from .adapters import (
    DatabaseAdapter,
    DatabaseAdapterFactory,
    MySQLAdapter,
    PostgreSQLAdapter,
    TiDBAdapter,
    MariaDBAdapter,
    OracleAdapter,
    SQLServerAdapter,
    SQLiteAdapter
)
from .connection_manager import DatabaseConnectionManager
from .config_manager import DatabaseConfigManager
from .database_manager import DatabaseManager

__all__ = [
    # 基础类
    "BaseDatabaseConnector",
    "BaseSlowQueryReader",
    
    # 工厂函数
    "create_database_connector",
    "create_slow_query_reader",
    "create_and_connect_database_connector",
    
    # 配置模型
    "DatabaseConfig",
    "DatabaseType",
    "SlowQueryConfig",
    "MySQLConfig",
    "PostgreSQLConfig",
    "TiDBConfig",
    "MariaDBConfig",
    "OracleConfig",
    "SQLServerConfig",
    "SQLiteConfig",
    "ConnectionStatus",
    "DatabaseHealthCheck",
    "LoadBalancingStrategy",
    "DatabaseConnectionPool",
    "DatabaseCluster",
    
    # 适配器
    "DatabaseAdapter",
    "DatabaseAdapterFactory",
    "MySQLAdapter",
    "PostgreSQLAdapter",
    "TiDBAdapter",
    "MariaDBAdapter",
    "OracleAdapter",
    "SQLServerAdapter",
    "SQLiteAdapter",
    
    # 管理器
    "DatabaseConnectionManager",
    "DatabaseConfigManager",
    "DatabaseManager"
] 