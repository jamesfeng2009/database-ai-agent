"""数据库抽象层的数据模型."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class DatabaseType(str, Enum):
    """数据库类型枚举."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    TIDB = "tidb"
    MARIADB = "mariadb"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"


class DatabaseConfig(BaseModel, ABC):
    """数据库连接配置基类."""
    
    host: str = Field(..., description="数据库主机地址")
    port: int = Field(..., description="数据库端口")
    user: str = Field(..., description="数据库用户名")
    password: str = Field(..., description="数据库密码")
    database: str = Field(..., description="数据库名称")
    connect_timeout: int = Field(default=10, description="连接超时时间（秒）")
    max_connections: int = Field(default=10, description="最大连接数")
    
    @abstractmethod
    def get_database_type(self) -> DatabaseType:
        """获取数据库类型."""
        pass


class MySQLConfig(DatabaseConfig):
    """MySQL 数据库连接配置."""
    
    port: int = Field(default=3306, description="数据库端口")
    charset: str = Field(default="utf8mb4", description="字符集")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.MYSQL


class PostgreSQLConfig(DatabaseConfig):
    """PostgreSQL 数据库连接配置."""
    
    port: int = Field(default=5432, description="数据库端口")
    ssl_mode: str = Field(default="prefer", description="SSL模式")
    application_name: str = Field(default="sql_analyzer", description="应用名称")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL


class TiDBConfig(DatabaseConfig):
    """TiDB 数据库连接配置."""
    
    port: int = Field(default=4000, description="数据库端口")
    charset: str = Field(default="utf8mb4", description="字符集")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.TIDB


class MariaDBConfig(DatabaseConfig):
    """MariaDB 数据库连接配置."""
    
    port: int = Field(default=3306, description="数据库端口")
    charset: str = Field(default="utf8mb4", description="字符集")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.MARIADB


class OracleConfig(DatabaseConfig):
    """Oracle 数据库连接配置."""
    
    port: int = Field(default=1521, description="数据库端口")
    service_name: str = Field(..., description="服务名称")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.ORACLE


class SQLServerConfig(DatabaseConfig):
    """SQL Server 数据库连接配置."""
    
    port: int = Field(default=1433, description="数据库端口")
    driver: str = Field(default="ODBC Driver 17 for SQL Server", description="ODBC驱动")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.SQLSERVER


class SQLiteConfig(DatabaseConfig):
    """SQLite 数据库连接配置."""
    
    database_path: str = Field(..., description="数据库文件路径")
    
    # Override fields that don't apply to SQLite
    host: str = Field(default="localhost", description="数据库主机地址")
    port: int = Field(default=0, description="数据库端口")
    user: str = Field(default="", description="数据库用户名")
    password: str = Field(default="", description="数据库密码")
    database: str = Field(default="", description="数据库名称")
    
    def get_database_type(self) -> DatabaseType:
        return DatabaseType.SQLITE


class SlowQueryConfig(BaseModel):
    """慢查询日志配置."""
    
    log_file_path: Optional[str] = Field(None, description="慢查询日志文件路径")
    use_performance_schema: bool = Field(default=True, description="是否使用 performance_schema")
    query_time_threshold: float = Field(default=1.0, description="查询时间阈值（秒）")
    rows_examined_threshold: int = Field(default=1000, description="扫描行数阈值")
    limit: int = Field(default=100, description="限制返回的慢查询数量")
    time_range_hours: int = Field(default=24, description="时间范围（小时）")


class SlowQueryEntry(BaseModel):
    """慢查询日志条目."""
    
    query_time: float = Field(..., description="查询执行时间（秒）")
    lock_time: float = Field(..., description="锁等待时间（秒）")
    rows_sent: int = Field(..., description="返回的行数")
    rows_examined: int = Field(..., description="检查的行数")
    sql_statement: str = Field(..., description="SQL 语句")
    timestamp: datetime = Field(..., description="查询时间戳")
    user: Optional[str] = Field(None, description="执行用户")
    host: Optional[str] = Field(None, description="客户端主机")
    database: Optional[str] = Field(None, description="数据库名称")


class ExplainResult(BaseModel):
    """EXPLAIN 查询结果的数据模型."""
    
    id: Optional[int] = Field(None, description="查询中 select 语句的 id")
    select_type: Optional[str] = Field(None, description="select 查询的类型")
    table: Optional[str] = Field(None, description="表名")
    partitions: Optional[str] = Field(None, description="分区信息")
    type: Optional[str] = Field(None, description="连接类型")
    possible_keys: Optional[str] = Field(None, description="可能使用的索引")
    key: Optional[str] = Field(None, description="实际使用的索引")
    key_len: Optional[str] = Field(None, description="索引长度")
    ref: Optional[str] = Field(None, description="与索引比较的列")
    rows: Optional[int] = Field(None, description="预估扫描的行数")
    filtered: Optional[float] = Field(None, description="按条件过滤的行的百分比")
    extra: Optional[str] = Field(None, description="额外的执行信息")
    
    # PostgreSQL 特有字段
    startup_cost: Optional[float] = Field(None, description="启动成本")
    total_cost: Optional[float] = Field(None, description="总成本")
    plan_width: Optional[int] = Field(None, description="计划宽度")
    actual_rows: Optional[int] = Field(None, description="实际行数")
    actual_time: Optional[float] = Field(None, description="实际时间")
    actual_loops: Optional[int] = Field(None, description="实际循环次数")


class ConnectionStatus(str, Enum):
    """连接状态枚举."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


class DatabaseHealthCheck(BaseModel):
    """数据库健康检查结果."""
    
    connection_id: str = Field(..., description="连接ID")
    database_type: DatabaseType = Field(..., description="数据库类型")
    host: str = Field(..., description="主机地址")
    port: int = Field(..., description="端口")
    database: str = Field(..., description="数据库名称")
    status: ConnectionStatus = Field(..., description="连接状态")
    response_time_ms: float = Field(..., description="响应时间（毫秒）")
    error_message: Optional[str] = Field(None, description="错误信息")
    last_check_time: datetime = Field(..., description="最后检查时间")
    consecutive_failures: int = Field(default=0, description="连续失败次数")


class LoadBalancingStrategy(str, Enum):
    """负载均衡策略枚举."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"


class DatabaseConnectionPool(BaseModel):
    """数据库连接池配置."""
    
    pool_id: str = Field(..., description="连接池ID")
    database_configs: List[DatabaseConfig] = Field(..., description="数据库配置列表")
    load_balancing_strategy: LoadBalancingStrategy = Field(default=LoadBalancingStrategy.ROUND_ROBIN, description="负载均衡策略")
    max_connections_per_db: int = Field(default=10, description="每个数据库的最大连接数")
    health_check_interval: int = Field(default=30, description="健康检查间隔（秒）")
    failover_enabled: bool = Field(default=True, description="是否启用故障转移")
    retry_attempts: int = Field(default=3, description="重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）")


class DatabaseCluster(BaseModel):
    """数据库集群配置."""
    
    cluster_id: str = Field(..., description="集群ID")
    cluster_name: str = Field(..., description="集群名称")
    primary_config: DatabaseConfig = Field(..., description="主数据库配置")
    replica_configs: List[DatabaseConfig] = Field(default_factory=list, description="从数据库配置列表")
    read_write_split: bool = Field(default=True, description="是否启用读写分离")
    auto_failover: bool = Field(default=True, description="是否启用自动故障转移")
    health_check_config: Dict[str, Any] = Field(default_factory=dict, description="健康检查配置") 