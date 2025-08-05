"""SQL 分析相关的数据模型."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExplainResult(BaseModel):
    """EXPLAIN 查询结果的数据模型，兼容MySQL和PostgreSQL."""
    
    # 通用字段
    id: Optional[int] = Field(None, description="查询中 select 语句的 id")
    table: Optional[str] = Field(None, description="表名")
    extra: Optional[str] = Field(None, description="额外的执行信息")
    
    # MySQL特有字段
    select_type: Optional[str] = Field(None, description="select 查询的类型")
    partitions: Optional[str] = Field(None, description="分区信息")
    type: Optional[str] = Field(None, description="连接类型")
    possible_keys: Optional[str] = Field(None, description="可能使用的索引")
    key: Optional[str] = Field(None, description="实际使用的索引")
    key_len: Optional[str] = Field(None, description="索引长度")
    ref: Optional[str] = Field(None, description="与索引比较的列")
    rows: Optional[int] = Field(None, description="预估扫描的行数")
    filtered: Optional[float] = Field(None, description="按条件过滤的行的百分比")
    
    # PostgreSQL特有字段
    plan_rows: Optional[int] = Field(None, description="计划扫描的行数")
    actual_rows: Optional[int] = Field(None, description="实际扫描的行数")
    startup_cost: Optional[float] = Field(None, description="启动成本")
    total_cost: Optional[float] = Field(None, description="总成本")
    actual_time: Optional[float] = Field(None, description="实际执行时间")
    actual_loops: Optional[int] = Field(None, description="实际循环次数")


class SQLAnalysisRequest(BaseModel):
    """SQL 分析请求模型."""
    
    sql_statement: str = Field(..., description="要分析的 SQL 语句")
    explain_results: List[ExplainResult] = Field(..., description="EXPLAIN 分析结果")
    database_schema: Optional[str] = Field(None, description="数据库名称")
    table_schemas: Optional[Dict[str, Any]] = Field(None, description="相关表的结构信息")
    indexes: Optional[Dict[str, List[str]]] = Field(None, description="表的索引信息")
    additional_context: Optional[str] = Field(None, description="额外的上下文信息")


class PerformanceIssue(BaseModel):
    """性能问题描述模型."""
    
    severity: str = Field(..., description="问题严重程度: critical, high, medium, low")
    issue_type: str = Field(..., description="问题类型")
    description: str = Field(..., description="问题描述")
    impact: str = Field(..., description="对性能的影响")
    affected_tables: List[str] = Field(default_factory=list, description="受影响的表")


class OptimizationSuggestion(BaseModel):
    """优化建议模型."""
    
    priority: str = Field(..., description="优化优先级: high, medium, low")
    category: str = Field(..., description="优化类别: index, query_rewrite, schema, etc.")
    suggestion: str = Field(..., description="具体的优化建议")
    expected_improvement: str = Field(..., description="预期的性能改善")
    implementation_difficulty: str = Field(..., description="实施难度: easy, medium, hard")
    sql_example: Optional[str] = Field(None, description="优化后的 SQL 示例")


class SQLAnalysisResponse(BaseModel):
    """SQL 分析响应模型."""
    
    summary: str = Field(..., description="分析结果总结")
    performance_score: int = Field(..., description="性能评分 (0-100)")
    issues: List[PerformanceIssue] = Field(default_factory=list, description="发现的性能问题")
    suggestions: List[OptimizationSuggestion] = Field(default_factory=list, description="优化建议")
    detailed_analysis: str = Field(..., description="详细的分析报告")
    execution_plan_analysis: str = Field(..., description="执行计划分析")
    # 新增 EXPLAIN 结果字段
    explain_results: List[ExplainResult] = Field(
        default_factory=list,
        description="EXPLAIN 原始结果数据"
    )



class MySQLConfig(BaseModel):
    """MySQL 数据库连接配置."""
    
    host: str = Field(..., description="数据库主机地址")
    port: int = Field(default=3306, description="数据库端口")
    user: str = Field(..., description="数据库用户名")
    password: str = Field(..., description="数据库密码")
    database: str = Field(..., description="数据库名称")
    charset: str = Field(default="utf8mb4", description="字符集")
    connect_timeout: int = Field(default=10, description="连接超时时间（秒）")
    max_connections: int = Field(default=10, description="最大连接数")


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


class SlowQueryAnalysisResult(BaseModel):
    """慢查询分析结果."""
    
    slow_query: SlowQueryEntry = Field(..., description="慢查询信息")
    analysis_response: SQLAnalysisResponse = Field(..., description="分析结果")
    explain_results: List[ExplainResult] = Field(..., description="EXPLAIN 结果")


class BatchAnalysisResult(BaseModel):
    """批量分析结果."""
    
    total_queries: int = Field(..., description="总查询数量")
    analyzed_queries: int = Field(..., description="已分析查询数量")
    failed_queries: int = Field(..., description="分析失败查询数量")
    results: List[SlowQueryAnalysisResult] = Field(..., description="分析结果列表")
    summary_stats: Dict[str, Any] = Field(default_factory=dict, description="统计摘要")
    start_time: datetime = Field(..., description="分析开始时间")
    end_time: datetime = Field(..., description="分析结束时间") 