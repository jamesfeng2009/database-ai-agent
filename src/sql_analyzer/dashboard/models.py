"""仪表板数据模型."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """指标类型枚举."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(str, Enum):
    """指标单位枚举."""
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    BYTES = "bytes"
    KILOBYTES = "kb"
    MEGABYTES = "mb"
    GIGABYTES = "gb"
    PERCENT = "%"
    COUNT = "count"
    RATE = "rate"


class ComponentType(str, Enum):
    """组件类型枚举."""
    CHART = "chart"
    TABLE = "table"
    GAUGE = "gauge"
    ALERT = "alert"
    TEXT = "text"
    COMPARISON = "comparison"


class ChartType(str, Enum):
    """图表类型枚举."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


class AlertLevel(str, Enum):
    """告警级别枚举."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricDefinition(BaseModel):
    """指标定义."""
    
    metric_id: str = Field(..., description="指标ID")
    name: str = Field(..., description="指标名称")
    description: str = Field(..., description="指标描述")
    metric_type: MetricType = Field(..., description="指标类型")
    unit: MetricUnit = Field(..., description="指标单位")
    database_types: List[str] = Field(default_factory=list, description="适用的数据库类型")
    collection_interval: int = Field(default=60, description="收集间隔（秒）")
    retention_days: int = Field(default=30, description="数据保留天数")
    tags: Dict[str, str] = Field(default_factory=dict, description="标签")


class PerformanceMetrics(BaseModel):
    """性能指标数据."""
    
    database_id: str = Field(..., description="数据库ID")
    database_type: str = Field(..., description="数据库类型")
    timestamp: datetime = Field(..., description="时间戳")
    metrics: Dict[str, Union[float, int, str]] = Field(..., description="指标数据")
    
    # 常用性能指标
    cpu_usage: Optional[float] = Field(None, description="CPU使用率(%)")
    memory_usage: Optional[float] = Field(None, description="内存使用率(%)")
    disk_usage: Optional[float] = Field(None, description="磁盘使用率(%)")
    connections_active: Optional[int] = Field(None, description="活跃连接数")
    connections_total: Optional[int] = Field(None, description="总连接数")
    queries_per_second: Optional[float] = Field(None, description="每秒查询数")
    slow_queries: Optional[int] = Field(None, description="慢查询数量")
    response_time_avg: Optional[float] = Field(None, description="平均响应时间(ms)")
    response_time_p95: Optional[float] = Field(None, description="95%响应时间(ms)")
    response_time_p99: Optional[float] = Field(None, description="99%响应时间(ms)")
    lock_waits: Optional[int] = Field(None, description="锁等待数")
    deadlocks: Optional[int] = Field(None, description="死锁数")
    buffer_hit_ratio: Optional[float] = Field(None, description="缓冲区命中率(%)")
    index_hit_ratio: Optional[float] = Field(None, description="索引命中率(%)")


class ComponentConfig(BaseModel):
    """组件配置."""
    
    component_id: str = Field(..., description="组件ID")
    component_type: ComponentType = Field(..., description="组件类型")
    title: str = Field(..., description="组件标题")
    description: Optional[str] = Field(None, description="组件描述")
    position: Dict[str, int] = Field(..., description="位置信息 {x, y, width, height}")
    
    # 图表特定配置
    chart_type: Optional[ChartType] = Field(None, description="图表类型")
    metrics: List[str] = Field(default_factory=list, description="显示的指标")
    time_range: int = Field(default=3600, description="时间范围（秒）")
    refresh_interval: int = Field(default=30, description="刷新间隔（秒）")
    
    # 过滤和分组
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    group_by: List[str] = Field(default_factory=list, description="分组字段")
    
    # 样式配置
    style: Dict[str, Any] = Field(default_factory=dict, description="样式配置")
    
    # 告警配置
    alert_rules: List[Dict[str, Any]] = Field(default_factory=list, description="告警规则")


class DashboardConfig(BaseModel):
    """仪表板配置."""
    
    dashboard_id: str = Field(..., description="仪表板ID")
    name: str = Field(..., description="仪表板名称")
    description: Optional[str] = Field(None, description="仪表板描述")
    owner: str = Field(..., description="所有者")
    is_public: bool = Field(default=False, description="是否公开")
    
    # 布局配置
    layout: Dict[str, Any] = Field(default_factory=dict, description="布局配置")
    components: List[ComponentConfig] = Field(default_factory=list, description="组件列表")
    
    # 数据源配置
    data_sources: List[str] = Field(default_factory=list, description="数据源列表")
    default_time_range: int = Field(default=3600, description="默认时间范围（秒）")
    auto_refresh: bool = Field(default=True, description="是否自动刷新")
    refresh_interval: int = Field(default=30, description="刷新间隔（秒）")
    
    # 权限配置
    permissions: Dict[str, List[str]] = Field(default_factory=dict, description="权限配置")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    version: int = Field(default=1, description="版本号")
    tags: List[str] = Field(default_factory=list, description="标签")


class AlertRule(BaseModel):
    """告警规则."""
    
    rule_id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    metric: str = Field(..., description="监控指标")
    condition: str = Field(..., description="告警条件")
    threshold: float = Field(..., description="阈值")
    level: AlertLevel = Field(..., description="告警级别")
    duration: int = Field(default=300, description="持续时间（秒）")
    enabled: bool = Field(default=True, description="是否启用")
    
    # 通知配置
    notification_channels: List[str] = Field(default_factory=list, description="通知渠道")
    notification_template: Optional[str] = Field(None, description="通知模板")
    
    # 抑制配置
    suppression_rules: List[Dict[str, Any]] = Field(default_factory=list, description="抑制规则")
    
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class Alert(BaseModel):
    """告警实例."""
    
    alert_id: str = Field(..., description="告警ID")
    rule_id: str = Field(..., description="规则ID")
    database_id: str = Field(..., description="数据库ID")
    metric: str = Field(..., description="指标名称")
    current_value: float = Field(..., description="当前值")
    threshold: float = Field(..., description="阈值")
    level: AlertLevel = Field(..., description="告警级别")
    message: str = Field(..., description="告警消息")
    
    # 状态信息
    status: str = Field(default="active", description="告警状态")
    first_seen: datetime = Field(default_factory=datetime.now, description="首次发现时间")
    last_seen: datetime = Field(default_factory=datetime.now, description="最后发现时间")
    resolved_at: Optional[datetime] = Field(None, description="解决时间")
    
    # 上下文信息
    labels: Dict[str, str] = Field(default_factory=dict, description="标签")
    annotations: Dict[str, str] = Field(default_factory=dict, description="注释")


class DatabaseComparison(BaseModel):
    """数据库对比数据."""
    
    comparison_id: str = Field(..., description="对比ID")
    name: str = Field(..., description="对比名称")
    database_ids: List[str] = Field(..., description="对比的数据库ID列表")
    metrics: List[str] = Field(..., description="对比的指标列表")
    time_range: int = Field(default=3600, description="时间范围（秒）")
    
    # 对比结果
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="对比结果")
    summary: Dict[str, Any] = Field(default_factory=dict, description="对比摘要")
    
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class DashboardSnapshot(BaseModel):
    """仪表板快照."""
    
    snapshot_id: str = Field(..., description="快照ID")
    dashboard_id: str = Field(..., description="仪表板ID")
    name: str = Field(..., description="快照名称")
    description: Optional[str] = Field(None, description="快照描述")
    
    # 快照数据
    config: DashboardConfig = Field(..., description="仪表板配置")
    data: Dict[str, Any] = Field(..., description="快照数据")
    
    # 元数据
    created_by: str = Field(..., description="创建者")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    is_public: bool = Field(default=False, description="是否公开")


class MetricsQuery(BaseModel):
    """指标查询."""
    
    metrics: List[str] = Field(..., description="查询的指标列表")
    database_ids: Optional[List[str]] = Field(None, description="数据库ID列表")
    database_types: Optional[List[str]] = Field(None, description="数据库类型列表")
    time_range: int = Field(default=3600, description="时间范围（秒）")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    
    # 聚合配置
    aggregation: Optional[str] = Field(None, description="聚合方式")
    group_by: List[str] = Field(default_factory=list, description="分组字段")
    
    # 过滤条件
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")
    
    # 排序和限制
    order_by: Optional[str] = Field(None, description="排序字段")
    limit: Optional[int] = Field(None, description="结果限制")


class MetricsQueryResult(BaseModel):
    """指标查询结果."""
    
    query: MetricsQuery = Field(..., description="查询条件")
    data: List[Dict[str, Any]] = Field(..., description="查询结果数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    # 统计信息
    total_records: int = Field(..., description="总记录数")
    execution_time_ms: float = Field(..., description="执行时间（毫秒）")
    
    created_at: datetime = Field(default_factory=datetime.now, description="查询时间")