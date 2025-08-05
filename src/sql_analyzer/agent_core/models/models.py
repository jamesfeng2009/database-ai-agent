"""Agent核心框架的数据模型."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """消息角色枚举."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """消息模型."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="消息ID")
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")


class IntentType(str, Enum):
    """用户意图类型枚举."""
    QUERY_ANALYSIS = "query_analysis"
    OPTIMIZATION_REQUEST = "optimization_request"
    MONITORING_SETUP = "monitoring_setup"
    KNOWLEDGE_QUERY = "knowledge_query"
    HELP_REQUEST = "help_request"
    UNKNOWN = "unknown"


class UserIntent(BaseModel):
    """用户意图模型."""
    intent_type: IntentType = Field(..., description="意图类型")
    entities: Dict[str, Any] = Field(default_factory=dict, description="提取的实体")
    confidence: float = Field(..., description="置信度")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="意图参数")
    raw_input: str = Field(..., description="原始用户输入")


class SessionState(str, Enum):
    """会话状态枚举."""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ConversationContext(BaseModel):
    """对话上下文模型."""
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    current_database: Optional[str] = Field(None, description="当前数据库")
    conversation_history: List[Message] = Field(default_factory=list, description="对话历史")
    active_tasks: List[str] = Field(default_factory=list, description="活跃任务ID列表")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好设置")
    context_variables: Dict[str, Any] = Field(default_factory=dict, description="上下文变量")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    last_activity: datetime = Field(default_factory=datetime.now, description="最后活动时间")
    state: SessionState = Field(default=SessionState.ACTIVE, description="会话状态")


class AgentResponse(BaseModel):
    """Agent响应模型."""
    response_id: str = Field(default_factory=lambda: str(uuid4()), description="响应ID")
    content: str = Field(..., description="响应内容")
    intent_handled: IntentType = Field(..., description="处理的意图类型")
    suggested_actions: List[str] = Field(default_factory=list, description="建议的后续操作")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    requires_followup: bool = Field(default=False, description="是否需要后续交互")


class TaskStatus(str, Enum):
    """任务状态枚举."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """任务优先级枚举."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """任务模型."""
    task_id: str = Field(default_factory=lambda: str(uuid4()), description="任务ID")
    task_type: str = Field(..., description="任务类型")
    description: str = Field(..., description="任务描述")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="任务优先级")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息")
    dependencies: List[str] = Field(default_factory=list, description="依赖的任务ID列表")
    session_id: str = Field(..., description="关联的会话ID")


class EventType(str, Enum):
    """事件类型枚举."""
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    CONTEXT_UPDATED = "context_updated"
    ERROR_OCCURRED = "error_occurred"
    
    # Agent生命周期事件
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    AGENT_HEALTH_CHECK = "agent_health_check"
    
    # 数据库操作事件
    DATABASE_CONNECTED = "database_connected"
    DATABASE_DISCONNECTED = "database_disconnected"
    QUERY_EXECUTED = "query_executed"
    SLOW_QUERY_DETECTED = "slow_query_detected"
    PERFORMANCE_ALERT = "performance_alert"
    
    # 跨数据库事件
    CROSS_DB_QUERY_STARTED = "cross_db_query_started"
    CROSS_DB_QUERY_COMPLETED = "cross_db_query_completed"
    CROSS_DB_DEPENDENCY_DETECTED = "cross_db_dependency_detected"
    CROSS_DB_PERFORMANCE_ISSUE = "cross_db_performance_issue"
    
    # 学习和优化事件
    PATTERN_DISCOVERED = "pattern_discovered"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    OPTIMIZATION_APPLIED = "optimization_applied"
    FEEDBACK_RECEIVED = "feedback_received"
    
    # 系统监控事件
    RESOURCE_THRESHOLD_EXCEEDED = "resource_threshold_exceeded"
    SYSTEM_HEALTH_DEGRADED = "system_health_degraded"
    BACKUP_COMPLETED = "backup_completed"
    MAINTENANCE_SCHEDULED = "maintenance_scheduled"


class EventPriority(str, Enum):
    """事件优先级枚举."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventFilter(BaseModel):
    """事件过滤器模型."""
    filter_id: str = Field(default_factory=lambda: str(uuid4()), description="过滤器ID")
    name: str = Field(..., description="过滤器名称")
    event_types: Optional[List[EventType]] = Field(None, description="允许的事件类型")
    source_patterns: Optional[List[str]] = Field(None, description="源模式匹配")
    data_filters: Optional[Dict[str, Any]] = Field(None, description="数据字段过滤")
    priority_threshold: Optional[EventPriority] = Field(None, description="优先级阈值")
    enabled: bool = Field(default=True, description="是否启用")


class EventRoute(BaseModel):
    """事件路由模型."""
    route_id: str = Field(default_factory=lambda: str(uuid4()), description="路由ID")
    name: str = Field(..., description="路由名称")
    filter: EventFilter = Field(..., description="路由过滤器")
    target_handlers: List[str] = Field(..., description="目标处理器列表")
    enabled: bool = Field(default=True, description="是否启用")


class Event(BaseModel):
    """事件模型."""
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="事件ID")
    event_type: EventType = Field(..., description="事件类型")
    source: str = Field(..., description="事件源")
    data: Dict[str, Any] = Field(default_factory=dict, description="事件数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="事件时间戳")
    session_id: Optional[str] = Field(None, description="关联的会话ID")
    priority: EventPriority = Field(default=EventPriority.MEDIUM, description="事件优先级")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    tags: List[str] = Field(default_factory=list, description="事件标签")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="事件元数据")


class EventSubscription(BaseModel):
    """事件订阅模型."""
    subscription_id: str = Field(default_factory=lambda: str(uuid4()), description="订阅ID")
    event_type: EventType = Field(..., description="订阅的事件类型")
    handler_id: str = Field(..., description="处理器ID")
    filter: Optional[EventFilter] = Field(None, description="订阅过滤器")
    active: bool = Field(default=True, description="是否活跃")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    last_triggered: Optional[datetime] = Field(None, description="最后触发时间")


class EventBatch(BaseModel):
    """事件批次模型."""
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="批次ID")
    events: List[Event] = Field(..., description="批次中的事件")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    processed_at: Optional[datetime] = Field(None, description="处理时间")
    status: str = Field(default="pending", description="批次状态")


class EventPattern(BaseModel):
    """事件模式模型."""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()), description="模式ID")
    name: str = Field(..., description="模式名称")
    description: str = Field(..., description="模式描述")
    event_sequence: List[EventType] = Field(..., description="事件序列")
    time_window: timedelta = Field(..., description="时间窗口")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="匹配条件")
    enabled: bool = Field(default=True, description="是否启用")


class EventAggregation(BaseModel):
    """事件聚合模型."""
    aggregation_id: str = Field(default_factory=lambda: str(uuid4()), description="聚合ID")
    name: str = Field(..., description="聚合名称")
    event_types: List[EventType] = Field(..., description="聚合的事件类型")
    aggregation_function: str = Field(..., description="聚合函数")
    time_window: timedelta = Field(..., description="时间窗口")
    group_by: List[str] = Field(default_factory=list, description="分组字段")
    result: Optional[Dict[str, Any]] = Field(None, description="聚合结果")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class EventCorrelation(BaseModel):
    """事件关联模型."""
    correlation_id: str = Field(default_factory=lambda: str(uuid4()), description="关联ID")
    name: str = Field(..., description="关联名称")
    primary_event_type: EventType = Field(..., description="主事件类型")
    related_event_types: List[EventType] = Field(..., description="相关事件类型")
    correlation_key: str = Field(..., description="关联键")
    time_window: timedelta = Field(..., description="关联时间窗口")
    strength: float = Field(default=0.0, description="关联强度")


class EventWindow(BaseModel):
    """事件窗口模型."""
    window_id: str = Field(default_factory=lambda: str(uuid4()), description="窗口ID")
    window_type: str = Field(..., description="窗口类型")  # sliding, tumbling, session
    size: timedelta = Field(..., description="窗口大小")
    slide: Optional[timedelta] = Field(None, description="滑动间隔")
    events: List[Event] = Field(default_factory=list, description="窗口中的事件")
    start_time: datetime = Field(default_factory=datetime.now, description="窗口开始时间")
    end_time: Optional[datetime] = Field(None, description="窗口结束时间")
    is_active: bool = Field(default=True, description="窗口是否活跃")


class EventTrace(BaseModel):
    """事件链路追踪模型."""
    trace_id: str = Field(default_factory=lambda: str(uuid4()), description="追踪ID")
    root_event_id: str = Field(..., description="根事件ID")
    events: List[Event] = Field(default_factory=list, description="追踪的事件链")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="事件依赖关系")
    start_time: datetime = Field(default_factory=datetime.now, description="追踪开始时间")
    end_time: Optional[datetime] = Field(None, description="追踪结束时间")
    status: str = Field(default="active", description="追踪状态")


class EventCondition(BaseModel):
    """事件条件模型."""
    condition_id: str = Field(default_factory=lambda: str(uuid4()), description="条件ID")
    field: str = Field(..., description="条件字段")
    operator: str = Field(..., description="操作符")  # eq, ne, gt, lt, gte, lte, contains, matches, in, not_in
    value: Any = Field(..., description="条件值")
    logical_operator: str = Field(default="AND", description="逻辑操作符")  # AND, OR, NOT


class EventAction(BaseModel):
    """事件动作模型."""
    action_id: str = Field(default_factory=lambda: str(uuid4()), description="动作ID")
    action_type: str = Field(..., description="动作类型")  # notify, execute, trigger, log, webhook
    target: str = Field(..., description="动作目标")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="动作参数")
    enabled: bool = Field(default=True, description="是否启用")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")


class EventRule(BaseModel):
    """事件规则模型."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()), description="规则ID")
    name: str = Field(..., description="规则名称")
    description: str = Field(..., description="规则描述")
    trigger_events: List[EventType] = Field(..., description="触发事件类型")
    conditions: List[EventCondition] = Field(default_factory=list, description="触发条件")
    actions: List[EventAction] = Field(..., description="执行动作")
    priority: int = Field(default=100, description="规则优先级")
    enabled: bool = Field(default=True, description="是否启用")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    last_triggered: Optional[datetime] = Field(None, description="最后触发时间")
    trigger_count: int = Field(default=0, description="触发次数")
    cooldown_period: Optional[timedelta] = Field(None, description="冷却期")


class RuleExecution(BaseModel):
    """规则执行记录模型."""
    execution_id: str = Field(default_factory=lambda: str(uuid4()), description="执行ID")
    rule_id: str = Field(..., description="规则ID")
    trigger_event: Event = Field(..., description="触发事件")
    conditions_met: List[str] = Field(default_factory=list, description="满足的条件ID")
    actions_executed: List[str] = Field(default_factory=list, description="执行的动作ID")
    execution_time: datetime = Field(default_factory=datetime.now, description="执行时间")
    success: bool = Field(default=True, description="执行是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    execution_duration: Optional[float] = Field(None, description="执行耗时（秒）")


class StreamWindow(BaseModel):
    """流窗口模型."""
    window_id: str = Field(default_factory=lambda: str(uuid4()), description="窗口ID")
    window_type: str = Field(..., description="窗口类型")  # sliding, tumbling, session
    size: timedelta = Field(..., description="窗口大小")
    slide: Optional[timedelta] = Field(None, description="滑动间隔")
    session_timeout: Optional[timedelta] = Field(None, description="会话超时")
    events: List[Event] = Field(default_factory=list, description="窗口中的事件")
    start_time: datetime = Field(default_factory=datetime.now, description="窗口开始时间")
    end_time: Optional[datetime] = Field(None, description="窗口结束时间")
    is_active: bool = Field(default=True, description="窗口是否活跃")
    watermark: Optional[datetime] = Field(None, description="水位线时间")


class StreamTransformation(BaseModel):
    """流转换模型."""
    transformation_id: str = Field(default_factory=lambda: str(uuid4()), description="转换ID")
    name: str = Field(..., description="转换名称")
    transformation_type: str = Field(..., description="转换类型")  # filter, map, flatmap, reduce
    parameters: Dict[str, Any] = Field(default_factory=dict, description="转换参数")
    enabled: bool = Field(default=True, description="是否启用")


class StreamAggregator(BaseModel):
    """流聚合器模型."""
    aggregator_id: str = Field(default_factory=lambda: str(uuid4()), description="聚合器ID")
    name: str = Field(..., description="聚合器名称")
    aggregation_function: str = Field(..., description="聚合函数")  # sum, count, avg, max, min, collect
    key_selector: Optional[str] = Field(None, description="分组键选择器")
    window_config: Optional[StreamWindow] = Field(None, description="窗口配置")
    enabled: bool = Field(default=True, description="是否启用")


class BackpressureConfig(BaseModel):
    """背压配置模型."""
    max_buffer_size: int = Field(default=10000, description="最大缓冲区大小")
    high_watermark: float = Field(default=0.8, description="高水位线")
    low_watermark: float = Field(default=0.2, description="低水位线")
    backpressure_strategy: str = Field(default="drop_oldest", description="背压策略")  # drop_oldest, drop_newest, block
    throttle_rate: Optional[float] = Field(None, description="限流速率（事件/秒）")


class StreamMetrics(BaseModel):
    """流处理指标模型."""
    stream_id: str = Field(..., description="流ID")
    events_processed: int = Field(default=0, description="已处理事件数")
    events_dropped: int = Field(default=0, description="丢弃事件数")
    processing_rate: float = Field(default=0.0, description="处理速率（事件/秒）")
    buffer_utilization: float = Field(default=0.0, description="缓冲区利用率")
    average_latency: float = Field(default=0.0, description="平均延迟（毫秒）")
    error_count: int = Field(default=0, description="错误计数")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")