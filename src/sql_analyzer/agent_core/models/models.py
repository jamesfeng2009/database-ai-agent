"""Agent核心框架的数据模型."""

from datetime import datetime
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


class Event(BaseModel):
    """事件模型."""
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="事件ID")
    event_type: EventType = Field(..., description="事件类型")
    source: str = Field(..., description="事件源")
    data: Dict[str, Any] = Field(default_factory=dict, description="事件数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="事件时间戳")
    session_id: Optional[str] = Field(None, description="关联的会话ID")