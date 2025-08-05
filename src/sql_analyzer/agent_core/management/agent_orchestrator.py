"""Agent编排器 - 基于A2A通信协议的高级Agent协调和工作流管理系统."""

import asyncio
import dataclasses
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# 直接在文件中定义必要的类型，避免复杂的导入依赖
import sys
import os

# 尝试相对导入，如果失败则使用本地定义
try:
    from ..communication.a2a_protocol import (
        A2AMessage, AgentInfo, MessageBus, MessageHandler, MessageType, Priority,
        get_message_bus
    )
    from ..models.models import Task, TaskStatus, TaskPriority, EventType
except ImportError:
    # 本地定义必要的枚举和类，避免复杂依赖
    from enum import Enum
    from pydantic import BaseModel, Field
    from uuid import uuid4
    from datetime import datetime
    from typing import Any, Dict, List, Optional
    
    class MessageType(str, Enum):
        REQUEST = "request"
        RESPONSE = "response"
        NOTIFICATION = "notification"
        BROADCAST = "broadcast"
        ERROR = "error"
    
    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class TaskStatus(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class TaskPriority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class EventType(str, Enum):
        TASK_CREATED = "task_created"
        TASK_STARTED = "task_started"
        TASK_COMPLETED = "task_completed"
        TASK_FAILED = "task_failed"
    
    class A2AMessage(BaseModel):
        message_id: str = Field(default_factory=lambda: str(uuid4()))
        from_agent: str
        to_agent: str
        message_type: MessageType
        action: str
        payload: Dict[str, Any] = Field(default_factory=dict)
        correlation_id: Optional[str] = None
        timestamp: datetime = Field(default_factory=datetime.now)
        priority: Priority = Field(default=Priority.MEDIUM)
        timeout: int = Field(default=30)
    
    class AgentInfo(BaseModel):
        agent_id: str
        agent_name: str
        agent_type: str
        capabilities: List[str] = Field(default_factory=list)
        endpoint: str
        status: str = Field(default="active")
        last_heartbeat: datetime = Field(default_factory=datetime.now)
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class MessageHandler:
        def __init__(self, agent_id: str):
            self.agent_id = agent_id
            self._handlers: Dict[str, callable] = {}
        
        def register_handler(self, action: str, handler: callable):
            self._handlers[action] = handler
        
        async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
            handler = self._handlers.get(message.action)
            if not handler:
                return None
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message)
                else:
                    result = handler(message)
                
                if message.message_type == MessageType.REQUEST:
                    return A2AMessage(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        message_type=MessageType.RESPONSE,
                        action=f"{message.action}_response",
                        payload={"result": result},
                        correlation_id=message.message_id
                    )
                return None
            except Exception as e:
                return A2AMessage(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    message_type=MessageType.ERROR,
                    action=f"{message.action}_error",
                    payload={"error": str(e)},
                    correlation_id=message.message_id
                )
    
    class MessageBus:
        def __init__(self):
            self._agents: Dict[str, AgentInfo] = {}
            self._message_handlers: Dict[str, MessageHandler] = {}
            self._running = False
        
        async def start(self):
            self._running = True
        
        async def stop(self):
            self._running = False
        
        async def register_agent(self, agent_info: AgentInfo, handler: MessageHandler):
            self._agents[agent_info.agent_id] = agent_info
            self._message_handlers[agent_info.agent_id] = handler
        
        async def unregister_agent(self, agent_id: str):
            self._agents.pop(agent_id, None)
            self._message_handlers.pop(agent_id, None)
        
        async def send_message(self, message: A2AMessage) -> Optional[A2AMessage]:
            handler = self._message_handlers.get(message.to_agent)
            if not handler:
                return None
            return await handler.handle_message(message)
        
        async def send_request(
            self, 
            from_agent: str, 
            to_agent: str, 
            action: str, 
            payload: Dict[str, Any],
            timeout: int = 30
        ) -> Optional[A2AMessage]:
            """发送请求消息并等待响应."""
            message = A2AMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=MessageType.REQUEST,
                action=action,
                payload=payload,
                timeout=timeout
            )
            return await self.send_message(message)
        
        def get_agents(self) -> List[AgentInfo]:
            return list(self._agents.values())
    
    _global_message_bus = None
    
    def get_message_bus() -> MessageBus:
        global _global_message_bus
        if _global_message_bus is None:
            _global_message_bus = MessageBus()
        return _global_message_bus
    
    class Task(BaseModel):
        task_id: str = Field(default_factory=lambda: str(uuid4()))
        task_type: str
        description: str
        parameters: Dict[str, Any] = Field(default_factory=dict)
        status: TaskStatus = Field(default=TaskStatus.PENDING)
        priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
        created_at: datetime = Field(default_factory=datetime.now)
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        dependencies: List[str] = Field(default_factory=list)
        session_id: str

logger = logging.getLogger(__name__)


class AgentHealth(str, Enum):
    """Agent健康状态枚举."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class WorkflowStatus(str, Enum):
    """工作流状态枚举."""
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class RecoveryAction(str, Enum):
    """恢复动作枚举."""
    RESTART_AGENT = "restart_agent"
    RETRY_TASK = "retry_task"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"


class WorkflowStep(BaseModel):
    """工作流步骤模型."""
    step_id: str = Field(default_factory=lambda: str(uuid4()), description="步骤ID")
    name: str = Field(..., description="步骤名称")
    agent_id: str = Field(..., description="执行Agent ID")
    action: str = Field(..., description="操作名称")
    payload: Dict[str, Any] = Field(default_factory=dict, description="操作参数")
    dependencies: List[str] = Field(default_factory=list, description="依赖的步骤ID列表")
    timeout: int = Field(default=60, description="超时时间(秒)")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="步骤状态")
    result: Optional[Dict[str, Any]] = Field(None, description="步骤结果")
    error: Optional[str] = Field(None, description="错误信息")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    execution_time: Optional[float] = Field(None, description="执行时间(秒)")


class Workflow(BaseModel):
    """工作流模型."""
    workflow_id: str = Field(default_factory=lambda: str(uuid4()), description="工作流ID")
    name: str = Field(..., description="工作流名称")
    description: str = Field(default="", description="工作流描述")
    workflow_type: str = Field(..., description="工作流类型")
    steps: Dict[str, WorkflowStep] = Field(default_factory=dict, description="工作流步骤")
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED, description="工作流状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    result: Optional[Dict[str, Any]] = Field(None, description="工作流结果")
    error: Optional[str] = Field(None, description="错误信息")
    session_id: Optional[str] = Field(None, description="关联会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def add_step(self, step: WorkflowStep):
        """添加工作流步骤."""
        self.steps[step.step_id] = step
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """获取可以执行的步骤（依赖已完成）."""
        ready_steps = []
        
        for step in self.steps.values():
            if step.status != TaskStatus.PENDING:
                continue
            
            # 检查依赖是否都已完成
            dependencies_completed = all(
                self.steps[dep_id].status == TaskStatus.COMPLETED
                for dep_id in step.dependencies
                if dep_id in self.steps
            )
            
            if dependencies_completed:
                ready_steps.append(step)
        
        return ready_steps
    
    def is_completed(self) -> bool:
        """检查工作流是否完成."""
        return all(
            step.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            for step in self.steps.values()
        )
    
    def has_failed_steps(self) -> bool:
        """检查是否有失败的步骤."""
        return any(step.status == TaskStatus.FAILED for step in self.steps.values())
    
    def get_completion_percentage(self) -> float:
        """获取完成百分比."""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(
            1 for step in self.steps.values()
            if step.status == TaskStatus.COMPLETED
        )
        
        return (completed_steps / len(self.steps)) * 100.0


@dataclasses.dataclass
class AgentRegistration:
    """Agent注册信息."""
    agent_info: AgentInfo
    handler: MessageHandler
    health_status: AgentHealth = AgentHealth.UNKNOWN
    last_heartbeat: datetime = dataclasses.field(default_factory=datetime.now)
    failure_count: int = 0
    recovery_attempts: int = 0
    performance_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)


class AgentOrchestrator:
    """Agent编排器 - 基于A2A通信协议的高级Agent协调和工作流管理系统."""
    
    def __init__(self):
        """初始化Agent编排器."""
        
        # 消息总线
        self._message_bus: Optional[MessageBus] = None
        
        # Agent管理
        self._registered_agents: Dict[str, AgentRegistration] = {}
        self._agent_capabilities: Dict[str, Set[str]] = {}  # capability -> agent_ids
        
        # 工作流管理
        self._workflows: Dict[str, Workflow] = {}
        self._active_workflows: Dict[str, asyncio.Task] = {}
        self._workflow_queue: asyncio.Queue = asyncio.Queue()
        
        # 任务管理
        self._tasks: Dict[str, Task] = {}
        self._task_workflows: Dict[str, str] = {}  # task_id -> workflow_id
        
        # 错误处理和恢复
        self._error_handlers: Dict[str, callable] = {}
        self._recovery_strategies: Dict[str, callable] = {}
        
        # 性能监控
        self._performance_metrics: Dict[str, Any] = {}
        self._health_check_interval = 30  # 秒
        
        # 并发控制
        self._max_concurrent_workflows = 10
        self._max_concurrent_tasks = 50
        self._workflow_semaphore = asyncio.Semaphore(self._max_concurrent_workflows)
        
        # 状态管理
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # 后台任务
        self._health_check_task: Optional[asyncio.Task] = None
        self._workflow_processor_task: Optional[asyncio.Task] = None
        self._metrics_collector_task: Optional[asyncio.Task] = None
        
        # 工作流模板
        self._workflow_templates: Dict[str, callable] = {}
        
        # 注册默认错误处理器和恢复策略
        self._register_default_handlers()
    
    async def start(self):
        """启动Agent编排器."""
        if self._running:
            logger.warning("Agent编排器已经在运行")
            return
        
        try:
            logger.info("启动Agent编排器...")
            
            # 获取消息总线
            self._message_bus = get_message_bus()
            if not self._message_bus._running:
                await self._message_bus.start()
            
            # 启动后台任务
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._workflow_processor_task = asyncio.create_task(self._workflow_processor_loop())
            self._metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())
            
            self._running = True
            logger.info("Agent编排器启动成功")
            
        except Exception as e:
            logger.error(f"Agent编排器启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止Agent编排器."""
        if not self._running:
            return
        
        logger.info("停止Agent编排器...")
        
        try:
            self._running = False
            self._shutdown_event.set()
            
            # 取消所有活跃的工作流
            for workflow_task in self._active_workflows.values():
                workflow_task.cancel()
            
            # 等待工作流完成
            if self._active_workflows:
                await asyncio.gather(*self._active_workflows.values(), return_exceptions=True)
            
            # 停止后台任务
            for task in [self._health_check_task, self._workflow_processor_task, self._metrics_collector_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            logger.info("Agent编排器已停止")
            
        except Exception as e:
            logger.error(f"停止Agent编排器时发生错误: {e}")
    
    # Agent注册和生命周期管理
    
    async def register_agent(self, agent_info: AgentInfo, handler: MessageHandler) -> bool:
        """注册Agent.
        
        Args:
            agent_info: Agent信息
            handler: 消息处理器
            
        Returns:
            是否成功注册
        """
        try:
            # 检查Agent是否已注册
            if agent_info.agent_id in self._registered_agents:
                logger.warning(f"Agent {agent_info.agent_id} 已经注册")
                return False
            
            # 创建注册信息
            registration = AgentRegistration(
                agent_info=agent_info,
                handler=handler,
                health_status=AgentHealth.HEALTHY
            )
            
            # 注册到消息总线
            await self._message_bus.register_agent(agent_info, handler)
            
            # 保存注册信息
            self._registered_agents[agent_info.agent_id] = registration
            
            # 更新能力映射
            for capability in agent_info.capabilities:
                if capability not in self._agent_capabilities:
                    self._agent_capabilities[capability] = set()
                self._agent_capabilities[capability].add(agent_info.agent_id)
            
            logger.info(f"Agent注册成功: {agent_info.agent_id} ({agent_info.agent_type})")
            return True
            
        except Exception as e:
            logger.error(f"Agent注册失败: {agent_info.agent_id}, 错误: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销Agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功注销
        """
        try:
            registration = self._registered_agents.get(agent_id)
            if not registration:
                logger.warning(f"Agent {agent_id} 未注册")
                return False
            
            # 从消息总线注销
            await self._message_bus.unregister_agent(agent_id)
            
            # 更新能力映射
            for capability in registration.agent_info.capabilities:
                if capability in self._agent_capabilities:
                    self._agent_capabilities[capability].discard(agent_id)
                    if not self._agent_capabilities[capability]:
                        del self._agent_capabilities[capability]
            
            # 移除注册信息
            del self._registered_agents[agent_id]
            
            logger.info(f"Agent注销成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent注销失败: {agent_id}, 错误: {e}")
            return False
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """根据能力获取Agent列表.
        
        Args:
            capability: 能力名称
            
        Returns:
            具有该能力的Agent ID列表
        """
        return list(self._agent_capabilities.get(capability, set()))
    
    def get_healthy_agents(self) -> List[str]:
        """获取健康的Agent列表.
        
        Returns:
            健康Agent ID列表
        """
        return [
            agent_id for agent_id, registration in self._registered_agents.items()
            if registration.health_status == AgentHealth.HEALTHY
        ]
    
    async def monitor_agent_health(self) -> Dict[str, AgentHealth]:
        """监控所有Agent的健康状态.
        
        Returns:
            Agent健康状态映射
        """
        health_status = {}
        
        for agent_id, registration in self._registered_agents.items():
            try:
                # 发送ping消息检查Agent健康状态
                response = await self._message_bus.send_request(
                    "orchestrator", agent_id, "ping", {}, timeout=5
                )
                
                if response and response.payload.get("result", {}).get("pong"):
                    registration.health_status = AgentHealth.HEALTHY
                    registration.last_heartbeat = datetime.now()
                    registration.failure_count = 0
                else:
                    registration.health_status = AgentHealth.UNHEALTHY
                    registration.failure_count += 1
                
            except Exception as e:
                logger.warning(f"Agent健康检查失败: {agent_id}, 错误: {e}")
                registration.health_status = AgentHealth.UNHEALTHY
                registration.failure_count += 1
            
            health_status[agent_id] = registration.health_status
        
        return health_status
    
    # 工作流执行引擎
    
    async def create_workflow(
        self,
        workflow_type: str,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """创建工作流.
        
        Args:
            workflow_type: 工作流类型
            name: 工作流名称
            description: 工作流描述
            parameters: 工作流参数
            priority: 优先级
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            工作流ID
        """
        try:
            # 检查工作流模板是否存在
            if workflow_type not in self._workflow_templates:
                raise ValueError(f"未知的工作流类型: {workflow_type}")
            
            # 创建工作流
            workflow = Workflow(
                name=name,
                description=description,
                workflow_type=workflow_type,
                priority=priority,
                session_id=session_id,
                user_id=user_id,
                metadata=parameters or {}
            )
            
            # 使用模板生成工作流步骤
            template_func = self._workflow_templates[workflow_type]
            await template_func(workflow, parameters or {})
            
            # 保存工作流
            self._workflows[workflow.workflow_id] = workflow
            
            # 添加到执行队列
            await self._workflow_queue.put(workflow.workflow_id)
            
            logger.info(f"工作流创建成功: {workflow.workflow_id} ({workflow_type})")
            return workflow.workflow_id
            
        except Exception as e:
            logger.error(f"创建工作流失败: {workflow_type}, 错误: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """执行工作流.
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            是否成功启动执行
        """
        try:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                logger.error(f"工作流不存在: {workflow_id}")
                return False
            
            if workflow.status != WorkflowStatus.CREATED:
                logger.warning(f"工作流状态不允许执行: {workflow_id}, 状态: {workflow.status}")
                return False
            
            # 启动工作流执行任务
            execution_task = asyncio.create_task(self._execute_workflow_async(workflow_id))
            self._active_workflows[workflow_id] = execution_task
            
            return True
            
        except Exception as e:
            logger.error(f"启动工作流执行失败: {workflow_id}, 错误: {e}")
            return False
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流.
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            是否成功取消
        """
        try:
            # 取消执行任务
            if workflow_id in self._active_workflows:
                self._active_workflows[workflow_id].cancel()
                del self._active_workflows[workflow_id]
            
            # 更新工作流状态
            workflow = self._workflows.get(workflow_id)
            if workflow:
                workflow.status = WorkflowStatus.CANCELLED
                workflow.completed_at = datetime.now()
            
            logger.info(f"工作流已取消: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消工作流失败: {workflow_id}, 错误: {e}")
            return False
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态.
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            工作流状态信息
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "workflow_type": workflow.workflow_type,
            "status": workflow.status,
            "priority": workflow.priority,
            "completion_percentage": workflow.get_completion_percentage(),
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "session_id": workflow.session_id,
            "user_id": workflow.user_id,
            "step_count": len(workflow.steps),
            "completed_steps": sum(1 for step in workflow.steps.values() if step.status == TaskStatus.COMPLETED),
            "failed_steps": sum(1 for step in workflow.steps.values() if step.status == TaskStatus.FAILED),
            "error": workflow.error,
            "result": workflow.result
        }
    
    def list_workflows(
        self,
        status_filter: Optional[WorkflowStatus] = None,
        workflow_type_filter: Optional[str] = None,
        user_id_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出工作流.
        
        Args:
            status_filter: 状态过滤器
            workflow_type_filter: 类型过滤器
            user_id_filter: 用户ID过滤器
            
        Returns:
            工作流列表
        """
        workflows = []
        
        for workflow in self._workflows.values():
            # 应用过滤器
            if status_filter and workflow.status != status_filter:
                continue
            if workflow_type_filter and workflow.workflow_type != workflow_type_filter:
                continue
            if user_id_filter and workflow.user_id != user_id_filter:
                continue
            
            workflows.append({
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "workflow_type": workflow.workflow_type,
                "status": workflow.status,
                "priority": workflow.priority,
                "completion_percentage": workflow.get_completion_percentage(),
                "created_at": workflow.created_at.isoformat(),
                "user_id": workflow.user_id,
                "session_id": workflow.session_id
            })
        
        return workflows
    
    # 消息路由和通信机制
    
    async def route_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """路由消息到合适的Agent.
        
        Args:
            message: 消息对象
            
        Returns:
            路由后的响应消息
        """
        try:
            # 检查目标Agent是否健康
            if message.to_agent in self._registered_agents:
                registration = self._registered_agents[message.to_agent]
                if registration.health_status != AgentHealth.HEALTHY:
                    # 尝试找到替代Agent
                    alternative_agent = await self._find_alternative_agent(message.to_agent, message.action)
                    if alternative_agent:
                        logger.info(f"路由消息到替代Agent: {message.to_agent} -> {alternative_agent}")
                        message.to_agent = alternative_agent
                    else:
                        logger.warning(f"目标Agent不健康且无替代Agent: {message.to_agent}")
                        return None
            
            # 发送消息
            response = await self._message_bus.send_message(message)
            
            # 更新性能指标
            if message.to_agent in self._registered_agents:
                registration = self._registered_agents[message.to_agent]
                if "message_count" not in registration.performance_metrics:
                    registration.performance_metrics["message_count"] = 0
                registration.performance_metrics["message_count"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"消息路由失败: {e}")
            return None
    
    async def broadcast_to_capability(
        self,
        capability: str,
        action: str,
        payload: Dict[str, Any],
        exclude_agents: Optional[List[str]] = None
    ) -> Dict[str, Optional[A2AMessage]]:
        """向具有特定能力的所有Agent广播消息.
        
        Args:
            capability: 能力名称
            action: 操作名称
            payload: 消息载荷
            exclude_agents: 排除的Agent列表
            
        Returns:
            Agent ID到响应消息的映射
        """
        responses = {}
        exclude_agents = exclude_agents or []
        
        agent_ids = self.get_agents_by_capability(capability)
        healthy_agents = [
            agent_id for agent_id in agent_ids
            if agent_id not in exclude_agents and
            self._registered_agents[agent_id].health_status == AgentHealth.HEALTHY
        ]
        
        # 并行发送消息
        tasks = []
        for agent_id in healthy_agents:
            message = A2AMessage(
                from_agent="orchestrator",
                to_agent=agent_id,
                message_type=MessageType.REQUEST,
                action=action,
                payload=payload
            )
            tasks.append(self.route_message(message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                agent_id = healthy_agents[i]
                if isinstance(result, Exception):
                    logger.error(f"广播消息失败: {agent_id}, 错误: {result}")
                    responses[agent_id] = None
                else:
                    responses[agent_id] = result
        
        return responses
    
    # 错误处理和恢复策略
    
    async def handle_agent_failure(self, agent_id: str, error: Exception) -> RecoveryAction:
        """处理Agent失败.
        
        Args:
            agent_id: 失败的Agent ID
            error: 错误信息
            
        Returns:
            恢复动作
        """
        try:
            registration = self._registered_agents.get(agent_id)
            if not registration:
                return RecoveryAction.IGNORE
            
            # 更新失败计数
            registration.failure_count += 1
            registration.health_status = AgentHealth.UNHEALTHY
            
            # 根据失败次数决定恢复策略
            if registration.failure_count <= 3:
                # 尝试重启Agent
                logger.info(f"尝试重启Agent: {agent_id}")
                return RecoveryAction.RESTART_AGENT
            elif registration.failure_count <= 5:
                # 尝试故障转移
                logger.info(f"尝试故障转移: {agent_id}")
                return RecoveryAction.FAILOVER
            else:
                # 需要人工干预
                logger.error(f"Agent多次失败，需要人工干预: {agent_id}")
                return RecoveryAction.MANUAL_INTERVENTION
            
        except Exception as e:
            logger.error(f"处理Agent失败时发生错误: {e}")
            return RecoveryAction.MANUAL_INTERVENTION
    
    def register_error_handler(self, error_type: str, handler: callable):
        """注册错误处理器.
        
        Args:
            error_type: 错误类型
            handler: 处理函数
        """
        self._error_handlers[error_type] = handler
        logger.debug(f"注册错误处理器: {error_type}")
    
    def register_recovery_strategy(self, failure_type: str, strategy: callable):
        """注册恢复策略.
        
        Args:
            failure_type: 失败类型
            strategy: 恢复策略函数
        """
        self._recovery_strategies[failure_type] = strategy
        logger.debug(f"注册恢复策略: {failure_type}")
    
    def register_workflow_template(self, workflow_type: str, template_func: callable):
        """注册工作流模板.
        
        Args:
            workflow_type: 工作流类型
            template_func: 模板函数
        """
        self._workflow_templates[workflow_type] = template_func
        logger.info(f"注册工作流模板: {workflow_type}")
    
    # 私有方法
    
    async def _execute_workflow_async(self, workflow_id: str):
        """异步执行工作流.
        
        Args:
            workflow_id: 工作流ID
        """
        async with self._workflow_semaphore:
            try:
                workflow = self._workflows[workflow_id]
                workflow.status = WorkflowStatus.RUNNING
                workflow.started_at = datetime.now()
                
                logger.info(f"开始执行工作流: {workflow_id} ({workflow.name})")
                
                while not workflow.is_completed():
                    # 获取可以执行的步骤
                    ready_steps = workflow.get_ready_steps()
                    
                    if not ready_steps:
                        # 没有可执行的步骤，检查是否有失败的步骤
                        if workflow.has_failed_steps():
                            workflow.status = WorkflowStatus.FAILED
                            workflow.error = "工作流包含失败的步骤"
                            break
                        else:
                            # 等待其他步骤完成
                            await asyncio.sleep(1)
                            continue
                    
                    # 并行执行所有可执行的步骤
                    step_tasks = []
                    for step in ready_steps:
                        step_task = asyncio.create_task(self._execute_workflow_step(step))
                        step_tasks.append(step_task)
                    
                    # 等待所有步骤完成
                    if step_tasks:
                        await asyncio.gather(*step_tasks, return_exceptions=True)
                
                # 聚合结果
                if workflow.status != WorkflowStatus.FAILED:
                    workflow.result = self._aggregate_workflow_results(workflow)
                    workflow.status = WorkflowStatus.COMPLETED
                
                workflow.completed_at = datetime.now()
                
                logger.info(f"工作流执行完成: {workflow_id}, 状态: {workflow.status}")
                
            except Exception as e:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = str(e)
                workflow.completed_at = datetime.now()
                logger.error(f"工作流执行失败: {workflow_id}, 错误: {e}")
            finally:
                # 清理活跃工作流记录
                if workflow_id in self._active_workflows:
                    del self._active_workflows[workflow_id]
    
    async def _execute_workflow_step(self, step: WorkflowStep):
        """执行工作流步骤.
        
        Args:
            step: 工作流步骤
        """
        try:
            step.status = TaskStatus.RUNNING
            step.started_at = datetime.now()
            
            logger.debug(f"执行工作流步骤: {step.step_id} -> {step.agent_id}.{step.action}")
            
            # 检查目标Agent是否健康
            if step.agent_id in self._registered_agents:
                registration = self._registered_agents[step.agent_id]
                if registration.health_status != AgentHealth.HEALTHY:
                    # 尝试找到替代Agent
                    alternative_agent = await self._find_alternative_agent(step.agent_id, step.action)
                    if alternative_agent:
                        logger.info(f"使用替代Agent执行步骤: {step.agent_id} -> {alternative_agent}")
                        step.agent_id = alternative_agent
                    else:
                        raise Exception(f"目标Agent不健康且无替代Agent: {step.agent_id}")
            
            # 创建消息
            message = A2AMessage(
                from_agent="orchestrator",
                to_agent=step.agent_id,
                message_type=MessageType.REQUEST,
                action=step.action,
                payload=step.payload,
                timeout=step.timeout
            )
            
            # 发送消息并等待响应
            response = await self.route_message(message)
            
            if response and response.payload.get("result"):
                step.result = response.payload["result"]
                step.status = TaskStatus.COMPLETED
            else:
                error_msg = response.payload.get("error", "未知错误") if response else "无响应"
                raise Exception(error_msg)
            
            step.completed_at = datetime.now()
            step.execution_time = (step.completed_at - step.started_at).total_seconds()
            
        except Exception as e:
            step.status = TaskStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            step.execution_time = (step.completed_at - step.started_at).total_seconds() if step.started_at else 0
            
            # 尝试重试
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = TaskStatus.PENDING
                logger.warning(f"工作流步骤失败，准备重试: {step.step_id}, 重试次数: {step.retry_count}")
                await asyncio.sleep(2 ** step.retry_count)  # 指数退避
                await self._execute_workflow_step(step)
            else:
                logger.error(f"工作流步骤执行失败: {step.step_id}, 错误: {e}")
    
    def _aggregate_workflow_results(self, workflow: Workflow) -> Dict[str, Any]:
        """聚合工作流结果.
        
        Args:
            workflow: 工作流对象
            
        Returns:
            聚合的结果
        """
        results = {}
        
        for step_id, step in workflow.steps.items():
            if step.result:
                results[step_id] = step.result
        
        execution_time = 0
        if workflow.completed_at and workflow.started_at:
            execution_time = (workflow.completed_at - workflow.started_at).total_seconds()
        
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "workflow_type": workflow.workflow_type,
            "execution_time": execution_time,
            "step_results": results,
            "completed_steps": sum(1 for step in workflow.steps.values() if step.status == TaskStatus.COMPLETED),
            "failed_steps": sum(1 for step in workflow.steps.values() if step.status == TaskStatus.FAILED),
            "total_steps": len(workflow.steps),
            "success_rate": (sum(1 for step in workflow.steps.values() if step.status == TaskStatus.COMPLETED) / len(workflow.steps)) * 100 if workflow.steps else 0
        }
    
    async def _find_alternative_agent(self, failed_agent_id: str, action: str) -> Optional[str]:
        """寻找替代Agent.
        
        Args:
            failed_agent_id: 失败的Agent ID
            action: 需要执行的操作
            
        Returns:
            替代Agent ID，如果没有则返回None
        """
        failed_registration = self._registered_agents.get(failed_agent_id)
        if not failed_registration:
            return None
        
        # 寻找具有相同能力的健康Agent
        for capability in failed_registration.agent_info.capabilities:
            candidate_agents = self.get_agents_by_capability(capability)
            for agent_id in candidate_agents:
                if (agent_id != failed_agent_id and
                    agent_id in self._registered_agents and
                    self._registered_agents[agent_id].health_status == AgentHealth.HEALTHY):
                    return agent_id
        
        return None
    
    async def _health_check_loop(self):
        """健康检查循环."""
        while self._running:
            try:
                await self.monitor_agent_health()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _workflow_processor_loop(self):
        """工作流处理循环."""
        while self._running:
            try:
                # 从队列获取工作流ID
                workflow_id = await asyncio.wait_for(
                    self._workflow_queue.get(),
                    timeout=1.0
                )
                
                # 执行工作流
                await self.execute_workflow(workflow_id)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作流处理失败: {e}")
    
    async def _metrics_collector_loop(self):
        """性能指标收集循环."""
        while self._running:
            try:
                # 收集系统性能指标
                self._performance_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "registered_agents": len(self._registered_agents),
                    "healthy_agents": len(self.get_healthy_agents()),
                    "active_workflows": len(self._active_workflows),
                    "total_workflows": len(self._workflows),
                    "completed_workflows": sum(1 for w in self._workflows.values() if w.status == WorkflowStatus.COMPLETED),
                    "failed_workflows": sum(1 for w in self._workflows.values() if w.status == WorkflowStatus.FAILED),
                    "workflow_queue_size": self._workflow_queue.qsize()
                }
                
                await asyncio.sleep(60)  # 每分钟收集一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能指标收集失败: {e}")
                await asyncio.sleep(60)
    
    def _register_default_handlers(self):
        """注册默认的错误处理器和恢复策略."""
        # 默认错误处理器
        self.register_error_handler("timeout", self._handle_timeout_error)
        self.register_error_handler("connection", self._handle_connection_error)
        self.register_error_handler("validation", self._handle_validation_error)
        
        # 默认恢复策略
        self.register_recovery_strategy("agent_failure", self._recover_from_agent_failure)
        self.register_recovery_strategy("workflow_failure", self._recover_from_workflow_failure)
    
    async def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]):
        """处理超时错误."""
        logger.warning(f"处理超时错误: {error}, 上下文: {context}")
        # 实现超时错误处理逻辑
    
    async def _handle_connection_error(self, error: Exception, context: Dict[str, Any]):
        """处理连接错误."""
        logger.warning(f"处理连接错误: {error}, 上下文: {context}")
        # 实现连接错误处理逻辑
    
    async def _handle_validation_error(self, error: Exception, context: Dict[str, Any]):
        """处理验证错误."""
        logger.warning(f"处理验证错误: {error}, 上下文: {context}")
        # 实现验证错误处理逻辑
    
    async def _recover_from_agent_failure(self, agent_id: str, failure_info: Dict[str, Any]):
        """从Agent失败中恢复."""
        logger.info(f"从Agent失败中恢复: {agent_id}")
        # 实现Agent失败恢复逻辑
    
    async def _recover_from_workflow_failure(self, workflow_id: str, failure_info: Dict[str, Any]):
        """从工作流失败中恢复."""
        logger.info(f"从工作流失败中恢复: {workflow_id}")
        # 实现工作流失败恢复逻辑
    
    # 公共接口方法
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标.
        
        Returns:
            性能指标字典
        """
        return self._performance_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态.
        
        Returns:
            系统状态信息
        """
        return {
            "running": self._running,
            "registered_agents": len(self._registered_agents),
            "healthy_agents": len(self.get_healthy_agents()),
            "active_workflows": len(self._active_workflows),
            "workflow_queue_size": self._workflow_queue.qsize(),
            "performance_metrics": self._performance_metrics
        }