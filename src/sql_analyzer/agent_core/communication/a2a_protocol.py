"""Agent-to-Agent (A2A) 通信协议实现."""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型枚举."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    ERROR = "error"


class Priority(str, Enum):
    """消息优先级枚举."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageStatus(str, Enum):
    """消息状态枚举."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSED = "processed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class A2AMessage(BaseModel):
    """A2A通信消息模型."""
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="消息ID")
    from_agent: str = Field(..., description="发送方Agent")
    to_agent: str = Field(..., description="接收方Agent")
    message_type: MessageType = Field(..., description="消息类型")
    action: str = Field(..., description="具体操作")
    payload: Dict[str, Any] = Field(default_factory=dict, description="消息载荷")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    priority: Priority = Field(default=Priority.MEDIUM, description="优先级")
    timeout: int = Field(default=30, description="超时时间(秒)")
    retry_count: int = Field(default=0, description="重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")
    status: MessageStatus = Field(default=MessageStatus.PENDING, description="消息状态")
    error_message: Optional[str] = Field(None, description="错误信息")


class AgentInfo(BaseModel):
    """Agent信息模型."""
    agent_id: str = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent名称")
    agent_type: str = Field(..., description="Agent类型")
    capabilities: List[str] = Field(default_factory=list, description="能力列表")
    endpoint: str = Field(..., description="通信端点")
    status: str = Field(default="active", description="状态")
    last_heartbeat: datetime = Field(default_factory=datetime.now, description="最后心跳时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class MessageHandler:
    """消息处理器基类."""
    
    def __init__(self, agent_id: str):
        """初始化消息处理器.
        
        Args:
            agent_id: Agent ID
        """
        self.agent_id = agent_id
        self._handlers: Dict[str, Callable] = {}
    
    def register_handler(self, action: str, handler: Callable):
        """注册消息处理器.
        
        Args:
            action: 操作名称
            handler: 处理函数
        """
        self._handlers[action] = handler
        logger.debug(f"注册消息处理器: {action} -> {handler.__name__}")
    
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """处理消息.
        
        Args:
            message: 接收到的消息
            
        Returns:
            响应消息（如果需要）
        """
        handler = self._handlers.get(message.action)
        if not handler:
            logger.warning(f"未找到处理器: {message.action}")
            return self._create_error_response(
                message, f"Unknown action: {message.action}"
            )
        
        try:
            logger.debug(f"处理消息: {message.action} from {message.from_agent}")
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(message)
            else:
                result = handler(message)
            
            # 如果是请求消息，需要返回响应
            if message.message_type == MessageType.REQUEST:
                return self._create_response(message, result)
            
            return None
            
        except Exception as e:
            logger.error(f"消息处理失败: {message.action}, 错误: {e}")
            return self._create_error_response(message, str(e))
    
    def _create_response(self, request: A2AMessage, result: Any) -> A2AMessage:
        """创建响应消息.
        
        Args:
            request: 原始请求消息
            result: 处理结果
            
        Returns:
            响应消息
        """
        return A2AMessage(
            from_agent=self.agent_id,
            to_agent=request.from_agent,
            message_type=MessageType.RESPONSE,
            action=f"{request.action}_response",
            payload={"result": result},
            correlation_id=request.message_id
        )
    
    def _create_error_response(self, request: A2AMessage, error: str) -> A2AMessage:
        """创建错误响应消息.
        
        Args:
            request: 原始请求消息
            error: 错误信息
            
        Returns:
            错误响应消息
        """
        return A2AMessage(
            from_agent=self.agent_id,
            to_agent=request.from_agent,
            message_type=MessageType.ERROR,
            action=f"{request.action}_error",
            payload={"error": error},
            correlation_id=request.message_id
        )


class MessageBus:
    """消息总线，负责Agent间的消息路由和传递."""
    
    def __init__(self):
        """初始化消息总线."""
        self._agents: Dict[str, AgentInfo] = {}
        self._message_handlers: Dict[str, MessageHandler] = {}
        self._pending_messages: Dict[str, A2AMessage] = {}
        self._message_history: List[A2AMessage] = []
        self._subscribers: Dict[str, List[str]] = {}  # action -> [agent_ids]
        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动消息总线."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        logger.info("消息总线已启动")
    
    async def stop(self):
        """停止消息总线."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("消息总线已停止")
    
    async def register_agent(self, agent_info: AgentInfo, handler: MessageHandler):
        """注册Agent.
        
        Args:
            agent_info: Agent信息
            handler: 消息处理器
        """
        async with self._lock:
            self._agents[agent_info.agent_id] = agent_info
            self._message_handlers[agent_info.agent_id] = handler
        
        logger.info(f"Agent已注册: {agent_info.agent_id} ({agent_info.agent_type})")
    
    async def unregister_agent(self, agent_id: str):
        """注销Agent.
        
        Args:
            agent_id: Agent ID
        """
        async with self._lock:
            self._agents.pop(agent_id, None)
            self._message_handlers.pop(agent_id, None)
        
        logger.info(f"Agent已注销: {agent_id}")
    
    async def send_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """发送消息.
        
        Args:
            message: 要发送的消息
            
        Returns:
            响应消息（如果是同步请求）
        """
        # 检查目标Agent是否存在
        if message.to_agent not in self._agents:
            logger.error(f"目标Agent不存在: {message.to_agent}")
            message.status = MessageStatus.FAILED
            message.error_message = f"Target agent not found: {message.to_agent}"
            return None
        
        # 记录消息
        message.status = MessageStatus.SENT
        await self._record_message(message)
        
        # 获取目标Agent的处理器
        handler = self._message_handlers.get(message.to_agent)
        if not handler:
            logger.error(f"目标Agent处理器不存在: {message.to_agent}")
            message.status = MessageStatus.FAILED
            return None
        
        try:
            # 处理消息
            response = await handler.handle_message(message)
            message.status = MessageStatus.PROCESSED
            
            if response:
                await self._record_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"消息发送失败: {e}")
            message.status = MessageStatus.FAILED
            message.error_message = str(e)
            return None
    
    async def send_request(
        self, 
        from_agent: str, 
        to_agent: str, 
        action: str, 
        payload: Dict[str, Any],
        timeout: int = 30
    ) -> Optional[A2AMessage]:
        """发送请求消息并等待响应.
        
        Args:
            from_agent: 发送方Agent
            to_agent: 接收方Agent
            action: 操作名称
            payload: 消息载荷
            timeout: 超时时间
            
        Returns:
            响应消息
        """
        message = A2AMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.REQUEST,
            action=action,
            payload=payload,
            timeout=timeout
        )
        
        # 记录待处理消息
        self._pending_messages[message.message_id] = message
        
        try:
            # 发送消息并等待响应
            response = await asyncio.wait_for(
                self.send_message(message),
                timeout=timeout
            )
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"消息超时: {message.message_id}")
            message.status = MessageStatus.TIMEOUT
            return None
        finally:
            # 清理待处理消息
            self._pending_messages.pop(message.message_id, None)
    
    async def send_notification(
        self, 
        from_agent: str, 
        to_agent: str, 
        action: str, 
        payload: Dict[str, Any]
    ):
        """发送通知消息（异步，不等待响应）.
        
        Args:
            from_agent: 发送方Agent
            to_agent: 接收方Agent
            action: 操作名称
            payload: 消息载荷
        """
        message = A2AMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.NOTIFICATION,
            action=action,
            payload=payload
        )
        
        # 异步发送，不等待响应
        asyncio.create_task(self.send_message(message))
    
    async def broadcast_message(
        self, 
        from_agent: str, 
        action: str, 
        payload: Dict[str, Any],
        exclude_agents: Optional[List[str]] = None
    ):
        """广播消息给所有Agent.
        
        Args:
            from_agent: 发送方Agent
            action: 操作名称
            payload: 消息载荷
            exclude_agents: 排除的Agent列表
        """
        exclude_agents = exclude_agents or []
        
        for agent_id in self._agents:
            if agent_id != from_agent and agent_id not in exclude_agents:
                message = A2AMessage(
                    from_agent=from_agent,
                    to_agent=agent_id,
                    message_type=MessageType.BROADCAST,
                    action=action,
                    payload=payload
                )
                
                # 异步发送广播消息
                asyncio.create_task(self.send_message(message))
    
    async def subscribe_action(self, agent_id: str, action: str):
        """订阅特定操作的消息.
        
        Args:
            agent_id: Agent ID
            action: 操作名称
        """
        if action not in self._subscribers:
            self._subscribers[action] = []
        
        if agent_id not in self._subscribers[action]:
            self._subscribers[action].append(agent_id)
            logger.debug(f"Agent {agent_id} 订阅操作: {action}")
    
    async def unsubscribe_action(self, agent_id: str, action: str):
        """取消订阅操作.
        
        Args:
            agent_id: Agent ID
            action: 操作名称
        """
        if action in self._subscribers:
            self._subscribers[action] = [
                aid for aid in self._subscribers[action] if aid != agent_id
            ]
            logger.debug(f"Agent {agent_id} 取消订阅操作: {action}")
    
    async def publish_event(
        self, 
        from_agent: str, 
        action: str, 
        payload: Dict[str, Any]
    ):
        """发布事件给订阅者.
        
        Args:
            from_agent: 发送方Agent
            action: 操作名称
            payload: 消息载荷
        """
        subscribers = self._subscribers.get(action, [])
        
        for agent_id in subscribers:
            if agent_id != from_agent:
                await self.send_notification(from_agent, agent_id, action, payload)
    
    def get_agents(self) -> List[AgentInfo]:
        """获取所有注册的Agent信息.
        
        Returns:
            Agent信息列表
        """
        return list(self._agents.values())
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """获取指定Agent信息.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent信息
        """
        return self._agents.get(agent_id)
    
    def get_message_history(self, limit: int = 100) -> List[A2AMessage]:
        """获取消息历史.
        
        Args:
            limit: 返回的消息数量限制
            
        Returns:
            消息历史列表
        """
        return self._message_history[-limit:]
    
    async def _record_message(self, message: A2AMessage):
        """记录消息到历史.
        
        Args:
            message: 消息对象
        """
        async with self._lock:
            self._message_history.append(message)
            # 限制历史记录大小
            if len(self._message_history) > 10000:
                self._message_history = self._message_history[-5000:]
    
    async def _cleanup_expired_messages(self):
        """清理过期消息."""
        while self._running:
            try:
                current_time = time.time()
                expired_messages = []
                
                for msg_id, message in self._pending_messages.items():
                    message_time = message.timestamp.timestamp()
                    if current_time - message_time > message.timeout:
                        expired_messages.append(msg_id)
                
                for msg_id in expired_messages:
                    message = self._pending_messages.pop(msg_id, None)
                    if message:
                        message.status = MessageStatus.TIMEOUT
                        logger.warning(f"消息超时: {msg_id}")
                
                await asyncio.sleep(10)  # 每10秒清理一次
                
            except Exception as e:
                logger.error(f"清理过期消息失败: {e}")
                await asyncio.sleep(10)


# 全局消息总线实例
_global_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """获取全局消息总线实例.
    
    Returns:
        全局消息总线实例
    """
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = MessageBus()
    return _global_message_bus


async def send_a2a_request(
    from_agent: str,
    to_agent: str,
    action: str,
    payload: Dict[str, Any],
    timeout: int = 30
) -> Optional[A2AMessage]:
    """发送A2A请求的便捷函数.
    
    Args:
        from_agent: 发送方Agent
        to_agent: 接收方Agent
        action: 操作名称
        payload: 消息载荷
        timeout: 超时时间
        
    Returns:
        响应消息
    """
    message_bus = get_message_bus()
    return await message_bus.send_request(from_agent, to_agent, action, payload, timeout)


async def send_a2a_notification(
    from_agent: str,
    to_agent: str,
    action: str,
    payload: Dict[str, Any]
):
    """发送A2A通知的便捷函数.
    
    Args:
        from_agent: 发送方Agent
        to_agent: 接收方Agent
        action: 操作名称
        payload: 消息载荷
    """
    message_bus = get_message_bus()
    await message_bus.send_notification(from_agent, to_agent, action, payload)