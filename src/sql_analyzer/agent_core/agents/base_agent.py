"""基础Agent抽象类，定义所有Agent的通用接口和行为."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from ..communication.a2a_protocol import (
    A2AMessage, AgentInfo, MessageBus, MessageHandler, MessageType,
    get_message_bus, send_a2a_request, send_a2a_notification
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """基础Agent抽象类."""
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None,
        endpoint: str = "local"
    ):
        """初始化基础Agent.
        
        Args:
            agent_id: Agent唯一标识
            agent_name: Agent名称
            agent_type: Agent类型
            capabilities: Agent能力列表
            endpoint: 通信端点
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.endpoint = endpoint
        
        # 状态管理
        self._status = "inactive"
        self._last_heartbeat = datetime.now()
        self._metadata: Dict[str, Any] = {}
        
        # 通信组件
        self._message_bus: Optional[MessageBus] = None
        self._message_handler: Optional[MessageHandler] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 性能统计
        self._message_count = 0
        self._error_count = 0
        self._start_time: Optional[datetime] = None
    
    @property
    def status(self) -> str:
        """获取Agent状态."""
        return self._status
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """获取Agent元数据."""
        return self._metadata.copy()
    
    @property
    def is_running(self) -> bool:
        """检查Agent是否正在运行."""
        return self._running
    
    async def start(self):
        """启动Agent."""
        if self._running:
            logger.warning(f"Agent {self.agent_id} 已经在运行")
            return
        
        try:
            logger.info(f"启动Agent: {self.agent_id} ({self.agent_type})")
            
            # 初始化消息总线和处理器
            self._message_bus = get_message_bus()
            self._message_handler = MessageHandler(self.agent_id)
            
            # 注册消息处理器
            await self._register_handlers()
            
            # 创建Agent信息
            agent_info = AgentInfo(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                capabilities=self.capabilities,
                endpoint=self.endpoint,
                status="active",
                metadata=self._metadata
            )
            
            # 注册到消息总线
            await self._message_bus.register_agent(agent_info, self._message_handler)
            
            # 执行Agent特定的初始化
            await self._initialize()
            
            # 更新状态
            self._status = "active"
            self._running = True
            self._start_time = datetime.now()
            
            # 启动心跳任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"Agent {self.agent_id} 启动成功")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} 启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止Agent."""
        if not self._running:
            return
        
        logger.info(f"停止Agent: {self.agent_id}")
        
        try:
            # 更新状态
            self._status = "stopping"
            self._running = False
            
            # 停止心跳任务
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # 执行Agent特定的清理
            await self._cleanup()
            
            # 从消息总线注销
            if self._message_bus:
                await self._message_bus.unregister_agent(self.agent_id)
            
            self._status = "inactive"
            logger.info(f"Agent {self.agent_id} 已停止")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} 停止失败: {e}")
    
    async def send_request(
        self,
        to_agent: str,
        action: str,
        payload: Dict[str, Any],
        timeout: int = 30
    ) -> Optional[A2AMessage]:
        """发送请求消息.
        
        Args:
            to_agent: 目标Agent
            action: 操作名称
            payload: 消息载荷
            timeout: 超时时间
            
        Returns:
            响应消息
        """
        if not self._running:
            logger.error(f"Agent {self.agent_id} 未运行，无法发送消息")
            return None
        
        try:
            response = await send_a2a_request(
                self.agent_id, to_agent, action, payload, timeout
            )
            self._message_count += 1
            return response
            
        except Exception as e:
            logger.error(f"发送请求失败: {e}")
            self._error_count += 1
            return None
    
    async def send_notification(
        self,
        to_agent: str,
        action: str,
        payload: Dict[str, Any]
    ):
        """发送通知消息.
        
        Args:
            to_agent: 目标Agent
            action: 操作名称
            payload: 消息载荷
        """
        if not self._running:
            logger.error(f"Agent {self.agent_id} 未运行，无法发送消息")
            return
        
        try:
            await send_a2a_notification(self.agent_id, to_agent, action, payload)
            self._message_count += 1
            
        except Exception as e:
            logger.error(f"发送通知失败: {e}")
            self._error_count += 1
    
    async def broadcast_message(
        self,
        action: str,
        payload: Dict[str, Any],
        exclude_agents: Optional[List[str]] = None
    ):
        """广播消息.
        
        Args:
            action: 操作名称
            payload: 消息载荷
            exclude_agents: 排除的Agent列表
        """
        if not self._running or not self._message_bus:
            logger.error(f"Agent {self.agent_id} 未运行，无法广播消息")
            return
        
        try:
            await self._message_bus.broadcast_message(
                self.agent_id, action, payload, exclude_agents
            )
            self._message_count += 1
            
        except Exception as e:
            logger.error(f"广播消息失败: {e}")
            self._error_count += 1
    
    async def subscribe_action(self, action: str):
        """订阅操作.
        
        Args:
            action: 操作名称
        """
        if self._message_bus:
            await self._message_bus.subscribe_action(self.agent_id, action)
    
    async def publish_event(self, action: str, payload: Dict[str, Any]):
        """发布事件.
        
        Args:
            action: 操作名称
            payload: 消息载荷
        """
        if self._message_bus:
            await self._message_bus.publish_event(self.agent_id, action, payload)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Agent统计信息.
        
        Returns:
            统计信息字典
        """
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self._status,
            "uptime_seconds": uptime,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._message_count, 1),
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "capabilities": self.capabilities,
            "metadata": self._metadata
        }
    
    async def _register_handlers(self):
        """注册消息处理器."""
        if not self._message_handler:
            return
        
        # 注册通用处理器
        self._message_handler.register_handler("ping", self._handle_ping)
        self._message_handler.register_handler("get_status", self._handle_get_status)
        self._message_handler.register_handler("get_stats", self._handle_get_stats)
        self._message_handler.register_handler("get_capabilities", self._handle_get_capabilities)
        
        # 注册Agent特定的处理器
        await self._register_custom_handlers()
    
    async def _handle_ping(self, message: A2AMessage) -> Dict[str, Any]:
        """处理ping消息.
        
        Args:
            message: 消息对象
            
        Returns:
            pong响应
        """
        return {
            "pong": True,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
    
    async def _handle_get_status(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取状态消息.
        
        Args:
            message: 消息对象
            
        Returns:
            状态信息
        """
        return {
            "status": self._status,
            "uptime": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
            "last_heartbeat": self._last_heartbeat.isoformat()
        }
    
    async def _handle_get_stats(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取统计信息消息.
        
        Args:
            message: 消息对象
            
        Returns:
            统计信息
        """
        return self.get_stats()
    
    async def _handle_get_capabilities(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取能力列表消息.
        
        Args:
            message: 消息对象
            
        Returns:
            能力列表
        """
        return {
            "capabilities": self.capabilities,
            "agent_type": self.agent_type
        }
    
    async def _heartbeat_loop(self):
        """心跳循环."""
        while self._running:
            try:
                self._last_heartbeat = datetime.now()
                await self._send_heartbeat()
                await asyncio.sleep(30)  # 每30秒发送一次心跳
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳发送失败: {e}")
                await asyncio.sleep(30)
    
    async def _send_heartbeat(self):
        """发送心跳."""
        # 可以向协调器或监控Agent发送心跳
        # 这里暂时只更新时间戳
        pass
    
    # 抽象方法，子类必须实现
    
    @abstractmethod
    async def _initialize(self):
        """Agent特定的初始化逻辑."""
        pass
    
    @abstractmethod
    async def _cleanup(self):
        """Agent特定的清理逻辑."""
        pass
    
    @abstractmethod
    async def _register_custom_handlers(self):
        """注册Agent特定的消息处理器."""
        pass
    
    # 可选的钩子方法
    
    async def on_message_received(self, message: A2AMessage):
        """消息接收钩子.
        
        Args:
            message: 接收到的消息
        """
        pass
    
    async def on_message_sent(self, message: A2AMessage):
        """消息发送钩子.
        
        Args:
            message: 发送的消息
        """
        pass
    
    async def on_error(self, error: Exception, context: str):
        """错误处理钩子.
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        logger.error(f"Agent {self.agent_id} 错误 [{context}]: {error}")
        self._error_count += 1