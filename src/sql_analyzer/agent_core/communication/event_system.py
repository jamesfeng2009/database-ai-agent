"""统一的消息传递和事件系统."""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from weakref import WeakSet

from ..models.models import Event, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """事件总线，负责统一的消息传递和事件分发."""
    
    def __init__(self):
        """初始化事件总线."""
        self._subscribers: Dict[EventType, WeakSet] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """订阅事件类型.
        
        Args:
            event_type: 要订阅的事件类型
            handler: 事件处理函数
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = WeakSet()
            
            self._subscribers[event_type].add(handler)
            logger.debug(f"订阅事件类型: {event_type}, 处理器: {handler.__name__}")
    
    async def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """取消订阅事件类型.
        
        Args:
            event_type: 要取消订阅的事件类型
            handler: 事件处理函数
        """
        async with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(handler)
                logger.debug(f"取消订阅事件类型: {event_type}, 处理器: {handler.__name__}")
    
    async def publish(self, event: Event) -> None:
        """发布事件.
        
        Args:
            event: 要发布的事件
        """
        async with self._lock:
            # 记录事件历史
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history.pop(0)
            
            logger.debug(f"发布事件: {event.event_type}, ID: {event.event_id}")
        
        # 通知订阅者
        if event.event_type in self._subscribers:
            handlers = list(self._subscribers[event.event_type])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"事件处理器执行失败: {handler.__name__}, 错误: {e}")
    
    async def create_and_publish(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Event:
        """创建并发布事件.
        
        Args:
            event_type: 事件类型
            source: 事件源
            data: 事件数据
            session_id: 关联的会话ID
            
        Returns:
            创建的事件对象
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            session_id=session_id
        )
        await self.publish(event)
        return event
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """获取事件历史.
        
        Args:
            event_type: 过滤的事件类型，None表示获取所有类型
            limit: 返回的事件数量限制
            
        Returns:
            事件历史列表
        """
        if event_type is None:
            return self._event_history[-limit:]
        else:
            filtered_events = [e for e in self._event_history if e.event_type == event_type]
            return filtered_events[-limit:]
    
    def get_session_events(self, session_id: str, limit: int = 100) -> List[Event]:
        """获取特定会话的事件历史.
        
        Args:
            session_id: 会话ID
            limit: 返回的事件数量限制
            
        Returns:
            会话相关的事件列表
        """
        session_events = [e for e in self._event_history if e.session_id == session_id]
        return session_events[-limit:]
    
    def clear_history(self) -> None:
        """清空事件历史."""
        self._event_history.clear()
        logger.info("事件历史已清空")


# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例.
    
    Returns:
        全局事件总线实例
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


async def publish_event(
    event_type: EventType,
    source: str,
    data: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Event:
    """发布事件的便捷函数.
    
    Args:
        event_type: 事件类型
        source: 事件源
        data: 事件数据
        session_id: 关联的会话ID
        
    Returns:
        创建的事件对象
    """
    event_bus = get_event_bus()
    return await event_bus.create_and_publish(event_type, source, data, session_id)


async def subscribe_to_event(event_type: EventType, handler: Callable[[Event], None]) -> None:
    """订阅事件的便捷函数.
    
    Args:
        event_type: 要订阅的事件类型
        handler: 事件处理函数
    """
    event_bus = get_event_bus()
    await event_bus.subscribe(event_type, handler)