"""统一的消息传递和事件系统."""

import asyncio
import json
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

from ..models.models import (
    Event, EventType, EventPriority, EventFilter, EventRoute, 
    EventSubscription, EventBatch, EventPattern, EventAggregation,
    EventCorrelation, EventWindow, EventTrace, EventRule, 
    EventCondition, EventAction, RuleExecution, StreamWindow,
    StreamTransformation, StreamAggregator, BackpressureConfig, StreamMetrics
)

logger = logging.getLogger(__name__)


class EventPersistence:
    """事件持久化存储."""
    
    def __init__(self, db_path: str = "events.db"):
        """初始化事件持久化存储."""
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    priority TEXT NOT NULL,
                    correlation_id TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON events(session_id)
            """)
    
    async def store_event(self, event: Event) -> None:
        """存储事件到数据库."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events 
                (event_id, event_type, source, data, timestamp, session_id, 
                 priority, correlation_id, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.source,
                json.dumps(event.data),
                event.timestamp.isoformat(),
                event.session_id,
                event.priority.value,
                event.correlation_id,
                json.dumps(event.tags),
                json.dumps(event.metadata)
            ))
    
    async def query_events(
        self,
        event_types: Optional[List[EventType]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Event]:
        """查询事件."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        events = []
        for row in rows:
            event = Event(
                event_id=row["event_id"],
                event_type=EventType(row["event_type"]),
                source=row["source"],
                data=json.loads(row["data"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                session_id=row["session_id"],
                priority=EventPriority(row["priority"]),
                correlation_id=row["correlation_id"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            events.append(event)
        
        return events


class EventBus:
    """增强的事件总线，支持过滤、路由、优先级队列和批量处理."""
    
    def __init__(self, enable_persistence: bool = True, db_path: str = "events.db"):
        """初始化事件总线."""
        self._subscribers: Dict[EventType, WeakSet] = {}
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        self._lock = asyncio.Lock()
        
        # 优先级队列
        self._priority_queues: Dict[EventPriority, deque] = {
            priority: deque() for priority in EventPriority
        }
        
        # 过滤器和路由
        self._filters: Dict[str, EventFilter] = {}
        self._routes: Dict[str, EventRoute] = {}
        
        # 批量处理
        self._batch_size = 10
        self._batch_timeout = 5.0  # 秒
        self._current_batch: List[Event] = []
        self._batch_handlers: List[Callable[[EventBatch], None]] = []
        self._batch_timer: Optional[asyncio.Task] = None
        
        # 持久化
        self._persistence: Optional[EventPersistence] = None
        if enable_persistence:
            self._persistence = EventPersistence(db_path)
        
        # 处理任务
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 启动处理循环
        self._start_processing()
    
    def _start_processing(self):
        """启动事件处理循环."""
        self._processing_task = asyncio.create_task(self._process_events())
    
    async def _process_events(self):
        """事件处理循环."""
        while not self._shutdown_event.is_set():
            try:
                # 按优先级处理事件
                event = await self._get_next_event()
                if event:
                    await self._dispatch_event(event)
                else:
                    await asyncio.sleep(0.1)  # 没有事件时短暂休眠
            except Exception as e:
                logger.error(f"事件处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_event(self) -> Optional[Event]:
        """获取下一个要处理的事件（按优先级）."""
        async with self._lock:
            # 按优先级顺序检查队列
            for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                           EventPriority.MEDIUM, EventPriority.LOW]:
                if self._priority_queues[priority]:
                    return self._priority_queues[priority].popleft()
        return None
    
    async def _dispatch_event(self, event: Event):
        """分发事件给订阅者."""
        # 应用过滤器
        if not await self._apply_filters(event):
            return
        
        # 应用路由
        handlers = await self._apply_routes(event)
        
        # 如果没有路由匹配，使用默认订阅者
        if not handlers and event.event_type in self._subscribers:
            handlers = list(self._subscribers[event.event_type])
        
        # 执行处理器
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"事件处理器执行失败: {handler}, 错误: {e}")
    
    async def _apply_filters(self, event: Event) -> bool:
        """应用事件过滤器."""
        for filter_obj in self._filters.values():
            if not filter_obj.enabled:
                continue
            
            # 检查事件类型
            if filter_obj.event_types and event.event_type not in filter_obj.event_types:
                continue
            
            # 检查源模式
            if filter_obj.source_patterns:
                import re
                if not any(re.match(pattern, event.source) for pattern in filter_obj.source_patterns):
                    continue
            
            # 检查优先级阈值
            if filter_obj.priority_threshold:
                priority_order = {
                    EventPriority.LOW: 0,
                    EventPriority.MEDIUM: 1,
                    EventPriority.HIGH: 2,
                    EventPriority.CRITICAL: 3
                }
                if priority_order[event.priority] < priority_order[filter_obj.priority_threshold]:
                    continue
            
            # 检查数据过滤器
            if filter_obj.data_filters:
                if not self._match_data_filters(event.data, filter_obj.data_filters):
                    continue
            
            return True
        
        return True  # 如果没有过滤器，默认通过
    
    def _match_data_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """匹配数据过滤器."""
        for key, expected_value in filters.items():
            if key not in data:
                return False
            if data[key] != expected_value:
                return False
        return True
    
    async def _apply_routes(self, event: Event) -> List[Callable]:
        """应用事件路由."""
        handlers = []
        for route in self._routes.values():
            if not route.enabled:
                continue
            
            if await self._apply_filters(event):  # 使用路由的过滤器
                # 这里应该根据target_handlers获取实际的处理器
                # 简化实现，直接返回空列表
                pass
        
        return handlers
    
    async def subscribe(
        self, 
        event_type: EventType, 
        handler: Callable[[Event], None],
        filter_obj: Optional[EventFilter] = None
    ) -> str:
        """订阅事件类型.
        
        Args:
            event_type: 要订阅的事件类型
            handler: 事件处理函数
            filter_obj: 可选的过滤器
            
        Returns:
            订阅ID
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = WeakSet()
            
            self._subscribers[event_type].add(handler)
            
            # 创建订阅记录
            subscription = EventSubscription(
                event_type=event_type,
                handler_id=getattr(handler, '__name__', str(handler)),
                filter=filter_obj
            )
            self._subscriptions[subscription.subscription_id] = subscription
            
            logger.debug(f"订阅事件类型: {event_type}, 处理器: {handler.__name__ if hasattr(handler, '__name__') else str(handler)}")
            return subscription.subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """取消订阅.
        
        Args:
            subscription_id: 订阅ID
        """
        async with self._lock:
            if subscription_id in self._subscriptions:
                subscription = self._subscriptions[subscription_id]
                if subscription.event_type in self._subscribers:
                    # 这里需要根据handler_id找到实际的处理器，简化实现
                    pass
                del self._subscriptions[subscription_id]
                logger.debug(f"取消订阅: {subscription_id}")
    
    async def unsubscribe_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """取消订阅事件类型（兼容旧接口）.
        
        Args:
            event_type: 要取消订阅的事件类型
            handler: 事件处理函数
        """
        async with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(handler)
                logger.debug(f"取消订阅事件类型: {event_type}, 处理器: {handler.__name__ if hasattr(handler, '__name__') else str(handler)}")
    
    async def publish(self, event: Event) -> None:
        """发布事件到优先级队列.
        
        Args:
            event: 要发布的事件
        """
        async with self._lock:
            # 记录事件历史
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history.pop(0)
            
            # 添加到优先级队列
            self._priority_queues[event.priority].append(event)
            
            # 添加到批量处理
            self._current_batch.append(event)
            
            logger.debug(f"发布事件: {event.event_type}, ID: {event.event_id}, 优先级: {event.priority}")
        
        # 持久化存储
        if self._persistence:
            await self._persistence.store_event(event)
        
        # 检查是否需要处理批量事件
        await self._check_batch_processing()
    
    async def _check_batch_processing(self):
        """检查是否需要处理批量事件."""
        should_process = False
        
        async with self._lock:
            if len(self._current_batch) >= self._batch_size:
                should_process = True
        
        if should_process:
            await self._process_batch()
        elif not self._batch_timer:
            # 启动批量处理定时器
            self._batch_timer = asyncio.create_task(self._batch_timeout_handler())
    
    async def _batch_timeout_handler(self):
        """批量处理超时处理器."""
        await asyncio.sleep(self._batch_timeout)
        await self._process_batch()
    
    async def _process_batch(self):
        """处理批量事件."""
        batch_events = []
        
        async with self._lock:
            if self._current_batch:
                batch_events = self._current_batch.copy()
                self._current_batch.clear()
            
            # 取消定时器
            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None
        
        if batch_events:
            batch = EventBatch(events=batch_events)
            
            # 通知批量处理器
            for handler in self._batch_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(batch)
                    else:
                        handler(batch)
                except Exception as e:
                    logger.error(f"批量事件处理器执行失败: {handler}, 错误: {e}")
            
            logger.debug(f"处理批量事件: {len(batch_events)} 个事件")
    
    async def create_and_publish(
        self,
        event_type: EventType,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        correlation_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """创建并发布事件.
        
        Args:
            event_type: 事件类型
            source: 事件源
            data: 事件数据
            session_id: 关联的会话ID
            priority: 事件优先级
            correlation_id: 关联ID
            tags: 事件标签
            metadata: 事件元数据
            
        Returns:
            创建的事件对象
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            session_id=session_id,
            priority=priority,
            correlation_id=correlation_id,
            tags=tags or [],
            metadata=metadata or {}
        )
        await self.publish(event)
        return event
    
    # 过滤器管理
    async def add_filter(self, filter_obj: EventFilter) -> None:
        """添加事件过滤器."""
        async with self._lock:
            self._filters[filter_obj.filter_id] = filter_obj
            logger.debug(f"添加事件过滤器: {filter_obj.name}")
    
    async def remove_filter(self, filter_id: str) -> None:
        """移除事件过滤器."""
        async with self._lock:
            if filter_id in self._filters:
                del self._filters[filter_id]
                logger.debug(f"移除事件过滤器: {filter_id}")
    
    async def update_filter(self, filter_obj: EventFilter) -> None:
        """更新事件过滤器."""
        async with self._lock:
            self._filters[filter_obj.filter_id] = filter_obj
            logger.debug(f"更新事件过滤器: {filter_obj.name}")
    
    # 路由管理
    async def add_route(self, route: EventRoute) -> None:
        """添加事件路由."""
        async with self._lock:
            self._routes[route.route_id] = route
            logger.debug(f"添加事件路由: {route.name}")
    
    async def remove_route(self, route_id: str) -> None:
        """移除事件路由."""
        async with self._lock:
            if route_id in self._routes:
                del self._routes[route_id]
                logger.debug(f"移除事件路由: {route_id}")
    
    # 批量处理管理
    async def add_batch_handler(self, handler: Callable[[EventBatch], None]) -> None:
        """添加批量事件处理器."""
        async with self._lock:
            self._batch_handlers.append(handler)
            logger.debug(f"添加批量事件处理器: {handler}")
    
    async def remove_batch_handler(self, handler: Callable[[EventBatch], None]) -> None:
        """移除批量事件处理器."""
        async with self._lock:
            if handler in self._batch_handlers:
                self._batch_handlers.remove(handler)
                logger.debug(f"移除批量事件处理器: {handler}")
    
    async def set_batch_config(self, batch_size: int, batch_timeout: float) -> None:
        """设置批量处理配置."""
        async with self._lock:
            self._batch_size = batch_size
            self._batch_timeout = batch_timeout
            logger.debug(f"设置批量处理配置: size={batch_size}, timeout={batch_timeout}")
    
    # 历史回放功能
    async def replay_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None
    ) -> List[Event]:
        """回放历史事件."""
        if not self._persistence:
            logger.warning("事件持久化未启用，无法回放历史事件")
            return []
        
        events = await self._persistence.query_events(
            event_types=event_types,
            start_time=start_time,
            end_time=end_time,
            session_id=session_id
        )
        
        logger.info(f"回放历史事件: {len(events)} 个事件")
        return events
    
    # 动态订阅管理
    async def get_active_subscriptions(self) -> List[EventSubscription]:
        """获取活跃的订阅列表."""
        async with self._lock:
            return [sub for sub in self._subscriptions.values() if sub.active]
    
    async def update_subscription(self, subscription_id: str, **kwargs) -> None:
        """更新订阅配置."""
        async with self._lock:
            if subscription_id in self._subscriptions:
                subscription = self._subscriptions[subscription_id]
                for key, value in kwargs.items():
                    if hasattr(subscription, key):
                        setattr(subscription, key, value)
                logger.debug(f"更新订阅: {subscription_id}")
    
    # 统计和监控
    async def get_statistics(self) -> Dict[str, Any]:
        """获取事件总线统计信息."""
        async with self._lock:
            stats = {
                "total_subscribers": sum(len(handlers) for handlers in self._subscribers.values()),
                "total_subscriptions": len(self._subscriptions),
                "active_subscriptions": len([s for s in self._subscriptions.values() if s.active]),
                "total_filters": len(self._filters),
                "active_filters": len([f for f in self._filters.values() if f.enabled]),
                "total_routes": len(self._routes),
                "active_routes": len([r for r in self._routes.values() if r.enabled]),
                "queue_sizes": {
                    priority.value: len(queue) 
                    for priority, queue in self._priority_queues.items()
                },
                "current_batch_size": len(self._current_batch),
                "batch_handlers": len(self._batch_handlers),
                "history_size": len(self._event_history)
            }
        return stats
    
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
    
    async def shutdown(self) -> None:
        """关闭事件总线."""
        logger.info("正在关闭事件总线...")
        
        # 设置关闭标志
        self._shutdown_event.set()
        
        # 等待处理任务完成
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("事件处理任务关闭超时")
                self._processing_task.cancel()
        
        # 取消批量处理定时器
        if self._batch_timer:
            self._batch_timer.cancel()
        
        # 处理剩余的批量事件
        if self._current_batch:
            await self._process_batch()
        
        logger.info("事件总线已关闭")
    
    async def __aenter__(self):
        """异步上下文管理器入口."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口."""
        await self.shutdown()


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
    session_id: Optional[str] = None,
    priority: EventPriority = EventPriority.MEDIUM,
    correlation_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """发布事件的便捷函数.
    
    Args:
        event_type: 事件类型
        source: 事件源
        data: 事件数据
        session_id: 关联的会话ID
        priority: 事件优先级
        correlation_id: 关联ID
        tags: 事件标签
        metadata: 事件元数据
        
    Returns:
        创建的事件对象
    """
    event_bus = get_event_bus()
    return await event_bus.create_and_publish(
        event_type, source, data, session_id, priority, correlation_id, tags, metadata
    )


async def subscribe_to_event(
    event_type: EventType, 
    handler: Callable[[Event], None],
    filter_obj: Optional[EventFilter] = None
) -> str:
    """订阅事件的便捷函数.
    
    Args:
        event_type: 要订阅的事件类型
        handler: 事件处理函数
        filter_obj: 可选的过滤器
        
    Returns:
        订阅ID
    """
    event_bus = get_event_bus()
    return await event_bus.subscribe(event_type, handler, filter_obj)


class EventDriver:
    """事件驱动器 - 支持复杂事件处理逻辑."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化事件驱动器."""
        self.event_bus = event_bus or get_event_bus()
        self._patterns: Dict[str, EventPattern] = {}
        self._aggregations: Dict[str, EventAggregation] = {}
        self._correlations: Dict[str, EventCorrelation] = {}
        self._windows: Dict[str, EventWindow] = {}
        self._traces: Dict[str, EventTrace] = {}
        self._lock = asyncio.Lock()
        
        # 事件流处理
        self._event_stream: deque = deque(maxlen=10000)
        self._pattern_matches: List[Dict[str, Any]] = []
        
        # 启动处理任务
        self._processing_task = asyncio.create_task(self._process_event_stream())
        self._shutdown_event = asyncio.Event()
    
    async def register_pattern(self, pattern: EventPattern) -> None:
        """注册事件模式."""
        async with self._lock:
            self._patterns[pattern.pattern_id] = pattern
            logger.debug(f"注册事件模式: {pattern.name}")
    
    async def register_aggregation(self, aggregation: EventAggregation) -> None:
        """注册事件聚合."""
        async with self._lock:
            self._aggregations[aggregation.aggregation_id] = aggregation
            logger.debug(f"注册事件聚合: {aggregation.name}")
    
    async def register_correlation(self, correlation: EventCorrelation) -> None:
        """注册事件关联."""
        async with self._lock:
            self._correlations[correlation.correlation_id] = correlation
            logger.debug(f"注册事件关联: {correlation.name}")
    
    async def process_event(self, event: Event) -> None:
        """处理单个事件."""
        async with self._lock:
            # 添加到事件流
            self._event_stream.append(event)
            
            # 更新活跃窗口
            await self._update_windows(event)
            
            # 更新事件追踪
            await self._update_traces(event)
    
    async def _process_event_stream(self):
        """处理事件流的主循环."""
        while not self._shutdown_event.is_set():
            try:
                # 检测模式匹配
                await self._detect_patterns()
                
                # 执行聚合
                await self._execute_aggregations()
                
                # 分析关联
                await self._analyze_correlations()
                
                # 清理过期窗口
                await self._cleanup_expired_windows()
                
                await asyncio.sleep(1)  # 每秒处理一次
                
            except Exception as e:
                logger.error(f"事件流处理错误: {e}")
                await asyncio.sleep(5)
    
    async def _detect_patterns(self):
        """检测事件模式."""
        current_time = datetime.now()
        
        for pattern in self._patterns.values():
            if not pattern.enabled:
                continue
            
            # 获取时间窗口内的事件
            window_start = current_time - pattern.time_window
            window_events = [
                event for event in self._event_stream
                if event.timestamp >= window_start
            ]
            
            # 检查模式匹配
            matches = await self._match_pattern(pattern, window_events)
            if matches:
                self._pattern_matches.extend(matches)
                logger.info(f"检测到模式匹配: {pattern.name}, 匹配数: {len(matches)}")
    
    async def _match_pattern(self, pattern: EventPattern, events: List[Event]) -> List[Dict[str, Any]]:
        """匹配事件模式."""
        matches = []
        
        # 简化的模式匹配逻辑
        # 检查事件序列是否按顺序出现
        if len(pattern.event_sequence) == 0:
            return matches
        
        # 按时间排序事件
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # 查找匹配的事件序列
        i = 0
        sequence_start = 0
        matched_events = []
        
        for event in sorted_events:
            if i < len(pattern.event_sequence) and event.event_type == pattern.event_sequence[i]:
                if i == 0:
                    sequence_start = len(matched_events)
                matched_events.append(event)
                i += 1
                
                # 完整序列匹配
                if i == len(pattern.event_sequence):
                    # 检查条件
                    if await self._check_pattern_conditions(pattern, matched_events):
                        matches.append({
                            "pattern_id": pattern.pattern_id,
                            "pattern_name": pattern.name,
                            "matched_events": matched_events.copy(),
                            "match_time": datetime.now()
                        })
                    
                    # 重置匹配状态
                    i = 0
                    matched_events = []
            else:
                # 重置匹配状态
                i = 0
                matched_events = []
        
        return matches
    
    async def _check_pattern_conditions(self, pattern: EventPattern, events: List[Event]) -> bool:
        """检查模式条件."""
        if not pattern.conditions:
            return True
        
        # 简化的条件检查
        for condition_key, condition_value in pattern.conditions.items():
            # 这里可以实现更复杂的条件逻辑
            pass
        
        return True
    
    async def _execute_aggregations(self):
        """执行事件聚合."""
        current_time = datetime.now()
        
        for aggregation in self._aggregations.values():
            # 获取时间窗口内的相关事件
            window_start = current_time - aggregation.time_window
            relevant_events = [
                event for event in self._event_stream
                if (event.timestamp >= window_start and 
                    event.event_type in aggregation.event_types)
            ]
            
            if relevant_events:
                result = await self._aggregate_events(aggregation, relevant_events)
                aggregation.result = result
                logger.debug(f"执行聚合: {aggregation.name}, 结果: {result}")
    
    async def _aggregate_events(self, aggregation: EventAggregation, events: List[Event]) -> Dict[str, Any]:
        """聚合事件数据."""
        result = {
            "count": len(events),
            "time_range": {
                "start": min(e.timestamp for e in events).isoformat(),
                "end": max(e.timestamp for e in events).isoformat()
            }
        }
        
        # 根据聚合函数执行不同的聚合逻辑
        if aggregation.aggregation_function == "count":
            result["value"] = len(events)
        elif aggregation.aggregation_function == "sum":
            # 假设聚合data中的某个数值字段
            total = sum(event.data.get("value", 0) for event in events)
            result["value"] = total
        elif aggregation.aggregation_function == "avg":
            values = [event.data.get("value", 0) for event in events]
            result["value"] = sum(values) / len(values) if values else 0
        elif aggregation.aggregation_function == "max":
            values = [event.data.get("value", 0) for event in events]
            result["value"] = max(values) if values else 0
        elif aggregation.aggregation_function == "min":
            values = [event.data.get("value", 0) for event in events]
            result["value"] = min(values) if values else 0
        
        # 分组聚合
        if aggregation.group_by:
            groups = defaultdict(list)
            for event in events:
                group_key = tuple(
                    str(event.data.get(field, "")) for field in aggregation.group_by
                )
                groups[group_key].append(event)
            
            result["groups"] = {}
            for group_key, group_events in groups.items():
                group_result = await self._aggregate_events(
                    EventAggregation(
                        name=f"{aggregation.name}_group",
                        event_types=aggregation.event_types,
                        aggregation_function=aggregation.aggregation_function,
                        time_window=aggregation.time_window
                    ),
                    group_events
                )
                result["groups"]["|".join(group_key)] = group_result
        
        return result
    
    async def _analyze_correlations(self):
        """分析事件关联."""
        current_time = datetime.now()
        
        for correlation in self._correlations.values():
            # 获取时间窗口内的主事件
            window_start = current_time - correlation.time_window
            primary_events = [
                event for event in self._event_stream
                if (event.timestamp >= window_start and 
                    event.event_type == correlation.primary_event_type)
            ]
            
            # 分析每个主事件的关联
            for primary_event in primary_events:
                await self._analyze_event_correlation(correlation, primary_event)
    
    async def _analyze_event_correlation(self, correlation: EventCorrelation, primary_event: Event):
        """分析单个事件的关联性."""
        correlation_key_value = primary_event.data.get(correlation.correlation_key)
        if not correlation_key_value:
            return
        
        # 查找相关事件
        window_start = primary_event.timestamp - correlation.time_window
        window_end = primary_event.timestamp + correlation.time_window
        
        related_events = [
            event for event in self._event_stream
            if (window_start <= event.timestamp <= window_end and
                event.event_type in correlation.related_event_types and
                event.data.get(correlation.correlation_key) == correlation_key_value)
        ]
        
        if related_events:
            # 计算关联强度
            strength = len(related_events) / len(correlation.related_event_types)
            correlation.strength = max(correlation.strength, strength)
            
            logger.debug(f"发现事件关联: {correlation.name}, 强度: {strength}")
    
    async def _update_windows(self, event: Event):
        """更新事件窗口."""
        current_time = datetime.now()
        
        # 创建新窗口或更新现有窗口
        for window in list(self._windows.values()):
            if not window.is_active:
                continue
            
            # 检查事件是否属于此窗口
            if self._event_belongs_to_window(event, window):
                window.events.append(event)
                
                # 更新窗口结束时间
                if window.window_type == "session":
                    window.end_time = current_time + window.size
    
    def _event_belongs_to_window(self, event: Event, window: EventWindow) -> bool:
        """检查事件是否属于指定窗口."""
        if window.window_type == "tumbling":
            return (window.start_time <= event.timestamp < 
                   window.start_time + window.size)
        elif window.window_type == "sliding":
            return (event.timestamp >= window.start_time and
                   event.timestamp < window.start_time + window.size)
        elif window.window_type == "session":
            if not window.events:
                return True
            last_event_time = max(e.timestamp for e in window.events)
            return event.timestamp <= last_event_time + window.size
        
        return False
    
    async def _update_traces(self, event: Event):
        """更新事件链路追踪."""
        # 检查是否是新的追踪起点
        if event.correlation_id and event.correlation_id not in self._traces:
            trace = EventTrace(
                trace_id=event.correlation_id,
                root_event_id=event.event_id
            )
            self._traces[event.correlation_id] = trace
        
        # 添加事件到相应的追踪链
        if event.correlation_id and event.correlation_id in self._traces:
            trace = self._traces[event.correlation_id]
            trace.events.append(event)
            
            # 更新依赖关系
            if len(trace.events) > 1:
                prev_event = trace.events[-2]
                if prev_event.event_id not in trace.dependencies:
                    trace.dependencies[prev_event.event_id] = []
                trace.dependencies[prev_event.event_id].append(event.event_id)
    
    async def _cleanup_expired_windows(self):
        """清理过期的窗口."""
        current_time = datetime.now()
        expired_windows = []
        
        for window_id, window in self._windows.items():
            if window.end_time and current_time > window.end_time:
                window.is_active = False
                expired_windows.append(window_id)
        
        # 移除过期窗口
        for window_id in expired_windows:
            del self._windows[window_id]
            logger.debug(f"清理过期窗口: {window_id}")
    
    async def create_sliding_window(
        self, 
        window_size: timedelta, 
        slide_interval: timedelta
    ) -> str:
        """创建滑动窗口."""
        window = EventWindow(
            window_type="sliding",
            size=window_size,
            slide=slide_interval
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.debug(f"创建滑动窗口: {window.window_id}")
        return window.window_id
    
    async def create_tumbling_window(self, window_size: timedelta) -> str:
        """创建翻滚窗口."""
        window = EventWindow(
            window_type="tumbling",
            size=window_size
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.debug(f"创建翻滚窗口: {window.window_id}")
        return window.window_id
    
    async def create_session_window(self, session_timeout: timedelta) -> str:
        """创建会话窗口."""
        window = EventWindow(
            window_type="session",
            size=session_timeout
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.debug(f"创建会话窗口: {window.window_id}")
        return window.window_id
    
    async def get_pattern_matches(self, pattern_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取模式匹配结果."""
        if pattern_id:
            return [match for match in self._pattern_matches if match["pattern_id"] == pattern_id]
        return self._pattern_matches.copy()
    
    async def get_aggregation_result(self, aggregation_id: str) -> Optional[Dict[str, Any]]:
        """获取聚合结果."""
        if aggregation_id in self._aggregations:
            return self._aggregations[aggregation_id].result
        return None
    
    async def get_correlation_strength(self, correlation_id: str) -> float:
        """获取关联强度."""
        if correlation_id in self._correlations:
            return self._correlations[correlation_id].strength
        return 0.0
    
    async def get_event_trace(self, trace_id: str) -> Optional[EventTrace]:
        """获取事件追踪链."""
        return self._traces.get(trace_id)
    
    async def get_window_events(self, window_id: str) -> List[Event]:
        """获取窗口中的事件."""
        if window_id in self._windows:
            return self._windows[window_id].events.copy()
        return []
    
    async def shutdown(self):
        """关闭事件驱动器."""
        logger.info("正在关闭事件驱动器...")
        
        self._shutdown_event.set()
        
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("事件驱动器处理任务关闭超时")
                self._processing_task.cancel()
        
        logger.info("事件驱动器已关闭")


# 全局事件驱动器实例
_global_event_driver: Optional[EventDriver] = None


def get_event_driver() -> EventDriver:
    """获取全局事件驱动器实例."""
    global _global_event_driver
    if _global_event_driver is None:
        _global_event_driver = EventDriver()
    return _global_event_driver


class EventRulesEngine:
    """事件规则引擎 - 基于规则的事件触发和响应机制."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化事件规则引擎."""
        self.event_bus = event_bus or get_event_bus()
        self._rules: Dict[str, EventRule] = {}
        self._rule_executions: List[RuleExecution] = []
        self._lock = asyncio.Lock()
        
        # 动作执行器映射
        self._action_executors: Dict[str, Callable] = {
            "notify": self._execute_notify_action,
            "execute": self._execute_command_action,
            "trigger": self._execute_trigger_action,
            "log": self._execute_log_action,
            "webhook": self._execute_webhook_action
        }
        
        # 性能监控
        self._performance_stats = {
            "total_rules_processed": 0,
            "total_conditions_evaluated": 0,
            "total_actions_executed": 0,
            "average_execution_time": 0.0,
            "failed_executions": 0
        }
        
        # 订阅所有事件类型进行规则处理
        asyncio.create_task(self._subscribe_to_events())
    
    async def _subscribe_to_events(self):
        """订阅所有事件类型进行规则处理."""
        # 为每种事件类型订阅处理器
        for event_type in EventType:
            await self.event_bus.subscribe(event_type, self._process_event_for_rules)
    
    async def add_rule(self, rule: EventRule) -> None:
        """添加事件规则."""
        async with self._lock:
            self._rules[rule.rule_id] = rule
            logger.info(f"添加事件规则: {rule.name} (ID: {rule.rule_id})")
    
    async def remove_rule(self, rule_id: str) -> None:
        """移除事件规则."""
        async with self._lock:
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                del self._rules[rule_id]
                logger.info(f"移除事件规则: {rule.name} (ID: {rule_id})")
    
    async def update_rule(self, rule: EventRule) -> None:
        """更新事件规则."""
        async with self._lock:
            rule.updated_at = datetime.now()
            self._rules[rule.rule_id] = rule
            logger.info(f"更新事件规则: {rule.name} (ID: {rule.rule_id})")
    
    async def enable_rule(self, rule_id: str) -> None:
        """启用规则."""
        async with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = True
                logger.info(f"启用规则: {rule_id}")
    
    async def disable_rule(self, rule_id: str) -> None:
        """禁用规则."""
        async with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = False
                logger.info(f"禁用规则: {rule_id}")
    
    async def _process_event_for_rules(self, event: Event) -> None:
        """为规则处理事件."""
        start_time = datetime.now()
        
        try:
            # 获取匹配的规则
            matching_rules = await self._get_matching_rules(event)
            
            # 按优先级排序规则
            matching_rules.sort(key=lambda r: r.priority)
            
            # 执行匹配的规则
            for rule in matching_rules:
                await self._execute_rule(rule, event)
            
            # 更新性能统计
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._update_performance_stats(len(matching_rules), execution_time)
            
        except Exception as e:
            logger.error(f"规则处理事件时发生错误: {e}")
            self._performance_stats["failed_executions"] += 1
    
    async def _get_matching_rules(self, event: Event) -> List[EventRule]:
        """获取匹配事件的规则."""
        matching_rules = []
        
        async with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                
                # 检查事件类型是否匹配
                if event.event_type not in rule.trigger_events:
                    continue
                
                # 检查冷却期
                if rule.cooldown_period and rule.last_triggered:
                    if datetime.now() - rule.last_triggered < rule.cooldown_period:
                        continue
                
                # 检查条件
                if await self._evaluate_conditions(rule.conditions, event):
                    matching_rules.append(rule)
        
        return matching_rules
    
    async def _evaluate_conditions(self, conditions: List[EventCondition], event: Event) -> bool:
        """评估规则条件."""
        if not conditions:
            return True
        
        self._performance_stats["total_conditions_evaluated"] += len(conditions)
        
        # 简化的条件评估逻辑
        results = []
        
        for condition in conditions:
            result = await self._evaluate_single_condition(condition, event)
            results.append((result, condition.logical_operator))
        
        # 处理逻辑操作符
        if not results:
            return True
        
        # 从左到右评估逻辑表达式
        final_result = results[0][0]
        
        for i in range(1, len(results)):
            result, prev_operator = results[i-1]
            current_result = results[i][0]
            
            if prev_operator == "AND":
                final_result = final_result and current_result
            elif prev_operator == "OR":
                final_result = final_result or current_result
            elif prev_operator == "NOT":
                final_result = final_result and not current_result
        
        return final_result
    
    async def _evaluate_single_condition(self, condition: EventCondition, event: Event) -> bool:
        """评估单个条件."""
        # 获取字段值
        field_value = self._get_field_value(event, condition.field)
        condition_value = condition.value
        
        # 根据操作符进行比较
        if condition.operator == "eq":
            return field_value == condition_value
        elif condition.operator == "ne":
            return field_value != condition_value
        elif condition.operator == "gt":
            return field_value > condition_value
        elif condition.operator == "lt":
            return field_value < condition_value
        elif condition.operator == "gte":
            return field_value >= condition_value
        elif condition.operator == "lte":
            return field_value <= condition_value
        elif condition.operator == "contains":
            return str(condition_value) in str(field_value)
        elif condition.operator == "matches":
            import re
            return bool(re.match(str(condition_value), str(field_value)))
        elif condition.operator == "in":
            return field_value in condition_value
        elif condition.operator == "not_in":
            return field_value not in condition_value
        else:
            logger.warning(f"未知的条件操作符: {condition.operator}")
            return False
    
    def _get_field_value(self, event: Event, field_path: str) -> Any:
        """获取事件字段值，支持嵌套字段访问."""
        try:
            # 支持点号分隔的嵌套字段访问
            parts = field_path.split(".")
            value = event
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            return value
        except Exception:
            return None
    
    async def _execute_rule(self, rule: EventRule, event: Event) -> None:
        """执行规则."""
        start_time = datetime.now()
        execution = RuleExecution(
            rule_id=rule.rule_id,
            trigger_event=event
        )
        
        try:
            # 更新规则触发统计
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            # 执行所有动作
            for action in rule.actions:
                if action.enabled:
                    success = await self._execute_action(action, event, rule)
                    if success:
                        execution.actions_executed.append(action.action_id)
                    else:
                        execution.success = False
            
            # 记录执行时间
            execution.execution_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"执行规则: {rule.name}, 触发事件: {event.event_type}")
            
        except Exception as e:
            execution.success = False
            execution.error_message = str(e)
            logger.error(f"执行规则失败: {rule.name}, 错误: {e}")
        
        finally:
            # 记录执行历史
            async with self._lock:
                self._rule_executions.append(execution)
                # 限制执行历史大小
                if len(self._rule_executions) > 10000:
                    self._rule_executions = self._rule_executions[-5000:]
    
    async def _execute_action(self, action: EventAction, event: Event, rule: EventRule) -> bool:
        """执行动作."""
        try:
            executor = self._action_executors.get(action.action_type)
            if not executor:
                logger.warning(f"未知的动作类型: {action.action_type}")
                return False
            
            # 执行动作，支持重试
            for attempt in range(action.max_retries + 1):
                try:
                    await executor(action, event, rule)
                    self._performance_stats["total_actions_executed"] += 1
                    return True
                except Exception as e:
                    if attempt < action.max_retries:
                        logger.warning(f"动作执行失败，重试 {attempt + 1}/{action.max_retries}: {e}")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                    else:
                        raise e
            
            return False
            
        except Exception as e:
            logger.error(f"执行动作失败: {action.action_type}, 错误: {e}")
            return False
    
    async def _execute_notify_action(self, action: EventAction, event: Event, rule: EventRule) -> None:
        """执行通知动作."""
        message = action.parameters.get("message", f"规则 {rule.name} 被触发")
        target = action.target
        
        # 这里可以集成各种通知渠道
        logger.info(f"通知 [{target}]: {message}")
        
        # 发布通知事件
        await self.event_bus.create_and_publish(
            event_type=EventType.SYSTEM_HEALTH_DEGRADED,  # 使用合适的事件类型
            source="rules_engine",
            data={
                "notification_target": target,
                "message": message,
                "original_event": event.dict(),
                "rule_name": rule.name
            }
        )
    
    async def _execute_command_action(self, action: EventAction, event: Event, rule: EventRule) -> None:
        """执行命令动作."""
        command = action.parameters.get("command")
        if not command:
            raise ValueError("命令动作缺少 command 参数")
        
        # 安全检查
        allowed_commands = action.parameters.get("allowed_commands", [])
        if allowed_commands and command not in allowed_commands:
            raise ValueError(f"命令 {command} 不在允许列表中")
        
        logger.info(f"执行命令: {command}")
        # 这里可以实际执行命令，但为了安全起见，只记录日志
    
    async def _execute_trigger_action(self, action: EventAction, event: Event, rule: EventRule) -> None:
        """执行触发动作."""
        target_event_type = action.parameters.get("event_type")
        if not target_event_type:
            raise ValueError("触发动作缺少 event_type 参数")
        
        # 创建新事件
        await self.event_bus.create_and_publish(
            event_type=EventType(target_event_type),
            source=f"rules_engine_{rule.rule_id}",
            data={
                "triggered_by_rule": rule.name,
                "original_event": event.dict(),
                **action.parameters.get("data", {})
            },
            correlation_id=event.correlation_id
        )
        
        logger.info(f"触发新事件: {target_event_type}")
    
    async def _execute_log_action(self, action: EventAction, event: Event, rule: EventRule) -> None:
        """执行日志动作."""
        log_level = action.parameters.get("level", "info").lower()
        message = action.parameters.get("message", f"规则 {rule.name} 触发")
        
        # 格式化消息
        formatted_message = message.format(
            rule_name=rule.name,
            event_type=event.event_type,
            event_source=event.source,
            **event.data
        )
        
        # 根据级别记录日志
        if log_level == "debug":
            logger.debug(formatted_message)
        elif log_level == "info":
            logger.info(formatted_message)
        elif log_level == "warning":
            logger.warning(formatted_message)
        elif log_level == "error":
            logger.error(formatted_message)
        else:
            logger.info(formatted_message)
    
    async def _execute_webhook_action(self, action: EventAction, event: Event, rule: EventRule) -> None:
        """执行Webhook动作."""
        url = action.parameters.get("url")
        if not url:
            raise ValueError("Webhook动作缺少 url 参数")
        
        payload = {
            "rule_name": rule.name,
            "rule_id": rule.rule_id,
            "event": event.dict(),
            "timestamp": datetime.now().isoformat(),
            **action.parameters.get("payload", {})
        }
        
        # 这里可以实际发送HTTP请求
        logger.info(f"发送Webhook到: {url}, 载荷: {payload}")
    
    async def _update_performance_stats(self, rules_processed: int, execution_time: float) -> None:
        """更新性能统计."""
        self._performance_stats["total_rules_processed"] += rules_processed
        
        # 更新平均执行时间
        total_executions = self._performance_stats["total_rules_processed"]
        if total_executions > 0:
            current_avg = self._performance_stats["average_execution_time"]
            self._performance_stats["average_execution_time"] = (
                (current_avg * (total_executions - rules_processed) + execution_time) / total_executions
            )
    
    async def get_rules(self, enabled_only: bool = False) -> List[EventRule]:
        """获取规则列表."""
        async with self._lock:
            rules = list(self._rules.values())
            if enabled_only:
                rules = [rule for rule in rules if rule.enabled]
            return rules
    
    async def get_rule(self, rule_id: str) -> Optional[EventRule]:
        """获取指定规则."""
        async with self._lock:
            return self._rules.get(rule_id)
    
    async def get_rule_executions(
        self, 
        rule_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[RuleExecution]:
        """获取规则执行历史."""
        async with self._lock:
            executions = self._rule_executions
            if rule_id:
                executions = [ex for ex in executions if ex.rule_id == rule_id]
            return executions[-limit:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计."""
        async with self._lock:
            return self._performance_stats.copy()
    
    async def clear_execution_history(self) -> None:
        """清空执行历史."""
        async with self._lock:
            self._rule_executions.clear()
            logger.info("规则执行历史已清空")
    
    async def load_rules_from_config(self, config: Dict[str, Any]) -> None:
        """从配置加载规则."""
        for rule_config in config.get("rules", []):
            try:
                # 解析条件
                conditions = []
                for cond_config in rule_config.get("conditions", []):
                    condition = EventCondition(**cond_config)
                    conditions.append(condition)
                
                # 解析动作
                actions = []
                for action_config in rule_config.get("actions", []):
                    action = EventAction(**action_config)
                    actions.append(action)
                
                # 创建规则
                rule = EventRule(
                    name=rule_config["name"],
                    description=rule_config.get("description", ""),
                    trigger_events=[EventType(et) for et in rule_config["trigger_events"]],
                    conditions=conditions,
                    actions=actions,
                    priority=rule_config.get("priority", 100),
                    enabled=rule_config.get("enabled", True),
                    cooldown_period=timedelta(seconds=rule_config.get("cooldown_seconds", 0)) if rule_config.get("cooldown_seconds") else None
                )
                
                await self.add_rule(rule)
                
            except Exception as e:
                logger.error(f"加载规则配置失败: {rule_config.get('name', 'unknown')}, 错误: {e}")
    
    async def export_rules_config(self) -> Dict[str, Any]:
        """导出规则配置."""
        async with self._lock:
            rules_config = []
            for rule in self._rules.values():
                rule_config = {
                    "name": rule.name,
                    "description": rule.description,
                    "trigger_events": [et.value for et in rule.trigger_events],
                    "conditions": [cond.dict() for cond in rule.conditions],
                    "actions": [action.dict() for action in rule.actions],
                    "priority": rule.priority,
                    "enabled": rule.enabled
                }
                if rule.cooldown_period:
                    rule_config["cooldown_seconds"] = rule.cooldown_period.total_seconds()
                
                rules_config.append(rule_config)
            
            return {"rules": rules_config}


# 全局事件规则引擎实例
_global_rules_engine: Optional[EventRulesEngine] = None


def get_rules_engine() -> EventRulesEngine:
    """获取全局事件规则引擎实例."""
    global _global_rules_engine
    if _global_rules_engine is None:
        _global_rules_engine = EventRulesEngine()
    return _global_rules_engine


class EventStream:
    """事件流类."""
    
    def __init__(self, stream_id: str, source_types: List[EventType]):
        """初始化事件流."""
        self.stream_id = stream_id
        self.source_types = source_types
        self.events: asyncio.Queue = asyncio.Queue()
        self.subscribers: List[Callable[[Event], None]] = []
        self.is_active = True
        self.metrics = StreamMetrics(stream_id=stream_id)
    
    async def add_event(self, event: Event) -> None:
        """添加事件到流."""
        if self.is_active and event.event_type in self.source_types:
            await self.events.put(event)
            self.metrics.events_processed += 1
    
    async def get_event(self) -> Optional[Event]:
        """从流中获取事件."""
        try:
            return await asyncio.wait_for(self.events.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def subscribe(self, handler: Callable[[Event], None]) -> None:
        """订阅流事件."""
        self.subscribers.append(handler)
    
    async def close(self) -> None:
        """关闭流."""
        self.is_active = False


class EventStreamProcessor:
    """事件流处理器 - 支持实时事件流分析."""
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """初始化事件流处理器."""
        self.event_bus = event_bus or get_event_bus()
        self._streams: Dict[str, EventStream] = {}
        self._windows: Dict[str, StreamWindow] = {}
        self._transformations: Dict[str, StreamTransformation] = {}
        self._aggregators: Dict[str, StreamAggregator] = {}
        self._backpressure_config = BackpressureConfig()
        self._lock = asyncio.Lock()
        
        # 处理任务
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        
        # 背压控制
        self._buffer_sizes: Dict[str, int] = {}
        self._throttle_timers: Dict[str, float] = {}
        
        # 性能监控
        self._start_time = datetime.now()
        self._last_metrics_update = datetime.now()
    
    async def create_stream(
        self, 
        stream_id: str, 
        source_types: List[EventType],
        backpressure_config: Optional[BackpressureConfig] = None
    ) -> EventStream:
        """创建事件流."""
        async with self._lock:
            if stream_id in self._streams:
                raise ValueError(f"流 {stream_id} 已存在")
            
            stream = EventStream(stream_id, source_types)
            self._streams[stream_id] = stream
            self._buffer_sizes[stream_id] = 0
            
            if backpressure_config:
                # 为特定流设置背压配置
                pass
            
            # 启动流处理任务
            task = asyncio.create_task(self._process_stream(stream))
            self._processing_tasks[stream_id] = task
            
            # 订阅相关事件类型
            for event_type in source_types:
                await self.event_bus.subscribe(
                    event_type, 
                    lambda event, s=stream: asyncio.create_task(s.add_event(event))
                )
            
            logger.info(f"创建事件流: {stream_id}, 源类型: {source_types}")
            return stream
    
    async def _process_stream(self, stream: EventStream) -> None:
        """处理事件流."""
        while not self._shutdown_event.is_set() and stream.is_active:
            try:
                event = await stream.get_event()
                if event:
                    # 应用转换
                    transformed_event = await self._apply_transformations(stream.stream_id, event)
                    if transformed_event:
                        # 更新窗口
                        await self._update_stream_windows(stream.stream_id, transformed_event)
                        
                        # 执行聚合
                        await self._execute_stream_aggregations(stream.stream_id, transformed_event)
                        
                        # 通知订阅者
                        for subscriber in stream.subscribers:
                            try:
                                if asyncio.iscoroutinefunction(subscriber):
                                    await subscriber(transformed_event)
                                else:
                                    subscriber(transformed_event)
                            except Exception as e:
                                logger.error(f"流订阅者执行失败: {e}")
                        
                        # 背压控制
                        await self._handle_backpressure(stream.stream_id)
                
                # 更新指标
                await self._update_stream_metrics(stream.stream_id)
                
            except Exception as e:
                logger.error(f"流处理错误: {stream.stream_id}, 错误: {e}")
                stream.metrics.error_count += 1
                await asyncio.sleep(1)
    
    async def _apply_transformations(self, stream_id: str, event: Event) -> Optional[Event]:
        """应用流转换."""
        current_event = event
        
        for transformation in self._transformations.values():
            if not transformation.enabled:
                continue
            
            try:
                if transformation.transformation_type == "filter":
                    if not await self._apply_filter_transformation(transformation, current_event):
                        return None
                elif transformation.transformation_type == "map":
                    current_event = await self._apply_map_transformation(transformation, current_event)
                elif transformation.transformation_type == "flatmap":
                    # FlatMap 可能产生多个事件，这里简化处理
                    events = await self._apply_flatmap_transformation(transformation, current_event)
                    if events:
                        current_event = events[0]  # 简化：只取第一个
                    else:
                        return None
                elif transformation.transformation_type == "reduce":
                    current_event = await self._apply_reduce_transformation(transformation, current_event)
                
            except Exception as e:
                logger.error(f"转换执行失败: {transformation.name}, 错误: {e}")
        
        return current_event
    
    async def _apply_filter_transformation(self, transformation: StreamTransformation, event: Event) -> bool:
        """应用过滤转换."""
        filter_condition = transformation.parameters.get("condition")
        if not filter_condition:
            return True
        
        # 简化的过滤逻辑
        field = filter_condition.get("field")
        operator = filter_condition.get("operator", "eq")
        value = filter_condition.get("value")
        
        if field and hasattr(event, field):
            event_value = getattr(event, field)
            if operator == "eq":
                return event_value == value
            elif operator == "ne":
                return event_value != value
            elif operator == "gt":
                return event_value > value
            elif operator == "lt":
                return event_value < value
            elif operator == "contains":
                return str(value) in str(event_value)
        
        return True
    
    async def _apply_map_transformation(self, transformation: StreamTransformation, event: Event) -> Event:
        """应用映射转换."""
        mapping = transformation.parameters.get("mapping", {})
        
        # 创建新事件，应用映射
        new_data = event.data.copy()
        for source_field, target_field in mapping.items():
            if source_field in event.data:
                new_data[target_field] = event.data[source_field]
        
        # 创建转换后的事件
        transformed_event = Event(
            event_type=event.event_type,
            source=f"{event.source}_transformed",
            data=new_data,
            session_id=event.session_id,
            priority=event.priority,
            correlation_id=event.correlation_id,
            tags=event.tags + ["transformed"],
            metadata=event.metadata
        )
        
        return transformed_event
    
    async def _apply_flatmap_transformation(self, transformation: StreamTransformation, event: Event) -> List[Event]:
        """应用平铺映射转换."""
        # 简化实现：根据配置拆分事件
        split_field = transformation.parameters.get("split_field")
        if split_field and split_field in event.data:
            split_values = event.data[split_field]
            if isinstance(split_values, list):
                events = []
                for value in split_values:
                    new_data = event.data.copy()
                    new_data[split_field] = value
                    
                    new_event = Event(
                        event_type=event.event_type,
                        source=f"{event.source}_split",
                        data=new_data,
                        session_id=event.session_id,
                        priority=event.priority,
                        correlation_id=event.correlation_id,
                        tags=event.tags + ["split"],
                        metadata=event.metadata
                    )
                    events.append(new_event)
                
                return events
        
        return [event]
    
    async def _apply_reduce_transformation(self, transformation: StreamTransformation, event: Event) -> Event:
        """应用归约转换."""
        # 简化实现：聚合数值字段
        reduce_field = transformation.parameters.get("reduce_field")
        reduce_function = transformation.parameters.get("function", "sum")
        
        if reduce_field and reduce_field in event.data:
            # 这里应该维护状态进行归约，简化实现
            pass
        
        return event
    
    async def _update_stream_windows(self, stream_id: str, event: Event) -> None:
        """更新流窗口."""
        current_time = datetime.now()
        
        for window in self._windows.values():
            if not window.is_active:
                continue
            
            # 检查事件是否属于窗口
            if self._event_belongs_to_stream_window(event, window):
                window.events.append(event)
                
                # 更新窗口时间
                if window.window_type == "session":
                    window.end_time = current_time + (window.session_timeout or timedelta(minutes=30))
                elif window.window_type == "tumbling":
                    if not window.end_time:
                        window.end_time = window.start_time + window.size
                elif window.window_type == "sliding":
                    # 滑动窗口持续更新
                    window.end_time = current_time
                
                # 检查窗口是否应该触发
                if await self._should_trigger_window(window):
                    await self._trigger_window(window)
    
    def _event_belongs_to_stream_window(self, event: Event, window: StreamWindow) -> bool:
        """检查事件是否属于流窗口."""
        if window.window_type == "tumbling":
            return (window.start_time <= event.timestamp < 
                   window.start_time + window.size)
        elif window.window_type == "sliding":
            return (event.timestamp >= window.start_time and
                   event.timestamp <= window.start_time + window.size)
        elif window.window_type == "session":
            if not window.events:
                return True
            last_event_time = max(e.timestamp for e in window.events)
            return event.timestamp <= last_event_time + (window.session_timeout or timedelta(minutes=30))
        
        return False
    
    async def _should_trigger_window(self, window: StreamWindow) -> bool:
        """检查窗口是否应该触发."""
        current_time = datetime.now()
        
        if window.window_type == "tumbling":
            return window.end_time and current_time >= window.end_time
        elif window.window_type == "session":
            if window.events:
                last_event_time = max(e.timestamp for e in window.events)
                timeout = window.session_timeout or timedelta(minutes=30)
                return current_time >= last_event_time + timeout
        elif window.window_type == "sliding":
            # 滑动窗口定期触发
            return len(window.events) > 0
        
        return False
    
    async def _trigger_window(self, window: StreamWindow) -> None:
        """触发窗口处理."""
        logger.info(f"触发窗口: {window.window_id}, 事件数: {len(window.events)}")
        
        # 发布窗口完成事件
        await self.event_bus.create_and_publish(
            event_type=EventType.TASK_COMPLETED,  # 使用合适的事件类型
            source="stream_processor",
            data={
                "window_id": window.window_id,
                "window_type": window.window_type,
                "event_count": len(window.events),
                "start_time": window.start_time.isoformat(),
                "end_time": window.end_time.isoformat() if window.end_time else None,
                "events": [event.dict() for event in window.events]
            }
        )
        
        # 清理窗口（对于翻滚窗口）
        if window.window_type == "tumbling":
            window.events.clear()
            window.start_time = window.end_time
            window.end_time = None
    
    async def _execute_stream_aggregations(self, stream_id: str, event: Event) -> None:
        """执行流聚合."""
        for aggregator in self._aggregators.values():
            if not aggregator.enabled:
                continue
            
            try:
                await self._execute_aggregation(aggregator, event)
            except Exception as e:
                logger.error(f"聚合执行失败: {aggregator.name}, 错误: {e}")
    
    async def _execute_aggregation(self, aggregator: StreamAggregator, event: Event) -> None:
        """执行单个聚合."""
        # 简化的聚合实现
        if aggregator.aggregation_function == "count":
            # 计数聚合
            pass
        elif aggregator.aggregation_function == "sum":
            # 求和聚合
            pass
        elif aggregator.aggregation_function == "avg":
            # 平均值聚合
            pass
        # 其他聚合函数...
    
    async def _handle_backpressure(self, stream_id: str) -> None:
        """处理背压."""
        current_buffer_size = self._buffer_sizes.get(stream_id, 0)
        max_buffer_size = self._backpressure_config.max_buffer_size
        
        if current_buffer_size > max_buffer_size * self._backpressure_config.high_watermark:
            # 触发背压
            strategy = self._backpressure_config.backpressure_strategy
            
            if strategy == "drop_oldest":
                # 丢弃最旧的事件
                if stream_id in self._streams:
                    stream = self._streams[stream_id]
                    if not stream.events.empty():
                        try:
                            stream.events.get_nowait()
                            stream.metrics.events_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
            elif strategy == "drop_newest":
                # 丢弃最新的事件（不添加到队列）
                pass
            elif strategy == "block":
                # 阻塞处理
                await asyncio.sleep(0.1)
            
            # 应用限流
            if self._backpressure_config.throttle_rate:
                throttle_delay = 1.0 / self._backpressure_config.throttle_rate
                await asyncio.sleep(throttle_delay)
    
    async def _update_stream_metrics(self, stream_id: str) -> None:
        """更新流指标."""
        if stream_id not in self._streams:
            return
        
        stream = self._streams[stream_id]
        current_time = datetime.now()
        
        # 计算处理速率
        time_diff = (current_time - self._last_metrics_update).total_seconds()
        if time_diff > 0:
            events_in_period = stream.metrics.events_processed
            stream.metrics.processing_rate = events_in_period / time_diff
        
        # 计算缓冲区利用率
        buffer_size = stream.events.qsize()
        max_size = self._backpressure_config.max_buffer_size
        stream.metrics.buffer_utilization = buffer_size / max_size if max_size > 0 else 0
        
        # 更新时间
        stream.metrics.last_updated = current_time
        self._buffer_sizes[stream_id] = buffer_size
    
    # 公共接口方法
    async def add_transformation(self, stream_id: str, transformation: StreamTransformation) -> None:
        """添加流转换."""
        async with self._lock:
            self._transformations[transformation.transformation_id] = transformation
            logger.info(f"添加流转换: {transformation.name} 到流 {stream_id}")
    
    async def add_aggregator(self, stream_id: str, aggregator: StreamAggregator) -> None:
        """添加流聚合器."""
        async with self._lock:
            self._aggregators[aggregator.aggregator_id] = aggregator
            logger.info(f"添加流聚合器: {aggregator.name} 到流 {stream_id}")
    
    async def create_sliding_window(
        self, 
        stream_id: str, 
        window_size: timedelta, 
        slide_interval: timedelta
    ) -> str:
        """创建滑动窗口."""
        window = StreamWindow(
            window_type="sliding",
            size=window_size,
            slide=slide_interval
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.info(f"为流 {stream_id} 创建滑动窗口: {window.window_id}")
        return window.window_id
    
    async def create_tumbling_window(self, stream_id: str, window_size: timedelta) -> str:
        """创建翻滚窗口."""
        window = StreamWindow(
            window_type="tumbling",
            size=window_size
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.info(f"为流 {stream_id} 创建翻滚窗口: {window.window_id}")
        return window.window_id
    
    async def create_session_window(
        self, 
        stream_id: str, 
        session_timeout: timedelta
    ) -> str:
        """创建会话窗口."""
        window = StreamWindow(
            window_type="session",
            size=session_timeout,
            session_timeout=session_timeout
        )
        
        async with self._lock:
            self._windows[window.window_id] = window
        
        logger.info(f"为流 {stream_id} 创建会话窗口: {window.window_id}")
        return window.window_id
    
    async def get_stream_metrics(self, stream_id: str) -> Optional[StreamMetrics]:
        """获取流指标."""
        if stream_id in self._streams:
            return self._streams[stream_id].metrics
        return None
    
    async def get_all_stream_metrics(self) -> Dict[str, StreamMetrics]:
        """获取所有流指标."""
        return {
            stream_id: stream.metrics 
            for stream_id, stream in self._streams.items()
        }
    
    async def set_backpressure_config(self, config: BackpressureConfig) -> None:
        """设置背压配置."""
        async with self._lock:
            self._backpressure_config = config
            logger.info("更新背压配置")
    
    async def close_stream(self, stream_id: str) -> None:
        """关闭流."""
        async with self._lock:
            if stream_id in self._streams:
                stream = self._streams[stream_id]
                await stream.close()
                
                # 取消处理任务
                if stream_id in self._processing_tasks:
                    self._processing_tasks[stream_id].cancel()
                    del self._processing_tasks[stream_id]
                
                del self._streams[stream_id]
                logger.info(f"关闭流: {stream_id}")
    
    async def shutdown(self) -> None:
        """关闭流处理器."""
        logger.info("正在关闭事件流处理器...")
        
        self._shutdown_event.set()
        
        # 关闭所有流
        for stream_id in list(self._streams.keys()):
            await self.close_stream(stream_id)
        
        # 等待所有任务完成
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
        
        logger.info("事件流处理器已关闭")


# 全局事件流处理器实例
_global_stream_processor: Optional[EventStreamProcessor] = None


def get_stream_processor() -> EventStreamProcessor:
    """获取全局事件流处理器实例."""
    global _global_stream_processor
    if _global_stream_processor is None:
        _global_stream_processor = EventStreamProcessor()
    return _global_stream_processor