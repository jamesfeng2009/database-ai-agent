"""会话状态管理器."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import uuid4

from .event_system import publish_event
from .models import ConversationContext, EventType, SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """会话状态管理器，负责管理用户会话的生命周期."""
    
    def __init__(self, session_timeout_minutes: int = 30):
        """初始化会话管理器.
        
        Args:
            session_timeout_minutes: 会话超时时间（分钟）
        """
        self._sessions: Dict[str, ConversationContext] = {}
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # 启动清理任务
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """启动会话清理任务."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self) -> None:
        """清理过期会话的后台任务."""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                await self._remove_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"会话清理任务异常: {e}")
    
    async def _remove_expired_sessions(self) -> None:
        """移除过期的会话."""
        now = datetime.now()
        expired_sessions = []
        
        async with self._lock:
            for session_id, context in self._sessions.items():
                if now - context.last_activity > self._session_timeout:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.expire_session(session_id)
    
    async def create_session(self, user_id: str, session_id: Optional[str] = None) -> ConversationContext:
        """创建新的会话.
        
        Args:
            user_id: 用户ID
            session_id: 可选的会话ID，如果不提供则自动生成
            
        Returns:
            创建的对话上下文
        """
        if session_id is None:
            session_id = str(uuid4())
        
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            state=SessionState.ACTIVE
        )
        
        async with self._lock:
            self._sessions[session_id] = context
        
        # 发布会话创建事件
        await publish_event(
            EventType.SESSION_CREATED,
            source="session_manager",
            data={"session_id": session_id, "user_id": user_id},
            session_id=session_id
        )
        
        logger.info(f"创建新会话: {session_id}, 用户: {user_id}")
        return context
    
    async def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """获取会话上下文.
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话上下文，如果不存在则返回None
        """
        async with self._lock:
            context = self._sessions.get(session_id)
            if context and context.state == SessionState.ACTIVE:
                # 更新最后活动时间
                context.last_activity = datetime.now()
                return context
            return None
    
    async def update_session(self, session_id: str, **updates) -> bool:
        """更新会话上下文.
        
        Args:
            session_id: 会话ID
            **updates: 要更新的字段
            
        Returns:
            是否更新成功
        """
        async with self._lock:
            context = self._sessions.get(session_id)
            if context and context.state == SessionState.ACTIVE:
                # 更新字段
                for key, value in updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                
                # 更新最后活动时间
                context.last_activity = datetime.now()
                
                # 发布上下文更新事件
                await publish_event(
                    EventType.CONTEXT_UPDATED,
                    source="session_manager",
                    data={"session_id": session_id, "updates": updates},
                    session_id=session_id
                )
                
                return True
            return False
    
    async def expire_session(self, session_id: str) -> bool:
        """使会话过期.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功过期
        """
        async with self._lock:
            context = self._sessions.get(session_id)
            if context:
                context.state = SessionState.EXPIRED
                
                # 发布会话过期事件
                await publish_event(
                    EventType.SESSION_EXPIRED,
                    source="session_manager",
                    data={"session_id": session_id},
                    session_id=session_id
                )
                
                logger.info(f"会话已过期: {session_id}")
                return True
            return False
    
    async def terminate_session(self, session_id: str) -> bool:
        """终止会话.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功终止
        """
        async with self._lock:
            context = self._sessions.get(session_id)
            if context:
                context.state = SessionState.TERMINATED
                
                logger.info(f"会话已终止: {session_id}")
                return True
            return False
    
    async def cleanup_session(self, session_id: str) -> bool:
        """清理会话（从内存中移除）.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功清理
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"会话已清理: {session_id}")
                return True
            return False
    
    def get_active_session_count(self) -> int:
        """获取活跃会话数量.
        
        Returns:
            活跃会话数量
        """
        return len([s for s in self._sessions.values() if s.state == SessionState.ACTIVE])
    
    def get_all_sessions(self) -> Dict[str, ConversationContext]:
        """获取所有会话（仅用于调试）.
        
        Returns:
            所有会话的字典
        """
        return self._sessions.copy()
    
    async def shutdown(self) -> None:
        """关闭会话管理器."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 终止所有活跃会话
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.terminate_session(session_id)
        
        logger.info("会话管理器已关闭")