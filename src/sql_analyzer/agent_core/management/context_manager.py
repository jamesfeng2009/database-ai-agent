"""上下文管理器，负责维护对话上下文和状态."""

import logging
from typing import Any, Dict, List, Optional

from .event_system import publish_event
from .models import ConversationContext, EventType, Message, MessageRole
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器，负责维护和管理对话上下文."""
    
    def __init__(self, session_manager: SessionManager):
        """初始化上下文管理器.
        
        Args:
            session_manager: 会话管理器实例
        """
        self.session_manager = session_manager
        self._max_history_length = 50  # 最大对话历史长度
    
    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """添加消息到对话历史.
        
        Args:
            session_id: 会话ID
            role: 消息角色
            content: 消息内容
            metadata: 消息元数据
            
        Returns:
            是否添加成功
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            logger.warning(f"会话不存在: {session_id}")
            return False
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # 添加消息到历史
        context.conversation_history.append(message)
        
        # 限制历史长度
        if len(context.conversation_history) > self._max_history_length:
            context.conversation_history = context.conversation_history[-self._max_history_length:]
        
        # 更新会话
        await self.session_manager.update_session(
            session_id,
            conversation_history=context.conversation_history
        )
        
        # 发布消息事件
        event_type = EventType.USER_MESSAGE if role == MessageRole.USER else EventType.AGENT_RESPONSE
        await publish_event(
            event_type,
            source="context_manager",
            data={
                "message_id": message.id,
                "role": role.value,
                "content": content,
                "metadata": metadata or {}
            },
            session_id=session_id
        )
        
        logger.debug(f"添加消息到会话 {session_id}: {role.value}")
        return True
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """获取对话历史.
        
        Args:
            session_id: 会话ID
            limit: 返回的消息数量限制
            
        Returns:
            对话历史消息列表
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return []
        
        history = context.conversation_history
        if limit:
            history = history[-limit:]
        
        return history
    
    async def set_current_database(self, session_id: str, database: str) -> bool:
        """设置当前数据库.
        
        Args:
            session_id: 会话ID
            database: 数据库名称
            
        Returns:
            是否设置成功
        """
        success = await self.session_manager.update_session(
            session_id,
            current_database=database
        )
        
        if success:
            logger.info(f"会话 {session_id} 切换到数据库: {database}")
        
        return success
    
    async def get_current_database(self, session_id: str) -> Optional[str]:
        """获取当前数据库.
        
        Args:
            session_id: 会话ID
            
        Returns:
            当前数据库名称
        """
        context = await self.session_manager.get_session(session_id)
        return context.current_database if context else None
    
    async def add_active_task(self, session_id: str, task_id: str) -> bool:
        """添加活跃任务.
        
        Args:
            session_id: 会话ID
            task_id: 任务ID
            
        Returns:
            是否添加成功
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return False
        
        if task_id not in context.active_tasks:
            context.active_tasks.append(task_id)
            
            success = await self.session_manager.update_session(
                session_id,
                active_tasks=context.active_tasks
            )
            
            if success:
                logger.debug(f"添加活跃任务到会话 {session_id}: {task_id}")
            
            return success
        
        return True
    
    async def remove_active_task(self, session_id: str, task_id: str) -> bool:
        """移除活跃任务.
        
        Args:
            session_id: 会话ID
            task_id: 任务ID
            
        Returns:
            是否移除成功
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return False
        
        if task_id in context.active_tasks:
            context.active_tasks.remove(task_id)
            
            success = await self.session_manager.update_session(
                session_id,
                active_tasks=context.active_tasks
            )
            
            if success:
                logger.debug(f"移除活跃任务从会话 {session_id}: {task_id}")
            
            return success
        
        return True
    
    async def get_active_tasks(self, session_id: str) -> List[str]:
        """获取活跃任务列表.
        
        Args:
            session_id: 会话ID
            
        Returns:
            活跃任务ID列表
        """
        context = await self.session_manager.get_session(session_id)
        return context.active_tasks if context else []
    
    async def set_user_preference(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """设置用户偏好.
        
        Args:
            session_id: 会话ID
            key: 偏好键
            value: 偏好值
            
        Returns:
            是否设置成功
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return False
        
        context.user_preferences[key] = value
        
        success = await self.session_manager.update_session(
            session_id,
            user_preferences=context.user_preferences
        )
        
        if success:
            logger.debug(f"设置用户偏好 {session_id}: {key} = {value}")
        
        return success
    
    async def get_user_preference(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """获取用户偏好.
        
        Args:
            session_id: 会话ID
            key: 偏好键
            default: 默认值
            
        Returns:
            偏好值
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return default
        
        return context.user_preferences.get(key, default)
    
    async def set_context_variable(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """设置上下文变量.
        
        Args:
            session_id: 会话ID
            key: 变量键
            value: 变量值
            
        Returns:
            是否设置成功
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return False
        
        context.context_variables[key] = value
        
        success = await self.session_manager.update_session(
            session_id,
            context_variables=context.context_variables
        )
        
        if success:
            logger.debug(f"设置上下文变量 {session_id}: {key} = {value}")
        
        return success
    
    async def get_context_variable(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """获取上下文变量.
        
        Args:
            session_id: 会话ID
            key: 变量键
            default: 默认值
            
        Returns:
            变量值
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return default
        
        return context.context_variables.get(key, default)
    
    async def clear_context(self, session_id: str) -> bool:
        """清空会话上下文（保留基本信息）.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否清空成功
        """
        success = await self.session_manager.update_session(
            session_id,
            conversation_history=[],
            active_tasks=[],
            context_variables={}
        )
        
        if success:
            logger.info(f"清空会话上下文: {session_id}")
        
        return success
    
    async def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取上下文摘要信息.
        
        Args:
            session_id: 会话ID
            
        Returns:
            上下文摘要字典
        """
        context = await self.session_manager.get_session(session_id)
        if not context:
            return {}
        
        return {
            "session_id": context.session_id,
            "user_id": context.user_id,
            "current_database": context.current_database,
            "message_count": len(context.conversation_history),
            "active_task_count": len(context.active_tasks),
            "preference_count": len(context.user_preferences),
            "variable_count": len(context.context_variables),
            "created_at": context.created_at,
            "last_activity": context.last_activity,
            "state": context.state
        }