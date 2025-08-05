"""对话管理器，负责处理用户与AI Agent的对话交互."""

import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from .context_manager import ContextManager
from .event_system import publish_event
from .models import (
    AgentResponse,
    EventType,
    IntentType,
    MessageRole,
    UserIntent,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """对话状态枚举."""
    GREETING = "greeting"
    QUERY_INPUT = "query_input"
    ANALYSIS_PENDING = "analysis_pending"
    ANALYSIS_COMPLETE = "analysis_complete"
    OPTIMIZATION_DISCUSSION = "optimization_discussion"
    MONITORING_SETUP = "monitoring_setup"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    TASK_EXECUTION = "task_execution"
    CONFIRMATION_PENDING = "confirmation_pending"
    ERROR_HANDLING = "error_handling"
    IDLE = "idle"


class ResponseFormat(str, Enum):
    """响应格式枚举."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"
    CODE_BLOCK = "code_block"
    TABLE = "table"


class ConversationFlow:
    """对话流程状态机."""
    
    def __init__(self):
        """初始化对话流程状态机."""
        # 定义状态转换规则
        self.transitions = {
            ConversationState.GREETING: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
                IntentType.KNOWLEDGE_QUERY: ConversationState.KNOWLEDGE_SHARING,
                IntentType.HELP_REQUEST: ConversationState.KNOWLEDGE_SHARING,
            },
            ConversationState.QUERY_INPUT: {
                IntentType.QUERY_ANALYSIS: ConversationState.ANALYSIS_PENDING,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.HELP_REQUEST: ConversationState.KNOWLEDGE_SHARING,
            },
            ConversationState.ANALYSIS_PENDING: {
                IntentType.QUERY_ANALYSIS: ConversationState.ANALYSIS_COMPLETE,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
            },
            ConversationState.ANALYSIS_COMPLETE: {
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
            },
            ConversationState.OPTIMIZATION_DISCUSSION: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.TASK_EXECUTION,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
            },
            ConversationState.MONITORING_SETUP: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.MONITORING_SETUP: ConversationState.TASK_EXECUTION,
            },
            ConversationState.KNOWLEDGE_SHARING: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
                IntentType.KNOWLEDGE_QUERY: ConversationState.KNOWLEDGE_SHARING,
            },
            ConversationState.TASK_EXECUTION: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.CONFIRMATION_PENDING,
                IntentType.MONITORING_SETUP: ConversationState.CONFIRMATION_PENDING,
            },
            ConversationState.CONFIRMATION_PENDING: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.TASK_EXECUTION,
                IntentType.MONITORING_SETUP: ConversationState.TASK_EXECUTION,
            },
            ConversationState.ERROR_HANDLING: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
                IntentType.HELP_REQUEST: ConversationState.KNOWLEDGE_SHARING,
            },
            ConversationState.IDLE: {
                IntentType.QUERY_ANALYSIS: ConversationState.QUERY_INPUT,
                IntentType.OPTIMIZATION_REQUEST: ConversationState.OPTIMIZATION_DISCUSSION,
                IntentType.MONITORING_SETUP: ConversationState.MONITORING_SETUP,
                IntentType.KNOWLEDGE_QUERY: ConversationState.KNOWLEDGE_SHARING,
                IntentType.HELP_REQUEST: ConversationState.KNOWLEDGE_SHARING,
            }
        }
    
    def get_next_state(
        self, 
        current_state: ConversationState, 
        intent: IntentType
    ) -> ConversationState:
        """根据当前状态和意图获取下一个状态.
        
        Args:
            current_state: 当前对话状态
            intent: 用户意图
            
        Returns:
            下一个对话状态
        """
        if current_state in self.transitions:
            return self.transitions[current_state].get(intent, ConversationState.IDLE)
        return ConversationState.IDLE
    
    def get_valid_intents(self, current_state: ConversationState) -> Set[IntentType]:
        """获取当前状态下有效的意图类型.
        
        Args:
            current_state: 当前对话状态
            
        Returns:
            有效意图类型集合
        """
        if current_state in self.transitions:
            return set(self.transitions[current_state].keys())
        return set()


class ResponseFormatter:
    """响应格式化器."""
    
    @staticmethod
    def format_plain_text(content: str) -> str:
        """格式化纯文本响应."""
        return content.strip()
    
    @staticmethod
    def format_markdown(content: str, title: Optional[str] = None) -> str:
        """格式化Markdown响应."""
        if title:
            return f"## {title}\n\n{content}"
        return content
    
    @staticmethod
    def format_code_block(code: str, language: str = "sql") -> str:
        """格式化代码块响应."""
        return f"```{language}\n{code}\n```"
    
    @staticmethod
    def format_structured_response(
        title: str,
        sections: Dict[str, str],
        suggestions: Optional[List[str]] = None
    ) -> str:
        """格式化结构化响应."""
        response = f"## {title}\n\n"
        
        for section_title, section_content in sections.items():
            response += f"### {section_title}\n{section_content}\n\n"
        
        if suggestions:
            response += "### 建议的后续操作\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
        
        return response.strip()
    
    @staticmethod
    def format_table(headers: List[str], rows: List[List[str]]) -> str:
        """格式化表格响应."""
        if not headers or not rows:
            return ""
        
        # 计算列宽
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # 构建表格
        table = "| " + " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers)) + " |\n"
        table += "| " + " | ".join("-" * width for width in col_widths) + " |\n"
        
        for row in rows:
            table += "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |\n"
        
        return table


class SessionPersistence:
    """会话持久化管理器."""
    
    def __init__(self, storage_path: str = ".agent_sessions"):
        """初始化会话持久化管理器.
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def save_session(self, session_id: str, context_data: Dict[str, Any]) -> bool:
        """保存会话数据.
        
        Args:
            session_id: 会话ID
            context_data: 上下文数据
            
        Returns:
            是否保存成功
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            
            # 准备序列化数据
            serializable_data = {
                "session_id": session_id,
                "saved_at": datetime.now().isoformat(),
                "context": context_data
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"会话已保存: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存会话失败 {session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载会话数据.
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据，如果不存在则返回None
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"会话已加载: {session_id}")
            return data.get("context")
            
        except Exception as e:
            logger.error(f"加载会话失败 {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话数据.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        try:
            session_file = self.storage_path / f"{session_id}.json"
            
            if session_file.exists():
                session_file.unlink()
                logger.debug(f"会话已删除: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除会话失败 {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[str]:
        """列出所有保存的会话.
        
        Returns:
            会话ID列表
        """
        try:
            session_files = self.storage_path.glob("*.json")
            return [f.stem for f in session_files]
        except Exception as e:
            logger.error(f"列出会话失败: {e}")
            return []


class ConversationManager:
    """对话管理器，管理用户与AI Agent的对话交互."""
    
    def __init__(
        self, 
        session_manager: SessionManager, 
        context_manager: ContextManager,
        enable_persistence: bool = True,
        storage_path: str = ".agent_sessions"
    ):
        """初始化对话管理器.
        
        Args:
            session_manager: 会话管理器
            context_manager: 上下文管理器
            enable_persistence: 是否启用会话持久化
            storage_path: 持久化存储路径
        """
        self.session_manager = session_manager
        self.context_manager = context_manager
        self.conversation_flow = ConversationFlow()
        self.response_formatter = ResponseFormatter()
        
        # SQL分析集成器（稍后设置）
        self.sql_integrator = None
        
        # 会话持久化
        self.enable_persistence = enable_persistence
        if enable_persistence:
            self.persistence = SessionPersistence(storage_path)
        else:
            self.persistence = None
        
        # 意图识别的关键词映射
        self._intent_keywords = {
            IntentType.QUERY_ANALYSIS: [
                "分析", "analyze", "查询", "query", "性能", "performance", 
                "慢", "slow", "优化", "optimize", "explain", "执行计划", "execution plan"
            ],
            IntentType.OPTIMIZATION_REQUEST: [
                "优化", "optimize", "改进", "improve", "建议", "suggest",
                "索引", "index", "重写", "rewrite", "调优", "tuning"
            ],
            IntentType.MONITORING_SETUP: [
                "监控", "monitor", "告警", "alert", "通知", "notify",
                "设置", "setup", "配置", "config", "阈值", "threshold"
            ],
            IntentType.KNOWLEDGE_QUERY: [
                "什么是", "what is", "如何", "how to", "为什么", "why",
                "解释", "explain", "帮助", "help", "概念", "concept"
            ],
            IntentType.HELP_REQUEST: [
                "帮助", "help", "指导", "guide", "教程", "tutorial",
                "文档", "doc", "使用", "usage", "功能", "feature"
            ]
        }
        
        # 上下文关键词，用于增强意图识别
        self._context_keywords = {
            "sql_related": ["select", "insert", "update", "delete", "join", "where", "group by", "order by"],
            "performance_related": ["慢", "slow", "快", "fast", "延迟", "latency", "响应时间", "response time"],
            "database_objects": ["表", "table", "索引", "index", "视图", "view", "存储过程", "procedure"]
        }
    
    async def process_user_input(self, user_input: str, session_id: str) -> AgentResponse:
        """处理用户输入并生成响应.
        
        Args:
            user_input: 用户输入的文本
            session_id: 会话ID
            
        Returns:
            Agent响应
        """
        try:
            # 获取当前对话状态
            current_state = await self._get_conversation_state(session_id)
            
            # 添加用户消息到上下文
            await self.context_manager.add_message(
                session_id, MessageRole.USER, user_input
            )
            
            # 识别用户意图（考虑上下文）
            intent = await self._extract_intent(user_input, session_id, current_state)
            
            # 状态转换
            next_state = self.conversation_flow.get_next_state(current_state, intent.intent_type)
            await self._set_conversation_state(session_id, next_state)
            
            # 生成响应
            response = await self._generate_response(intent, session_id, next_state)
            
            # 添加Agent响应到上下文
            await self.context_manager.add_message(
                session_id, MessageRole.ASSISTANT, response.content
            )
            
            # 保存会话状态（如果启用持久化）
            if self.enable_persistence:
                await self._save_session_state(session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"处理用户输入失败: {e}")
            
            # 设置错误处理状态
            await self._set_conversation_state(session_id, ConversationState.ERROR_HANDLING)
            
            # 发布错误事件
            await publish_event(
                EventType.ERROR_OCCURRED,
                source="conversation_manager",
                data={"error": str(e), "user_input": user_input},
                session_id=session_id
            )
            
            # 返回错误响应
            return AgentResponse(
                content=f"抱歉，处理您的请求时出现了错误：{str(e)}",
                intent_handled=IntentType.UNKNOWN,
                suggested_actions=["请重新尝试您的请求", "检查输入格式是否正确", "输入'帮助'获取使用指导"]
            )
    
    async def _extract_intent(
        self, 
        user_input: str, 
        session_id: str, 
        current_state: ConversationState
    ) -> UserIntent:
        """从用户输入中提取意图（考虑上下文）.
        
        Args:
            user_input: 用户输入文本
            session_id: 会话ID
            current_state: 当前对话状态
            
        Returns:
            用户意图对象
        """
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # 获取对话历史用于上下文分析
        history = await self.context_manager.get_conversation_history(session_id, limit=3)
        
        # 基于关键词的意图识别
        for intent_type, keywords in self._intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in user_input_lower:
                    score += 1
            
            if score > 0:
                intent_scores[intent_type] = score
        
        # 上下文增强：根据对话历史调整意图得分
        if history:
            last_messages = [msg.content.lower() for msg in history[-2:]]
            context_text = " ".join(last_messages)
            
            # 如果最近讨论了SQL相关内容，增加查询分析意图的权重
            if any(keyword in context_text for keyword in self._context_keywords["sql_related"]):
                intent_scores[IntentType.QUERY_ANALYSIS] = intent_scores.get(IntentType.QUERY_ANALYSIS, 0) + 0.5
            
            # 如果最近讨论了性能问题，增加优化请求意图的权重
            if any(keyword in context_text for keyword in self._context_keywords["performance_related"]):
                intent_scores[IntentType.OPTIMIZATION_REQUEST] = intent_scores.get(IntentType.OPTIMIZATION_REQUEST, 0) + 0.5
        
        # 状态上下文增强：根据当前状态调整意图识别
        valid_intents = self.conversation_flow.get_valid_intents(current_state)
        for intent_type in intent_scores:
            if intent_type in valid_intents:
                intent_scores[intent_type] += 0.3  # 增加有效意图的权重
        
        # 特殊处理：简单的确认/否定回答
        confirmation_patterns = ["是", "yes", "好的", "ok", "确认", "confirm", "同意", "agree"]
        negation_patterns = ["不", "no", "否", "取消", "cancel", "不要", "don't"]
        
        if any(pattern in user_input_lower for pattern in confirmation_patterns):
            if current_state == ConversationState.CONFIRMATION_PENDING:
                intent_scores[IntentType.OPTIMIZATION_REQUEST] = intent_scores.get(IntentType.OPTIMIZATION_REQUEST, 0) + 2
        elif any(pattern in user_input_lower for pattern in negation_patterns):
            if current_state == ConversationState.CONFIRMATION_PENDING:
                intent_scores[IntentType.HELP_REQUEST] = intent_scores.get(IntentType.HELP_REQUEST, 0) + 2
        
        # 选择得分最高的意图
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            max_possible_score = max(len(self._intent_keywords[best_intent]), 3)  # 考虑上下文加分
            confidence = min(intent_scores[best_intent] / max_possible_score, 1.0)
        else:
            best_intent = IntentType.UNKNOWN
            confidence = 0.0
        
        # 提取实体
        entities = await self._extract_entities(user_input)
        
        # 添加上下文参数
        parameters = {
            "current_state": current_state.value,
            "has_context": len(history) > 0,
            "context_keywords": self._identify_context_keywords(user_input_lower)
        }
        
        return UserIntent(
            intent_type=best_intent,
            entities=entities,
            confidence=confidence,
            parameters=parameters,
            raw_input=user_input
        )
    
    async def _extract_entities(self, user_input: str) -> dict:
        """从用户输入中提取实体.
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            提取的实体字典
        """
        entities = {}
        
        # 提取SQL语句
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'(SELECT\s+.*?(?:FROM|;))',
            r'(UPDATE\s+.*?(?:SET|;))',
            r'(INSERT\s+.*?(?:VALUES|;))',
            r'(DELETE\s+.*?(?:FROM|;))'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE | re.DOTALL)
            if matches:
                entities['sql_statement'] = matches[0].strip()
                break
        
        # 提取表名
        table_pattern = r'表\s*[`"]?(\w+)[`"]?|table\s+[`"]?(\w+)[`"]?'
        table_matches = re.findall(table_pattern, user_input, re.IGNORECASE)
        if table_matches:
            entities['table_name'] = [match[0] or match[1] for match in table_matches]
        
        # 提取数据库名
        db_pattern = r'数据库\s*[`"]?(\w+)[`"]?|database\s+[`"]?(\w+)[`"]?'
        db_matches = re.findall(db_pattern, user_input, re.IGNORECASE)
        if db_matches:
            entities['database_name'] = [match[0] or match[1] for match in db_matches]
        
        return entities
    
    def _identify_context_keywords(self, text: str) -> List[str]:
        """识别文本中的上下文关键词.
        
        Args:
            text: 输入文本
            
        Returns:
            识别到的上下文关键词列表
        """
        found_keywords = []
        for category, keywords in self._context_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    found_keywords.append(f"{category}:{keyword}")
        return found_keywords
    
    async def _get_conversation_state(self, session_id: str) -> ConversationState:
        """获取当前对话状态.
        
        Args:
            session_id: 会话ID
            
        Returns:
            当前对话状态
        """
        state_str = await self.context_manager.get_context_variable(
            session_id, "conversation_state", ConversationState.GREETING.value
        )
        try:
            return ConversationState(state_str)
        except ValueError:
            return ConversationState.GREETING
    
    async def _set_conversation_state(self, session_id: str, state: ConversationState) -> bool:
        """设置对话状态.
        
        Args:
            session_id: 会话ID
            state: 新的对话状态
            
        Returns:
            是否设置成功
        """
        success = await self.context_manager.set_context_variable(
            session_id, "conversation_state", state.value
        )
        
        if success:
            # 发布状态变更事件
            await publish_event(
                EventType.CONTEXT_UPDATED,
                source="conversation_manager",
                data={"conversation_state": state.value},
                session_id=session_id
            )
            logger.debug(f"对话状态已更新 {session_id}: {state.value}")
        
        return success
    
    async def _save_session_state(self, session_id: str) -> bool:
        """保存会话状态到持久化存储.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否保存成功
        """
        if not self.persistence:
            return False
        
        try:
            # 获取会话上下文
            context = await self.session_manager.get_session(session_id)
            if not context:
                return False
            
            # 准备持久化数据
            context_data = {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "current_database": context.current_database,
                "conversation_history": [
                    {
                        "id": msg.id,
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in context.conversation_history
                ],
                "active_tasks": context.active_tasks,
                "user_preferences": context.user_preferences,
                "context_variables": context.context_variables,
                "created_at": context.created_at.isoformat(),
                "last_activity": context.last_activity.isoformat(),
                "state": context.state.value
            }
            
            return await self.persistence.save_session(session_id, context_data)
            
        except Exception as e:
            logger.error(f"保存会话状态失败 {session_id}: {e}")
            return False
    
    async def restore_session(self, session_id: str) -> bool:
        """从持久化存储恢复会话.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否恢复成功
        """
        if not self.persistence:
            return False
        
        try:
            # 加载会话数据
            context_data = await self.persistence.load_session(session_id)
            if not context_data:
                return False
            
            # 重建会话上下文
            from .models import Message, SessionState
            
            # 恢复消息历史
            conversation_history = []
            for msg_data in context_data.get("conversation_history", []):
                message = Message(
                    id=msg_data["id"],
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data.get("metadata", {})
                )
                conversation_history.append(message)
            
            # 创建或更新会话
            context = await self.session_manager.get_session(session_id)
            if not context:
                context = await self.session_manager.create_session(
                    context_data["user_id"], session_id
                )
            
            # 更新会话数据
            await self.session_manager.update_session(
                session_id,
                current_database=context_data.get("current_database"),
                conversation_history=conversation_history,
                active_tasks=context_data.get("active_tasks", []),
                user_preferences=context_data.get("user_preferences", {}),
                context_variables=context_data.get("context_variables", {}),
                state=SessionState(context_data.get("state", "active"))
            )
            
            logger.info(f"会话已恢复: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复会话失败 {session_id}: {e}")
            return False
    
    async def _generate_response(
        self, 
        intent: UserIntent, 
        session_id: str, 
        conversation_state: ConversationState
    ) -> AgentResponse:
        """根据用户意图和对话状态生成响应.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 获取对话历史用于上下文
        history = await self.context_manager.get_conversation_history(session_id, limit=5)
        current_db = await self.context_manager.get_current_database(session_id)
        
        # 根据意图类型和对话状态生成不同的响应
        if intent.intent_type == IntentType.QUERY_ANALYSIS:
            return await self._handle_query_analysis_intent(intent, session_id, current_db, conversation_state)
        elif intent.intent_type == IntentType.OPTIMIZATION_REQUEST:
            return await self._handle_optimization_intent(intent, session_id, conversation_state)
        elif intent.intent_type == IntentType.MONITORING_SETUP:
            return await self._handle_monitoring_intent(intent, session_id, conversation_state)
        elif intent.intent_type == IntentType.KNOWLEDGE_QUERY:
            return await self._handle_knowledge_intent(intent, session_id, conversation_state)
        elif intent.intent_type == IntentType.HELP_REQUEST:
            return await self._handle_help_intent(intent, session_id, conversation_state)
        else:
            return await self._handle_unknown_intent(intent, session_id, conversation_state)
    
    async def _handle_query_analysis_intent(
        self, 
        intent: UserIntent, 
        session_id: str, 
        current_db: Optional[str],
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理查询分析意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            current_db: 当前数据库
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 使用SQL集成器处理分析请求
        if hasattr(self, 'sql_integrator') and self.sql_integrator:
            return await self.sql_integrator.process_sql_analysis_intent(intent, session_id)
        
        # 如果没有SQL集成器，使用原有逻辑
        sql_statement = intent.entities.get('sql_statement')
        
        if sql_statement:
            # 使用结构化响应格式
            sections = {
                "SQL查询": self.response_formatter.format_code_block(sql_statement),
                "分析状态": "正在执行EXPLAIN分析和性能评估...",
            }
            
            if current_db:
                sections["当前数据库"] = current_db
            
            # 根据对话状态调整响应内容
            if conversation_state == ConversationState.ANALYSIS_PENDING:
                sections["下一步"] = "分析完成后，我将为您提供详细的性能报告和优化建议。"
            
            content = self.response_formatter.format_structured_response(
                "SQL查询性能分析",
                sections
            )
            
            suggested_actions = [
                "等待分析完成",
                "查看详细的执行计划",
                "获取优化建议",
                "设置性能监控"
            ]
            
            # 添加分析任务到活跃任务列表
            await self.context_manager.add_active_task(session_id, f"sql_analysis_{sql_statement[:20]}")
            
        else:
            # 根据对话状态提供不同的提示
            if conversation_state == ConversationState.GREETING:
                title = "欢迎使用SQL性能分析"
                intro = "我可以帮助您分析SQL查询的性能问题。"
            else:
                title = "SQL查询分析"
                intro = "请提供您想要分析的SQL查询语句。"
            
            sections = {
                "使用方法": "请将您的SQL查询语句放在代码块中：",
                "示例格式": self.response_formatter.format_code_block(
                    "SELECT * FROM your_table WHERE condition;"
                ),
                "支持的分析": "• 执行计划分析\n• 性能瓶颈识别\n• 索引使用评估\n• 优化建议生成"
            }
            
            content = f"{intro}\n\n" + self.response_formatter.format_structured_response(
                title, sections
            )
            
            suggested_actions = [
                "提供SQL查询语句",
                "指定要分析的数据库",
                "查看分析示例",
                "了解分析功能"
            ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            requires_followup=sql_statement is not None,
            metadata={
                "has_sql": sql_statement is not None,
                "conversation_state": conversation_state.value,
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    
    async def _handle_optimization_intent(
        self, 
        intent: UserIntent, 
        session_id: str,
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理优化请求意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 如果是确认状态，使用SQL集成器处理确认
        if conversation_state == ConversationState.CONFIRMATION_PENDING:
            if hasattr(self, 'sql_integrator') and self.sql_integrator:
                return await self.sql_integrator.handle_optimization_confirmation(intent, session_id)
        
        sql_statement = intent.entities.get('sql_statement')
        
        if conversation_state == ConversationState.CONFIRMATION_PENDING:
            # 处理确认状态下的优化请求
            content = self.response_formatter.format_structured_response(
                "优化操作确认",
                {
                    "状态": "准备执行优化操作",
                    "安全提醒": "所有优化操作都将在安全模式下执行，并可以回滚",
                    "执行计划": "• 创建备份点\n• 执行优化操作\n• 验证优化效果\n• 生成优化报告"
                }
            )
            
            suggested_actions = [
                "确认执行优化",
                "查看详细计划",
                "取消操作",
                "修改优化方案"
            ]
            
        elif sql_statement:
            # 针对特定SQL的优化建议
            sections = {
                "目标查询": self.response_formatter.format_code_block(sql_statement),
                "优化分析": "正在分析查询结构和性能特征...",
                "优化方向": "• 索引优化\n• 查询重写\n• 执行计划优化\n• 统计信息更新"
            }
            
            content = self.response_formatter.format_structured_response(
                "SQL查询优化分析",
                sections
            )
            
            suggested_actions = [
                "查看具体优化建议",
                "执行推荐的优化",
                "比较优化前后性能",
                "设置性能监控"
            ]
            
        else:
            # 通用优化建议
            optimization_types = {
                "索引优化": "• 创建缺失索引\n• 删除冗余索引\n• 优化复合索引顺序",
                "查询优化": "• 查询重写建议\n• 子查询优化\n• JOIN顺序调整",
                "表结构优化": "• 数据类型优化\n• 分区策略\n• 表设计建议",
                "配置调优": "• 内存参数调整\n• 缓存配置优化\n• 连接池设置"
            }
            
            content = self.response_formatter.format_structured_response(
                "数据库优化服务",
                optimization_types,
                ["提供需要优化的SQL查询", "选择优化类型", "查看优化案例"]
            )
            
            suggested_actions = [
                "提供需要优化的SQL查询",
                "询问索引优化建议",
                "查看配置优化选项",
                "了解优化最佳实践"
            ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            requires_followup=conversation_state == ConversationState.CONFIRMATION_PENDING,
            metadata={
                "conversation_state": conversation_state.value,
                "has_sql": sql_statement is not None,
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    
    async def _handle_monitoring_intent(
        self, 
        intent: UserIntent, 
        session_id: str,
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理监控设置意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        current_db = await self.context_manager.get_current_database(session_id)
        
        if conversation_state == ConversationState.TASK_EXECUTION:
            # 监控设置执行状态
            sections = {
                "执行状态": "正在配置数据库性能监控...",
                "当前步骤": "• 验证数据库连接\n• 设置监控指标\n• 配置告警规则\n• 启动监控服务",
                "预计完成时间": "约2-3分钟"
            }
            
            if current_db:
                sections["目标数据库"] = current_db
            
            content = self.response_formatter.format_structured_response(
                "监控设置进行中",
                sections
            )
            
            suggested_actions = [
                "查看设置进度",
                "修改监控配置",
                "暂停设置过程"
            ]
            
        else:
            # 监控设置选项
            monitoring_options = {
                "慢查询监控": "• 自动识别慢查询\n• 设置执行时间阈值\n• 生成慢查询报告",
                "性能指标告警": "• CPU使用率监控\n• 内存使用监控\n• 磁盘I/O监控\n• 连接数监控",
                "异常检测": "• 性能异常自动识别\n• 智能告警过滤\n• 异常模式分析",
                "自动化报告": "• 定期性能报告\n• 趋势分析报告\n• 优化建议报告"
            }
            
            intro = "我可以帮您设置全面的数据库性能监控系统："
            if current_db:
                intro += f"\n\n**当前数据库**: {current_db}"
            
            content = intro + "\n\n" + self.response_formatter.format_structured_response(
                "数据库监控服务",
                monitoring_options
            )
            
            suggested_actions = [
                "设置慢查询告警",
                "配置性能指标监控",
                "启用异常检测",
                "查看监控模板"
            ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            requires_followup=conversation_state == ConversationState.TASK_EXECUTION,
            metadata={
                "conversation_state": conversation_state.value,
                "current_database": current_db,
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    
    async def _handle_knowledge_intent(
        self, 
        intent: UserIntent, 
        session_id: str,
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理知识查询意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 分析用户询问的具体知识点
        user_input = intent.raw_input.lower()
        knowledge_topics = {
            "索引": {
                "title": "数据库索引详解",
                "content": {
                    "基本概念": "索引是数据库中用于快速定位数据的数据结构，类似于书籍的目录。",
                    "索引类型": "• B-Tree索引（最常用）\n• 哈希索引\n• 位图索引\n• 全文索引",
                    "使用场景": "• WHERE子句中的条件列\n• JOIN操作的连接列\n• ORDER BY的排序列",
                    "注意事项": "• 索引会占用存储空间\n• 影响INSERT/UPDATE/DELETE性能\n• 需要定期维护"
                }
            },
            "执行计划": {
                "title": "SQL执行计划分析",
                "content": {
                    "什么是执行计划": "执行计划是数据库优化器为SQL查询选择的具体执行路径。",
                    "如何查看": "• MySQL: EXPLAIN SELECT ...\n• PostgreSQL: EXPLAIN ANALYZE SELECT ...\n• SQL Server: SET SHOWPLAN_ALL ON",
                    "关键指标": "• 扫描行数\n• 执行时间\n• 索引使用情况\n• JOIN算法",
                    "优化要点": "• 避免全表扫描\n• 合理使用索引\n• 优化JOIN顺序"
                }
            },
            "性能": {
                "title": "数据库性能优化策略",
                "content": {
                    "查询优化": "• 避免SELECT *\n• 使用合适的WHERE条件\n• 优化子查询",
                    "索引策略": "• 为常用查询创建索引\n• 避免过多索引\n• 定期重建索引",
                    "表设计": "• 选择合适的数据类型\n• 规范化与反规范化平衡\n• 分区策略",
                    "系统配置": "• 内存分配优化\n• 连接池配置\n• 缓存策略"
                }
            }
        }
        
        # 根据用户输入匹配知识主题
        matched_topic = None
        for topic_key, topic_data in knowledge_topics.items():
            if topic_key in user_input or any(keyword in user_input for keyword in [topic_key]):
                matched_topic = topic_data
                break
        
        if matched_topic:
            # 提供具体知识点的详细解释
            content = self.response_formatter.format_structured_response(
                matched_topic["title"],
                matched_topic["content"]
            )
            
            suggested_actions = [
                "查看相关示例",
                "了解最佳实践",
                "获取实际操作指导",
                "询问其他概念"
            ]
        else:
            # 提供知识库概览
            knowledge_categories = {
                "基础概念": "• 数据库索引原理\n• 查询执行计划\n• 事务和锁机制\n• 数据类型选择",
                "性能优化": "• SQL查询优化\n• 索引设计策略\n• 表结构优化\n• 系统参数调优",
                "最佳实践": "• 数据库设计原则\n• 安全配置指南\n• 备份恢复策略\n• 监控告警设置",
                "故障排查": "• 慢查询分析\n• 死锁问题处理\n• 性能瓶颈定位\n• 容量规划"
            }
            
            content = self.response_formatter.format_structured_response(
                "数据库知识库",
                knowledge_categories,
                ["询问具体概念", "查看实际案例", "获取操作指导"]
            )
            
            suggested_actions = [
                "询问索引相关问题",
                "了解执行计划分析",
                "查看性能优化策略",
                "学习最佳实践"
            ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            metadata={
                "conversation_state": conversation_state.value,
                "knowledge_topic": matched_topic["title"] if matched_topic else "general",
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    
    async def _handle_help_intent(
        self, 
        intent: UserIntent, 
        session_id: str,
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理帮助请求意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 根据对话状态提供上下文相关的帮助
        if conversation_state == ConversationState.ANALYSIS_PENDING:
            help_content = {
                "当前状态": "SQL查询分析进行中",
                "可用操作": "• 等待分析完成\n• 查看分析进度\n• 取消当前分析\n• 提交新的查询",
                "下一步建议": "分析完成后，您可以查看详细报告并获取优化建议。"
            }
            title = "分析状态帮助"
        elif conversation_state == ConversationState.CONFIRMATION_PENDING:
            help_content = {
                "当前状态": "等待操作确认",
                "确认方式": "• 输入'是'或'确认'来执行操作\n• 输入'否'或'取消'来取消操作",
                "安全提醒": "所有操作都有安全保护和回滚机制。"
            }
            title = "确认操作帮助"
        else:
            # 通用帮助信息
            help_content = {
                "🔍 查询分析": "• 分析SQL查询性能\n• 解读执行计划\n• 识别性能瓶颈\n• 生成优化报告",
                "⚡ 优化建议": "• 智能索引建议\n• 查询重写建议\n• 配置参数调优\n• 表结构优化",
                "📊 监控设置": "• 实时性能监控\n• 智能告警配置\n• 异常自动检测\n• 趋势分析报告",
                "💡 知识问答": "• 数据库概念解释\n• 最佳实践分享\n• 技术问题解答\n• 案例分析指导",
                "🛠️ 使用方法": "• 直接提问或描述问题\n• 提供SQL查询进行分析\n• 使用自然语言交互\n• 支持多轮对话"
            }
            title = "数据库性能优化AI Agent使用指南"
        
        content = self.response_formatter.format_structured_response(title, help_content)
        
        # 根据状态提供不同的建议操作
        if conversation_state == ConversationState.GREETING:
            suggested_actions = [
                "分析SQL查询性能",
                "获取数据库优化建议",
                "设置性能监控",
                "学习数据库知识"
            ]
        else:
            suggested_actions = [
                "继续当前操作",
                "开始新的分析",
                "查看功能介绍",
                "询问具体问题"
            ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            metadata={
                "conversation_state": conversation_state.value,
                "help_type": "contextual" if conversation_state != ConversationState.GREETING else "general",
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    
    async def _handle_unknown_intent(
        self, 
        intent: UserIntent, 
        session_id: str,
        conversation_state: ConversationState
    ) -> AgentResponse:
        """处理未知意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            conversation_state: 当前对话状态
            
        Returns:
            Agent响应
        """
        # 获取对话历史以提供更好的上下文建议
        history = await self.context_manager.get_conversation_history(session_id, limit=2)
        
        # 分析用户输入，尝试提供有用的建议
        user_input = intent.raw_input.lower()
        suggestions = []
        
        # 检查是否包含SQL关键词
        sql_keywords = ["select", "insert", "update", "delete", "create", "alter", "drop"]
        if any(keyword in user_input for keyword in sql_keywords):
            suggestions.append("看起来您想要分析SQL查询，请将完整的SQL语句放在代码块中")
        
        # 检查是否询问性能问题
        performance_keywords = ["慢", "slow", "性能", "performance", "优化", "optimize"]
        if any(keyword in user_input for keyword in performance_keywords):
            suggestions.append("如果您遇到性能问题，我可以帮助分析和优化")
        
        # 检查是否询问概念
        concept_keywords = ["什么是", "what is", "如何", "how", "为什么", "why"]
        if any(keyword in user_input for keyword in concept_keywords):
            suggestions.append("我可以解释数据库相关的概念和原理")
        
        if not suggestions:
            suggestions = [
                "尝试描述您遇到的具体问题",
                "提供需要分析的SQL查询",
                "询问数据库相关的概念"
            ]
        
        # 构建响应内容
        sections = {
            "理解状况": f"抱歉，我没有完全理解您的请求。置信度：{intent.confidence:.1%}",
            "我的能力": "• SQL查询性能分析\n• 数据库优化建议\n• 性能监控设置\n• 技术知识问答",
            "建议尝试": "\n".join(f"• {suggestion}" for suggestion in suggestions)
        }
        
        # 如果有对话历史，提供上下文相关的建议
        if history:
            last_message = history[-1]
            if last_message.role == MessageRole.ASSISTANT:
                sections["上下文提示"] = "基于我们之前的对话，您可能想要继续讨论相关话题。"
        
        content = self.response_formatter.format_structured_response(
            "需要更多信息",
            sections
        )
        
        suggested_actions = [
            "输入'帮助'查看详细功能",
            "提供SQL查询进行分析",
            "描述具体的数据库问题",
            "询问数据库概念"
        ]
        
        return AgentResponse(
            content=content,
            intent_handled=intent.intent_type,
            suggested_actions=suggested_actions,
            metadata={
                "conversation_state": conversation_state.value,
                "confidence": intent.confidence,
                "suggestions_provided": len(suggestions),
                "format": ResponseFormat.STRUCTURED.value
            }
        )
    

    async def handle_followup_questions(self, session_id: str) -> List[str]:
        """生成后续问题建议（基于对话状态和历史）.
        
        Args:
            session_id: 会话ID
            
        Returns:
            后续问题建议列表
        """
        # 获取当前对话状态和历史
        current_state = await self._get_conversation_state(session_id)
        history = await self.context_manager.get_conversation_history(session_id, limit=3)
        current_db = await self.context_manager.get_current_database(session_id)
        active_tasks = await self.context_manager.get_active_tasks(session_id)
        
        # 根据对话状态生成相关的后续问题
        if current_state == ConversationState.GREETING:
            return [
                "您想分析哪个SQL查询的性能？",
                "需要什么类型的数据库优化建议？",
                "想了解哪个数据库概念？",
                "需要设置数据库监控吗？"
            ]
        
        elif current_state == ConversationState.QUERY_INPUT:
            questions = [
                "请提供您要分析的SQL查询语句",
                "需要指定特定的数据库吗？"
            ]
            if current_db:
                questions.append(f"是否在{current_db}数据库中执行分析？")
            return questions
        
        elif current_state == ConversationState.ANALYSIS_PENDING:
            return [
                "分析完成后需要查看详细的执行计划吗？",
                "想了解具体的性能瓶颈在哪里吗？",
                "需要获取优化建议吗？",
                "要设置针对此查询的监控吗？"
            ]
        
        elif current_state == ConversationState.ANALYSIS_COMPLETE:
            return [
                "需要查看优化建议的详细说明吗？",
                "想了解如何实施这些优化吗？",
                "需要对比优化前后的性能吗？",
                "要为这个查询设置监控告警吗？"
            ]
        
        elif current_state == ConversationState.OPTIMIZATION_DISCUSSION:
            return [
                "需要我执行推荐的优化操作吗？",
                "想看查询重写的具体示例吗？",
                "需要了解索引创建的影响吗？",
                "要查看其他优化方案吗？"
            ]
        
        elif current_state == ConversationState.MONITORING_SETUP:
            return [
                "需要设置慢查询告警阈值吗？",
                "想配置性能指标监控吗？",
                "需要启用异常检测功能吗？",
                "要设置定期性能报告吗？"
            ]
        
        elif current_state == ConversationState.KNOWLEDGE_SHARING:
            return [
                "需要更详细的技术解释吗？",
                "想看相关的实际案例吗？",
                "需要了解最佳实践建议吗？",
                "要查看其他相关概念吗？"
            ]
        
        elif current_state == ConversationState.TASK_EXECUTION:
            return [
                "需要查看任务执行进度吗？",
                "想了解执行过程中的详细信息吗？",
                "需要修改执行参数吗？",
                "要暂停或取消当前任务吗？"
            ]
        
        elif current_state == ConversationState.CONFIRMATION_PENDING:
            return [
                "确认执行这个操作吗？",
                "需要查看操作的详细计划吗？",
                "想了解操作的风险和影响吗？",
                "要修改操作参数吗？"
            ]
        
        elif current_state == ConversationState.ERROR_HANDLING:
            return [
                "需要重新尝试刚才的操作吗？",
                "想了解错误的具体原因吗？",
                "需要查看解决方案建议吗？",
                "要切换到其他功能吗？"
            ]
        
        else:  # IDLE or other states
            # 基于对话历史和活跃任务生成建议
            questions = []
            
            if active_tasks:
                questions.append("需要查看当前任务的执行状态吗？")
            
            if history:
                last_message = history[-1] if history else None
                if last_message and "分析" in last_message.content:
                    questions.extend([
                        "还有其他SQL查询需要分析吗？",
                        "需要深入了解分析结果吗？"
                    ])
                elif last_message and "优化" in last_message.content:
                    questions.extend([
                        "需要执行更多优化操作吗？",
                        "想了解其他优化策略吗？"
                    ])
            
            if not questions:
                questions = [
                    "还有其他数据库问题需要帮助吗？",
                    "需要分析新的SQL查询吗？",
                    "想了解数据库优化的最佳实践吗？",
                    "需要设置数据库监控吗？"
                ]
            
            return questions
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """获取对话摘要信息.
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话摘要字典
        """
        # 获取基本上下文摘要
        summary = await self.context_manager.get_context_summary(session_id)
        
        # 添加对话管理器特有的信息
        current_state = await self._get_conversation_state(session_id)
        summary.update({
            "conversation_state": current_state.value,
            "persistence_enabled": self.enable_persistence,
            "valid_intents": [intent.value for intent in self.conversation_flow.get_valid_intents(current_state)]
        })
        
        return summary
    
    async def reset_conversation(self, session_id: str) -> bool:
        """重置对话状态到初始状态.
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否重置成功
        """
        try:
            # 重置对话状态
            await self._set_conversation_state(session_id, ConversationState.GREETING)
            
            # 清空上下文（保留基本信息）
            await self.context_manager.clear_context(session_id)
            
            # 发布重置事件
            await publish_event(
                EventType.CONTEXT_UPDATED,
                source="conversation_manager",
                data={"action": "conversation_reset"},
                session_id=session_id
            )
            
            logger.info(f"对话已重置: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"重置对话失败 {session_id}: {e}")
            return False
    
    def set_sql_integrator(self, sql_integrator):
        """设置SQL分析集成器.
        
        Args:
            sql_integrator: SQL分析集成器实例
        """
        self.sql_integrator = sql_integrator
        logger.info("SQL分析集成器已设置")