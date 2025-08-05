"""NLP Agent - 负责自然语言处理和理解."""

import logging
from typing import Any, Dict, List, Optional

from ..communication.a2a_protocol import A2AMessage
from .base_agent import BaseAgent
from ...nlp.processor import NLPProcessor
from ...nlp.intent_classifier import IntentClassifier
from ...nlp.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class NLPAgent(BaseAgent):
    """NLP Agent - 负责自然语言处理和理解."""
    
    def __init__(self):
        """初始化NLP Agent."""
        super().__init__(
            agent_id="nlp_agent",
            agent_name="Natural Language Processing Agent",
            agent_type="nlp",
            capabilities=[
                "intent_recognition",
                "entity_extraction",
                "natural_language_to_sql",
                "response_generation",
                "conversation_management"
            ]
        )
        
        # NLP组件
        self.nlp_processor: Optional[NLPProcessor] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        
        # 对话状态管理
        self._conversation_contexts: Dict[str, Dict[str, Any]] = {}
    
    async def _initialize(self):
        """初始化NLP Agent."""
        try:
            # 初始化NLP组件
            self.nlp_processor = NLPProcessor()
            self.intent_classifier = IntentClassifier()
            self.entity_extractor = EntityExtractor()
            
            # 加载模型和数据
            await self._load_models()
            
            logger.info("NLP Agent初始化完成")
            
        except Exception as e:
            logger.error(f"NLP Agent初始化失败: {e}")
            raise
    
    async def _cleanup(self):
        """清理NLP Agent."""
        # 清理对话上下文
        self._conversation_contexts.clear()
        logger.info("NLP Agent清理完成")
    
    async def _register_custom_handlers(self):
        """注册NLP Agent特定的消息处理器."""
        handlers = {
            "process_natural_language": self._handle_process_natural_language,
            "classify_intent": self._handle_classify_intent,
            "extract_entities": self._handle_extract_entities,
            "generate_sql": self._handle_generate_sql,
            "generate_response": self._handle_generate_response,
            "manage_conversation": self._handle_manage_conversation,
            "get_conversation_context": self._handle_get_conversation_context
        }
        
        for action, handler in handlers.items():
            self._message_handler.register_handler(action, handler)
    
    async def _load_models(self):
        """加载NLP模型和数据."""
        try:
            # 这里可以加载预训练的模型
            # 目前使用基础实现
            pass
        except Exception as e:
            logger.error(f"加载NLP模型失败: {e}")
            raise
    
    async def _handle_process_natural_language(self, message: A2AMessage) -> Dict[str, Any]:
        """处理自然语言处理请求.
        
        Args:
            message: 消息对象
            
        Returns:
            处理结果
        """
        try:
            text = message.payload.get("text", "")
            context = message.payload.get("context", {})
            session_id = message.payload.get("session_id")
            
            if not text:
                return {
                    "success": False,
                    "error": "Missing text parameter"
                }
            
            # 意图识别
            intent_result = await self.intent_classifier.classify_intent(text, context)
            
            # 实体提取
            entities = await self.entity_extractor.extract_entities(text, context)
            
            # 更新对话上下文
            if session_id:
                self._update_conversation_context(session_id, {
                    "last_input": text,
                    "intent": intent_result,
                    "entities": entities,
                    "timestamp": message.timestamp.isoformat()
                })
            
            return {
                "success": True,
                "intent": intent_result.dict() if intent_result else None,
                "entities": [entity.dict() for entity in entities],
                "processed_text": text,
                "context_updated": session_id is not None
            }
            
        except Exception as e:
            logger.error(f"自然语言处理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_classify_intent(self, message: A2AMessage) -> Dict[str, Any]:
        """处理意图分类请求.
        
        Args:
            message: 消息对象
            
        Returns:
            分类结果
        """
        try:
            text = message.payload.get("text", "")
            context = message.payload.get("context", {})
            
            if not text:
                return {
                    "success": False,
                    "error": "Missing text parameter"
                }
            
            intent_result = await self.intent_classifier.classify_intent(text, context)
            
            return {
                "success": True,
                "intent": intent_result.dict() if intent_result else None
            }
            
        except Exception as e:
            logger.error(f"意图分类失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_extract_entities(self, message: A2AMessage) -> Dict[str, Any]:
        """处理实体提取请求.
        
        Args:
            message: 消息对象
            
        Returns:
            提取结果
        """
        try:
            text = message.payload.get("text", "")
            context = message.payload.get("context", {})
            
            if not text:
                return {
                    "success": False,
                    "error": "Missing text parameter"
                }
            
            entities = await self.entity_extractor.extract_entities(text, context)
            
            return {
                "success": True,
                "entities": [entity.dict() for entity in entities]
            }
            
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_generate_sql(self, message: A2AMessage) -> Dict[str, Any]:
        """处理自然语言转SQL请求.
        
        Args:
            message: 消息对象
            
        Returns:
            SQL生成结果
        """
        try:
            natural_query = message.payload.get("natural_query", "")
            database_schema = message.payload.get("database_schema", {})
            context = message.payload.get("context", {})
            
            if not natural_query:
                return {
                    "success": False,
                    "error": "Missing natural_query parameter"
                }
            
            # 这里应该实现自然语言到SQL的转换逻辑
            # 目前返回模拟结果
            sql_query = await self._convert_to_sql(natural_query, database_schema, context)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "confidence": 0.85,
                "explanation": f"将自然语言查询 '{natural_query}' 转换为SQL"
            }
            
        except Exception as e:
            logger.error(f"SQL生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_generate_response(self, message: A2AMessage) -> Dict[str, Any]:
        """处理响应生成请求.
        
        Args:
            message: 消息对象
            
        Returns:
            响应生成结果
        """
        try:
            analysis_result = message.payload.get("analysis_result", {})
            user_intent = message.payload.get("user_intent", {})
            context = message.payload.get("context", {})
            
            # 生成自然语言响应
            response_text = await self._generate_natural_response(
                analysis_result, user_intent, context
            )
            
            return {
                "success": True,
                "response_text": response_text,
                "response_type": "natural_language"
            }
            
        except Exception as e:
            logger.error(f"响应生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_manage_conversation(self, message: A2AMessage) -> Dict[str, Any]:
        """处理对话管理请求.
        
        Args:
            message: 消息对象
            
        Returns:
            对话管理结果
        """
        try:
            session_id = message.payload.get("session_id")
            action = message.payload.get("action", "update")
            data = message.payload.get("data", {})
            
            if not session_id:
                return {
                    "success": False,
                    "error": "Missing session_id parameter"
                }
            
            if action == "update":
                self._update_conversation_context(session_id, data)
            elif action == "clear":
                self._conversation_contexts.pop(session_id, None)
            elif action == "get":
                context = self._conversation_contexts.get(session_id, {})
                return {
                    "success": True,
                    "context": context
                }
            
            return {
                "success": True,
                "action": action,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"对话管理失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_get_conversation_context(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取对话上下文请求.
        
        Args:
            message: 消息对象
            
        Returns:
            对话上下文
        """
        try:
            session_id = message.payload.get("session_id")
            
            if not session_id:
                return {
                    "success": False,
                    "error": "Missing session_id parameter"
                }
            
            context = self._conversation_contexts.get(session_id, {})
            
            return {
                "success": True,
                "session_id": session_id,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_conversation_context(self, session_id: str, data: Dict[str, Any]):
        """更新对话上下文.
        
        Args:
            session_id: 会话ID
            data: 上下文数据
        """
        if session_id not in self._conversation_contexts:
            self._conversation_contexts[session_id] = {}
        
        self._conversation_contexts[session_id].update(data)
    
    async def _convert_to_sql(
        self, 
        natural_query: str, 
        database_schema: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """将自然语言查询转换为SQL.
        
        Args:
            natural_query: 自然语言查询
            database_schema: 数据库模式
            context: 上下文信息
            
        Returns:
            SQL查询语句
        """
        # 这里应该实现复杂的NL2SQL转换逻辑
        # 目前返回简单的模拟结果
        
        query_lower = natural_query.lower()
        
        if "用户" in natural_query and "查询" in natural_query:
            return "SELECT * FROM users WHERE status = 'active'"
        elif "订单" in natural_query and "统计" in natural_query:
            return "SELECT COUNT(*) FROM orders WHERE created_at >= CURDATE()"
        elif "性能" in natural_query and "慢查询" in natural_query:
            return "SELECT * FROM slow_query_log WHERE query_time > 1.0"
        else:
            return f"-- 无法解析的查询: {natural_query}"
    
    async def _generate_natural_response(
        self, 
        analysis_result: Dict[str, Any], 
        user_intent: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """生成自然语言响应.
        
        Args:
            analysis_result: 分析结果
            user_intent: 用户意图
            context: 上下文信息
            
        Returns:
            自然语言响应
        """
        # 根据分析结果和用户意图生成响应
        
        if analysis_result.get("performance_score"):
            score = analysis_result["performance_score"]
            if score >= 80:
                return f"您的SQL查询性能良好，得分为{score}分。查询执行效率较高，无需特别优化。"
            elif score >= 60:
                return f"您的SQL查询性能尚可，得分为{score}分。建议考虑以下优化建议来提升性能。"
            else:
                return f"您的SQL查询性能较差，得分仅为{score}分。强烈建议进行优化以提升查询效率。"
        
        if analysis_result.get("issues"):
            issue_count = len(analysis_result["issues"])
            return f"分析发现了{issue_count}个性能问题，建议您查看详细的优化建议。"
        
        return "SQL分析已完成，请查看详细的分析报告。"