"""知识管理Agent - 负责数据库优化知识库管理和最佳实践推荐."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..communication.a2a_protocol import A2AMessage
from .base_agent import BaseAgent
from ..services.text_chunking import ChunkingManager, TextChunk

logger = logging.getLogger(__name__)


class KnowledgeItem:
    """知识条目模型."""
    
    def __init__(
        self,
        knowledge_id: str,
        title: str,
        description: str,
        category: str,
        database_types: List[str],
        problem_patterns: List[str],
        solution_steps: List[str],
        effectiveness_score: float = 0.0,
        confidence_score: float = 0.0
    ):
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.category = category
        self.database_types = database_types
        self.problem_patterns = problem_patterns
        self.solution_steps = solution_steps
        self.effectiveness_score = effectiveness_score
        self.confidence_score = confidence_score
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式."""
        return {
            "knowledge_id": self.knowledge_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "database_types": self.database_types,
            "problem_patterns": self.problem_patterns,
            "solution_steps": self.solution_steps,
            "effectiveness_score": self.effectiveness_score,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count
        }


class OptimizationPattern:
    """优化模式模型."""
    
    def __init__(
        self,
        pattern_id: str,
        pattern_name: str,
        problem_signature: str,
        solution_template: str,
        applicability_conditions: List[str],
        success_rate: float = 0.0
    ):
        self.pattern_id = pattern_id
        self.pattern_name = pattern_name
        self.problem_signature = problem_signature
        self.solution_template = solution_template
        self.applicability_conditions = applicability_conditions
        self.success_rate = success_rate
        self.discovered_from = []
        self.validation_status = "pending"
        self.created_at = datetime.now()


class KnowledgeAgent(BaseAgent):
    """知识管理Agent - 负责数据库优化知识库管理和最佳实践推荐."""
    
    def __init__(self):
        """初始化知识管理Agent."""
        super().__init__(
            agent_id="knowledge_agent",
            agent_name="Knowledge Management Agent",
            agent_type="knowledge",
            capabilities=[
                "knowledge_storage",
                "best_practices_recommendation",
                "pattern_recognition",
                "effectiveness_tracking",
                "knowledge_validation"
            ]
        )
        
        # 知识库存储
        self._knowledge_base: Dict[str, KnowledgeItem] = {}
        self._optimization_patterns: Dict[str, OptimizationPattern] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._database_type_index: Dict[str, List[str]] = {}
        
        # 初始化分块管理器
        self.chunking_manager = ChunkingManager()
        
        # 向量化存储（用于语义搜索）
        self._knowledge_embeddings: Dict[str, List[float]] = {}
        
        # 统计信息
        self._query_count = 0
        self._recommendation_count = 0
        self._pattern_discovery_count = 0
    
    async def _initialize(self):
        """初始化知识管理Agent."""
        # 加载预定义的知识库
        await self._load_default_knowledge()
        logger.info("知识管理Agent初始化完成")
    
    async def _cleanup(self):
        """清理知识管理Agent."""
        logger.info("知识管理Agent清理完成")
    
    async def _register_custom_handlers(self):
        """注册知识管理Agent特定的消息处理器."""
        handlers = {
            "query_knowledge": self._handle_query_knowledge,
            "add_knowledge": self._handle_add_knowledge,
            "update_effectiveness": self._handle_update_effectiveness,
            "get_best_practices": self._handle_get_best_practices,
            "store_pattern": self._handle_store_pattern,
            "validate_knowledge": self._handle_validate_knowledge,
            "get_knowledge_stats": self._handle_get_knowledge_stats
        }
        
        for action, handler in handlers.items():
            self._message_handler.register_handler(action, handler)
    
    async def _load_default_knowledge(self):
        """加载默认知识库."""
        # 索引优化知识
        index_knowledge = [
            KnowledgeItem(
                knowledge_id="idx_001",
                title="WHERE子句索引优化",
                description="为WHERE子句中的过滤条件创建合适的索引",
                category="index_optimization",
                database_types=["mysql", "postgresql"],
                problem_patterns=["full_table_scan", "where_clause_without_index"],
                solution_steps=[
                    "分析WHERE子句中的过滤条件",
                    "检查现有索引覆盖情况",
                    "创建复合索引覆盖多个过滤条件",
                    "验证索引使用效果"
                ],
                effectiveness_score=0.85,
                confidence_score=0.9
            ),
            KnowledgeItem(
                knowledge_id="idx_002",
                title="JOIN操作索引优化",
                description="为JOIN操作中的连接条件创建索引",
                category="index_optimization",
                database_types=["mysql", "postgresql"],
                problem_patterns=["nested_loop_join", "join_without_index"],
                solution_steps=[
                    "识别JOIN操作中的连接字段",
                    "为外键字段创建索引",
                    "考虑创建覆盖索引包含SELECT字段",
                    "优化JOIN顺序"
                ],
                effectiveness_score=0.8,
                confidence_score=0.85
            )
        ]
        
        # 查询优化知识
        query_knowledge = [
            KnowledgeItem(
                knowledge_id="qry_001",
                title="子查询优化",
                description="将相关子查询转换为JOIN操作",
                category="query_optimization",
                database_types=["mysql", "postgresql"],
                problem_patterns=["correlated_subquery", "exists_subquery"],
                solution_steps=[
                    "识别相关子查询",
                    "分析子查询与主查询的关系",
                    "转换为LEFT JOIN或INNER JOIN",
                    "验证结果一致性"
                ],
                effectiveness_score=0.75,
                confidence_score=0.8
            )
        ]
        
        # 添加知识到知识库
        all_knowledge = index_knowledge + query_knowledge
        for knowledge in all_knowledge:
            await self._store_knowledge_item(knowledge)
    
    async def _handle_query_knowledge(self, message: A2AMessage) -> Dict[str, Any]:
        """处理知识查询请求."""
        try:
            self._query_count += 1
            
            problem_type = message.payload.get("problem_type", "")
            database_type = message.payload.get("database_type", "")
            category = message.payload.get("category", "")
            query_text = message.payload.get("query_text", "")
            limit = message.payload.get("limit", 10)
            
            # 搜索相关知识
            relevant_knowledge = await self._search_knowledge(
                problem_type=problem_type,
                database_type=database_type,
                category=category,
                query_text=query_text
            )
            
            # 限制返回结果数量
            limited_knowledge = relevant_knowledge[:limit]
            
            # 更新使用统计
            for knowledge in limited_knowledge:
                knowledge.usage_count += 1
                knowledge.updated_at = datetime.now()
            
            return {
                "success": True,
                "knowledge_items": [k.to_dict() for k in limited_knowledge],
                "total_count": len(limited_knowledge),
                "search_metadata": {
                    "query_text": query_text,
                    "problem_type": problem_type,
                    "database_type": database_type,
                    "category": category,
                    "semantic_search_enabled": bool(query_text)
                }
            }
            
        except Exception as e:
            logger.error(f"知识查询失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_add_knowledge(self, message: A2AMessage) -> Dict[str, Any]:
        """处理添加知识请求."""
        try:
            knowledge_data = message.payload.get("knowledge_data", {})
            
            # 创建知识条目
            knowledge = KnowledgeItem(
                knowledge_id=knowledge_data.get("knowledge_id", str(uuid4())),
                title=knowledge_data.get("title", ""),
                description=knowledge_data.get("description", ""),
                category=knowledge_data.get("category", ""),
                database_types=knowledge_data.get("database_types", []),
                problem_patterns=knowledge_data.get("problem_patterns", []),
                solution_steps=knowledge_data.get("solution_steps", []),
                effectiveness_score=knowledge_data.get("effectiveness_score", 0.0),
                confidence_score=knowledge_data.get("confidence_score", 0.0)
            )
            
            # 存储知识条目
            await self._store_knowledge_item(knowledge)
            
            return {
                "success": True,
                "knowledge_id": knowledge.knowledge_id,
                "message": "知识条目添加成功"
            }
            
        except Exception as e:
            logger.error(f"添加知识失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_update_effectiveness(self, message: A2AMessage) -> Dict[str, Any]:
        """处理更新有效性请求."""
        try:
            knowledge_id = message.payload.get("knowledge_id", "")
            effectiveness_score = message.payload.get("effectiveness_score", 0.0)
            feedback_type = message.payload.get("feedback_type", "usage")
            
            knowledge = self._knowledge_base.get(knowledge_id)
            if not knowledge:
                return {
                    "success": False,
                    "error": f"知识条目不存在: {knowledge_id}"
                }
            
            # 更新有效性分数
            if feedback_type == "usage":
                knowledge.usage_count += 1
                # 使用加权平均更新有效性分数
                weight = 0.1
                knowledge.effectiveness_score = (
                    (1 - weight) * knowledge.effectiveness_score + 
                    weight * effectiveness_score
                )
            
            knowledge.updated_at = datetime.now()
            
            return {
                "success": True,
                "knowledge_id": knowledge_id,
                "new_effectiveness_score": knowledge.effectiveness_score,
                "usage_count": knowledge.usage_count
            }
            
        except Exception as e:
            logger.error(f"更新有效性失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_get_best_practices(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取最佳实践请求."""
        try:
            self._recommendation_count += 1
            
            database_type = message.payload.get("database_type", "")
            category = message.payload.get("category", "")
            limit = message.payload.get("limit", 10)
            
            # 获取最佳实践
            best_practices = await self._get_best_practices(
                database_type=database_type,
                category=category,
                limit=limit
            )
            
            return {
                "success": True,
                "best_practices": [bp.to_dict() for bp in best_practices],
                "total_count": len(best_practices)
            }
            
        except Exception as e:
            logger.error(f"获取最佳实践失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_store_pattern(self, message: A2AMessage) -> Dict[str, Any]:
        """处理存储优化模式请求."""
        try:
            self._pattern_discovery_count += 1
            
            pattern_data = message.payload.get("pattern_data", {})
            
            # 创建优化模式
            pattern = OptimizationPattern(
                pattern_id=pattern_data.get("pattern_id", str(uuid4())),
                pattern_name=pattern_data.get("pattern_name", ""),
                problem_signature=pattern_data.get("problem_signature", ""),
                solution_template=pattern_data.get("solution_template", ""),
                applicability_conditions=pattern_data.get("applicability_conditions", []),
                success_rate=pattern_data.get("success_rate", 0.0)
            )
            
            # 存储优化模式
            self._optimization_patterns[pattern.pattern_id] = pattern
            
            return {
                "success": True,
                "pattern_id": pattern.pattern_id,
                "message": "优化模式存储成功"
            }
            
        except Exception as e:
            logger.error(f"存储优化模式失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_validate_knowledge(self, message: A2AMessage) -> Dict[str, Any]:
        """处理知识验证请求."""
        try:
            validation_results = []
            
            for knowledge_id, knowledge in self._knowledge_base.items():
                # 简单的验证逻辑
                is_valid = True
                issues = []
                
                # 检查必要字段
                if not knowledge.title:
                    is_valid = False
                    issues.append("缺少标题")
                
                if not knowledge.solution_steps:
                    is_valid = False
                    issues.append("缺少解决步骤")
                
                # 检查有效性分数
                if knowledge.effectiveness_score < 0.3:
                    issues.append("有效性分数过低")
                
                validation_results.append({
                    "knowledge_id": knowledge_id,
                    "is_valid": is_valid,
                    "issues": issues,
                    "effectiveness_score": knowledge.effectiveness_score,
                    "usage_count": knowledge.usage_count
                })
            
            return {
                "success": True,
                "validation_results": validation_results,
                "total_validated": len(validation_results)
            }
            
        except Exception as e:
            logger.error(f"知识验证失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_get_knowledge_stats(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取知识统计请求."""
        try:
            # 统计各类别的知识数量
            category_stats = {}
            database_type_stats = {}
            
            for knowledge in self._knowledge_base.values():
                # 统计类别
                category = knowledge.category
                category_stats[category] = category_stats.get(category, 0) + 1
                
                # 统计数据库类型
                for db_type in knowledge.database_types:
                    database_type_stats[db_type] = database_type_stats.get(db_type, 0) + 1
            
            return {
                "success": True,
                "stats": {
                    "total_knowledge_items": len(self._knowledge_base),
                    "total_patterns": len(self._optimization_patterns),
                    "query_count": self._query_count,
                    "recommendation_count": self._recommendation_count,
                    "pattern_discovery_count": self._pattern_discovery_count,
                    "category_distribution": category_stats,
                    "database_type_distribution": database_type_stats
                }
            }
            
        except Exception as e:
            logger.error(f"获取知识统计失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_knowledge_item(self, knowledge: KnowledgeItem):
        """存储知识条目并更新索引."""
        # 存储知识条目
        self._knowledge_base[knowledge.knowledge_id] = knowledge
        
        # 生成知识条目的文本表示用于向量化
        knowledge_text = self._create_knowledge_text(knowledge)
        
        # 对知识内容进行分块和向量化
        chunks = self.chunking_manager.chunk_and_embed(
            text=knowledge_text,
            strategy="semantic",
            metadata={
                "knowledge_id": knowledge.knowledge_id,
                "category": knowledge.category,
                "database_types": knowledge.database_types,
                "type": "knowledge_entry"
            }
        )
        
        # 存储主要的嵌入向量（使用第一个块的向量）
        if chunks and chunks[0].embedding:
            self._knowledge_embeddings[knowledge.knowledge_id] = chunks[0].embedding
        
        # 更新类别索引
        category = knowledge.category
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(knowledge.knowledge_id)
        
        # 更新数据库类型索引
        for db_type in knowledge.database_types:
            if db_type not in self._database_type_index:
                self._database_type_index[db_type] = []
            self._database_type_index[db_type].append(knowledge.knowledge_id)
    
    def _create_knowledge_text(self, knowledge: KnowledgeItem) -> str:
        """创建知识条目的文本表示."""
        text_parts = [
            f"标题: {knowledge.title}",
            f"描述: {knowledge.description}",
            f"类别: {knowledge.category}",
            f"适用数据库: {', '.join(knowledge.database_types)}",
            f"问题模式: {', '.join(knowledge.problem_patterns)}",
            "解决步骤:",
        ]
        
        for i, step in enumerate(knowledge.solution_steps, 1):
            text_parts.append(f"{i}. {step}")
        
        return "\n".join(text_parts)
    
    async def _search_knowledge(
        self,
        problem_type: str = "",
        database_type: str = "",
        category: str = "",
        query_text: str = ""
    ) -> List[KnowledgeItem]:
        """搜索相关知识，支持语义搜索和关键词匹配."""
        relevant_knowledge = []
        
        # 如果有查询文本，使用语义搜索
        query_embedding = None
        if query_text:
            query_embedding = self.chunking_manager.embedder.embed_text(query_text)
        
        for knowledge in self._knowledge_base.values():
            score = 0
            
            # 语义相似度评分
            if query_embedding and knowledge.knowledge_id in self._knowledge_embeddings:
                knowledge_embedding = self._knowledge_embeddings[knowledge.knowledge_id]
                semantic_similarity = self._calculate_cosine_similarity(
                    query_embedding, knowledge_embedding
                )
                score += semantic_similarity * 3  # 语义相似度权重较高
            
            # 匹配问题类型
            if problem_type:
                for pattern in knowledge.problem_patterns:
                    if problem_type.lower() in pattern.lower():
                        score += 2
                        break
            
            # 关键词匹配
            if query_text:
                knowledge_text = self._create_knowledge_text(knowledge).lower()
                query_words = query_text.lower().split()
                matched_words = sum(1 for word in query_words if word in knowledge_text)
                if query_words:
                    keyword_score = matched_words / len(query_words)
                    score += keyword_score * 1.5
            
            # 匹配数据库类型
            if database_type and database_type in knowledge.database_types:
                score += 1
            
            # 匹配类别
            if category and category == knowledge.category:
                score += 1
            
            # 基于有效性分数和使用频率加权
            score *= (knowledge.effectiveness_score * 0.8 + 
                     min(knowledge.usage_count * 0.01, 0.2))
            
            if score > 0:
                relevant_knowledge.append((knowledge, score))
        
        # 按分数排序
        relevant_knowledge.sort(key=lambda x: x[1], reverse=True)
        
        return [k[0] for k in relevant_knowledge]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度."""
        try:
            import numpy as np
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    async def _get_best_practices(
        self,
        database_type: str = "",
        category: str = "",
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """获取最佳实践."""
        candidates = []
        
        for knowledge in self._knowledge_base.values():
            # 过滤条件
            if database_type and database_type not in knowledge.database_types:
                continue
            
            if category and category != knowledge.category:
                continue
            
            # 只返回高质量的知识条目
            if knowledge.effectiveness_score >= 0.7:
                candidates.append(knowledge)
        
        # 按有效性分数和使用次数排序
        candidates.sort(
            key=lambda k: (k.effectiveness_score, k.usage_count),
            reverse=True
        )
        
        return candidates[:limit]