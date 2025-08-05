"""记忆管理Agent - 负责存储和检索历史分析记录、用户上下文和向量化搜索."""

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from ..communication.a2a_protocol import A2AMessage
from .base_agent import BaseAgent
from ..services.text_chunking import ChunkingManager, TextChunk

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """记忆条目模型."""
    memory_id: str = Field(default_factory=lambda: str(uuid4()), description="记忆ID")
    user_id: str = Field(..., description="用户ID")
    content: str = Field(..., description="记忆内容")
    content_type: str = Field(default="general", description="内容类型")
    embedding: Optional[List[float]] = Field(default=None, description="嵌入向量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    memory_type: str = Field(default="short_term", description="记忆类型")
    importance_score: float = Field(default=0.5, description="重要性评分")
    access_count: int = Field(default=0, description="访问次数")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    last_accessed: datetime = Field(default_factory=datetime.now, description="最后访问时间")
    expires_at: Optional[datetime] = Field(default=None, description="过期时间")
    tags: List[str] = Field(default_factory=list, description="标签")
    related_memories: List[str] = Field(default_factory=list, description="相关记忆ID")


class UserContext(BaseModel):
    """用户上下文模型."""
    user_id: str = Field(..., description="用户ID")
    current_session: str = Field(..., description="当前会话ID")
    recent_queries: List[Dict[str, Any]] = Field(default_factory=list, description="最近查询")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好")
    skill_level: str = Field(default="intermediate", description="技能水平")
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, description="交互历史")
    context_summary: str = Field(default="", description="上下文摘要")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")


class MemorySearchResult(BaseModel):
    """记忆搜索结果."""
    memory_entry: MemoryEntry
    similarity_score: float
    relevance_reason: str


class MemoryAgent(BaseAgent):
    """记忆管理Agent - 负责存储和检索历史分析记录、用户上下文和向量化搜索."""
    
    def __init__(self, 
                 chroma_persist_directory: str = "./chroma_db",
                 max_short_term_memories: int = 1000,
                 max_long_term_memories: int = 10000):
        """
        初始化记忆管理Agent.
        
        Args:
            chroma_persist_directory: ChromaDB持久化目录
            max_short_term_memories: 短期记忆最大数量
            max_long_term_memories: 长期记忆最大数量
        """
        super().__init__(
            agent_id="memory_agent",
            agent_name="Memory Management Agent",
            agent_type="memory",
            capabilities=[
                "memory_storage",
                "memory_retrieval", 
                "vector_search",
                "context_management",
                "similarity_search",
                "memory_consolidation"
            ]
        )
        
        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建集合
        self.memory_collection = self._get_or_create_collection("memories")
        self.context_collection = self._get_or_create_collection("user_contexts")
        
        # 内存存储
        self._memories: Dict[str, MemoryEntry] = {}
        self._user_contexts: Dict[str, UserContext] = {}
        
        # 配置参数
        self.max_short_term_memories = max_short_term_memories
        self.max_long_term_memories = max_long_term_memories
        
        # 初始化分块管理器
        self.chunking_manager = ChunkingManager()
        
        # 注册消息处理器
        self._register_handlers()
        
        logger.info("记忆管理Agent初始化完成")
    
    def _get_or_create_collection(self, name: str):
        """获取或创建ChromaDB集合."""
        try:
            return self.chroma_client.get_collection(name)
        except Exception:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _register_handlers(self):
        """注册消息处理器."""
        self.register_handler("store_memory", self._handle_store_memory)
        self.register_handler("search_memory", self._handle_search_memory)
        self.register_handler("get_user_context", self._handle_get_user_context)
        self.register_handler("update_user_context", self._handle_update_user_context)
        self.register_handler("consolidate_memories", self._handle_consolidate_memories)
        self.register_handler("cleanup_memories", self._handle_cleanup_memories)
    
    async def _handle_store_memory(self, message: A2AMessage) -> Dict[str, Any]:
        """处理存储记忆请求."""
        try:
            payload = message.payload
            user_id = payload.get("user_id")
            content = payload.get("content")
            content_type = payload.get("content_type", "general")
            metadata = payload.get("metadata", {})
            memory_type = payload.get("memory_type", "short_term")
            importance_score = payload.get("importance_score", 0.5)
            
            if not user_id or not content:
                return {"success": False, "error": "缺少必要参数"}
            
            # 存储记忆
            memory_id = await self.store_memory(
                user_id=user_id,
                content=content,
                content_type=content_type,
                metadata=metadata,
                memory_type=memory_type,
                importance_score=importance_score
            )
            
            return {
                "success": True,
                "memory_id": memory_id,
                "message": "记忆存储成功"
            }
            
        except Exception as e:
            logger.error(f"存储记忆失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_search_memory(self, message: A2AMessage) -> Dict[str, Any]:
        """处理记忆搜索请求."""
        try:
            payload = message.payload
            user_id = payload.get("user_id")
            query = payload.get("query")
            limit = payload.get("limit", 10)
            memory_types = payload.get("memory_types", ["short_term", "long_term"])
            min_similarity = payload.get("min_similarity", 0.3)
            
            if not user_id or not query:
                return {"success": False, "error": "缺少必要参数"}
            
            # 搜索记忆
            results = await self.search_memories(
                user_id=user_id,
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_similarity=min_similarity
            )
            
            return {
                "success": True,
                "results": [
                    {
                        "memory_id": result.memory_entry.memory_id,
                        "content": result.memory_entry.content,
                        "similarity_score": result.similarity_score,
                        "relevance_reason": result.relevance_reason,
                        "metadata": result.memory_entry.metadata
                    }
                    for result in results
                ],
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_user_context(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取用户上下文请求."""
        try:
            user_id = message.payload.get("user_id")
            if not user_id:
                return {"success": False, "error": "缺少用户ID"}
            
            context = await self.get_user_context(user_id)
            
            return {
                "success": True,
                "context": context.dict() if context else None
            }
            
        except Exception as e:
            logger.error(f"获取用户上下文失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_update_user_context(self, message: A2AMessage) -> Dict[str, Any]:
        """处理更新用户上下文请求."""
        try:
            payload = message.payload
            user_id = payload.get("user_id")
            updates = payload.get("updates", {})
            
            if not user_id:
                return {"success": False, "error": "缺少用户ID"}
            
            await self.update_user_context(user_id, updates)
            
            return {"success": True, "message": "用户上下文更新成功"}
            
        except Exception as e:
            logger.error(f"更新用户上下文失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_consolidate_memories(self, message: A2AMessage) -> Dict[str, Any]:
        """处理记忆整合请求."""
        try:
            user_id = message.payload.get("user_id")
            consolidation_stats = await self.consolidate_memories(user_id)
            
            return {
                "success": True,
                "stats": consolidation_stats
            }
            
        except Exception as e:
            logger.error(f"记忆整合失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_cleanup_memories(self, message: A2AMessage) -> Dict[str, Any]:
        """处理记忆清理请求."""
        try:
            cleanup_stats = await self.cleanup_expired_memories()
            
            return {
                "success": True,
                "stats": cleanup_stats
            }
            
        except Exception as e:
            logger.error(f"记忆清理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def store_memory(self,
                          user_id: str,
                          content: str,
                          content_type: str = "general",
                          metadata: Dict[str, Any] = None,
                          memory_type: str = "short_term",
                          importance_score: float = 0.5) -> str:
        """
        存储记忆.
        
        Args:
            user_id: 用户ID
            content: 记忆内容
            content_type: 内容类型
            metadata: 元数据
            memory_type: 记忆类型
            importance_score: 重要性评分
            
        Returns:
            记忆ID
        """
        metadata = metadata or {}
        
        # 根据内容类型选择分块策略
        chunking_strategy = self._get_chunking_strategy(content_type)
        
        # 对内容进行分块和向量化
        chunks = self.chunking_manager.chunk_and_embed(
            text=content,
            strategy=chunking_strategy,
            metadata={
                **metadata,
                "user_id": user_id,
                "content_type": content_type,
                "memory_type": memory_type
            }
        )
        
        stored_memories = []
        
        for chunk in chunks:
            # 创建记忆条目
            memory_entry = MemoryEntry(
                user_id=user_id,
                content=chunk.content,
                content_type=content_type,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
                memory_type=memory_type,
                importance_score=importance_score,
                tags=metadata.get("tags", [])
            )
            
            # 存储到内存
            self._memories[memory_entry.memory_id] = memory_entry
            
            # 存储到向量数据库
            if chunk.embedding:
                self.memory_collection.add(
                    ids=[memory_entry.memory_id],
                    embeddings=[chunk.embedding],
                    metadatas=[{
                        "user_id": user_id,
                        "content_type": content_type,
                        "memory_type": memory_type,
                        "importance_score": importance_score,
                        "created_at": memory_entry.created_at.isoformat(),
                        **metadata
                    }],
                    documents=[chunk.content]
                )
            
            stored_memories.append(memory_entry.memory_id)
        
        # 检查是否需要清理旧记忆
        await self._check_memory_limits(user_id, memory_type)
        
        logger.info(f"为用户 {user_id} 存储了 {len(stored_memories)} 个记忆块")
        
        # 返回主记忆ID（第一个块的ID）
        return stored_memories[0] if stored_memories else ""
    
    def _get_chunking_strategy(self, content_type: str) -> str:
        """根据内容类型选择分块策略."""
        strategy_mapping = {
            "conversation": "conversation",
            "sql_analysis": "sql_analysis",
            "user_feedback": "semantic",
            "query_result": "sql_analysis",
            "general": "semantic"
        }
        return strategy_mapping.get(content_type, "semantic")
    
    async def search_memories(self,
                             user_id: str,
                             query: str,
                             limit: int = 10,
                             memory_types: List[str] = None,
                             min_similarity: float = 0.3) -> List[MemorySearchResult]:
        """
        搜索相关记忆.
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            limit: 返回结果数量限制
            memory_types: 记忆类型过滤
            min_similarity: 最小相似度阈值
            
        Returns:
            搜索结果列表
        """
        memory_types = memory_types or ["short_term", "long_term"]
        
        # 生成查询向量
        query_embedding = self.chunking_manager.embedder.embed_text(query)
        if not query_embedding:
            logger.warning("无法生成查询向量，使用文本匹配")
            return await self._text_based_search(user_id, query, limit, memory_types)
        
        # 构建过滤条件
        where_conditions = {
            "user_id": user_id,
            "memory_type": {"$in": memory_types}
        }
        
        try:
            # 向量搜索
            results = self.memory_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # 获取更多结果用于过滤
                where=where_conditions,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # 转换为相似度
                
                if similarity < min_similarity:
                    continue
                
                # 获取记忆条目
                memory_entry = self._memories.get(memory_id)
                if not memory_entry:
                    continue
                
                # 更新访问信息
                memory_entry.access_count += 1
                memory_entry.last_accessed = datetime.now()
                
                # 生成相关性原因
                relevance_reason = self._generate_relevance_reason(
                    query, memory_entry.content, similarity
                )
                
                search_results.append(MemorySearchResult(
                    memory_entry=memory_entry,
                    similarity_score=similarity,
                    relevance_reason=relevance_reason
                ))
            
            # 按相似度排序并限制结果数量
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return await self._text_based_search(user_id, query, limit, memory_types)
    
    async def _text_based_search(self,
                                user_id: str,
                                query: str,
                                limit: int,
                                memory_types: List[str]) -> List[MemorySearchResult]:
        """基于文本的搜索（备用方案）."""
        results = []
        query_lower = query.lower()
        
        for memory_entry in self._memories.values():
            if (memory_entry.user_id == user_id and 
                memory_entry.memory_type in memory_types):
                
                content_lower = memory_entry.content.lower()
                
                # 简单的文本匹配评分
                if query_lower in content_lower:
                    # 计算匹配度
                    match_count = content_lower.count(query_lower)
                    similarity = min(match_count * 0.2, 0.9)
                    
                    results.append(MemorySearchResult(
                        memory_entry=memory_entry,
                        similarity_score=similarity,
                        relevance_reason=f"文本匹配: 包含查询关键词 {match_count} 次"
                    ))
        
        # 按相似度排序并限制结果数量
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]
    
    def _generate_relevance_reason(self, query: str, content: str, similarity: float) -> str:
        """生成相关性原因说明."""
        if similarity > 0.8:
            return f"高度相关 (相似度: {similarity:.2f}) - 内容与查询高度匹配"
        elif similarity > 0.6:
            return f"相关 (相似度: {similarity:.2f}) - 内容与查询部分匹配"
        elif similarity > 0.4:
            return f"可能相关 (相似度: {similarity:.2f}) - 内容与查询有一定关联"
        else:
            return f"弱相关 (相似度: {similarity:.2f}) - 内容与查询关联较弱"
    
    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """获取用户上下文."""
        return self._user_contexts.get(user_id)
    
    async def update_user_context(self, user_id: str, updates: Dict[str, Any]):
        """更新用户上下文."""
        context = self._user_contexts.get(user_id)
        
        if not context:
            context = UserContext(
                user_id=user_id,
                current_session=updates.get("current_session", str(uuid4()))
            )
            self._user_contexts[user_id] = context
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        context.last_updated = datetime.now()
        
        # 存储到向量数据库（如果需要）
        context_text = f"用户偏好: {context.preferences}, 技能水平: {context.skill_level}, 上下文摘要: {context.context_summary}"
        context_embedding = self.chunking_manager.embedder.embed_text(context_text)
        
        if context_embedding:
            try:
                self.context_collection.upsert(
                    ids=[user_id],
                    embeddings=[context_embedding],
                    metadatas=[{
                        "user_id": user_id,
                        "skill_level": context.skill_level,
                        "last_updated": context.last_updated.isoformat()
                    }],
                    documents=[context_text]
                )
            except Exception as e:
                logger.error(f"更新用户上下文向量失败: {e}")
    
    async def consolidate_memories(self, user_id: str = None) -> Dict[str, Any]:
        """整合记忆 - 将相似的短期记忆合并为长期记忆."""
        consolidation_stats = {
            "processed_memories": 0,
            "consolidated_memories": 0,
            "created_long_term": 0
        }
        
        # 获取需要整合的短期记忆
        short_term_memories = [
            memory for memory in self._memories.values()
            if (memory.memory_type == "short_term" and 
                (user_id is None or memory.user_id == user_id) and
                memory.access_count > 2 and  # 被访问过多次
                memory.importance_score > 0.6)  # 重要性较高
        ]
        
        consolidation_stats["processed_memories"] = len(short_term_memories)
        
        # 按用户分组
        user_memories = {}
        for memory in short_term_memories:
            if memory.user_id not in user_memories:
                user_memories[memory.user_id] = []
            user_memories[memory.user_id].append(memory)
        
        # 对每个用户的记忆进行整合
        for uid, memories in user_memories.items():
            consolidated = await self._consolidate_user_memories(uid, memories)
            consolidation_stats["consolidated_memories"] += consolidated["consolidated"]
            consolidation_stats["created_long_term"] += consolidated["created"]
        
        return consolidation_stats
    
    async def _consolidate_user_memories(self, user_id: str, memories: List[MemoryEntry]) -> Dict[str, int]:
        """整合单个用户的记忆."""
        stats = {"consolidated": 0, "created": 0}
        
        if len(memories) < 2:
            return stats
        
        # 使用向量相似度找到相似的记忆
        similar_groups = []
        processed = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.memory_id in processed:
                continue
            
            group = [memory1]
            processed.add(memory1.memory_id)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2.memory_id in processed:
                    continue
                
                # 计算相似度
                if memory1.embedding and memory2.embedding:
                    similarity = self._calculate_cosine_similarity(
                        memory1.embedding, memory2.embedding
                    )
                    
                    if similarity > 0.7:  # 高相似度阈值
                        group.append(memory2)
                        processed.add(memory2.memory_id)
            
            if len(group) > 1:
                similar_groups.append(group)
        
        # 为每个相似组创建长期记忆
        for group in similar_groups:
            consolidated_content = self._merge_memory_contents(group)
            consolidated_metadata = self._merge_memory_metadata(group)
            
            # 创建长期记忆
            long_term_memory = MemoryEntry(
                user_id=user_id,
                content=consolidated_content,
                content_type="consolidated",
                metadata=consolidated_metadata,
                memory_type="long_term",
                importance_score=max(m.importance_score for m in group),
                tags=list(set(tag for m in group for tag in m.tags))
            )
            
            # 生成嵌入向量
            embedding = self.chunking_manager.embedder.embed_text(consolidated_content)
            if embedding:
                long_term_memory.embedding = embedding
                
                # 存储到向量数据库
                self.memory_collection.add(
                    ids=[long_term_memory.memory_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "user_id": user_id,
                        "content_type": "consolidated",
                        "memory_type": "long_term",
                        "importance_score": long_term_memory.importance_score,
                        "created_at": long_term_memory.created_at.isoformat(),
                        **consolidated_metadata
                    }],
                    documents=[consolidated_content]
                )
            
            # 存储到内存
            self._memories[long_term_memory.memory_id] = long_term_memory
            
            # 删除原始短期记忆
            for memory in group:
                if memory.memory_id in self._memories:
                    del self._memories[memory.memory_id]
                
                # 从向量数据库删除
                try:
                    self.memory_collection.delete(ids=[memory.memory_id])
                except Exception as e:
                    logger.warning(f"删除记忆向量失败: {e}")
            
            stats["consolidated"] += len(group)
            stats["created"] += 1
        
        return stats
    
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
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def _merge_memory_contents(self, memories: List[MemoryEntry]) -> str:
        """合并记忆内容."""
        contents = []
        for memory in memories:
            contents.append(f"[{memory.created_at.strftime('%Y-%m-%d %H:%M')}] {memory.content}")
        
        return "\n\n".join(contents)
    
    def _merge_memory_metadata(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """合并记忆元数据."""
        merged = {
            "source_memories": [m.memory_id for m in memories],
            "consolidation_date": datetime.now().isoformat(),
            "total_access_count": sum(m.access_count for m in memories)
        }
        
        # 合并其他元数据
        for memory in memories:
            for key, value in memory.metadata.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, list):
                    if isinstance(merged[key], list):
                        merged[key].extend(value)
                    else:
                        merged[key] = [merged[key], *value]
        
        return merged
    
    async def cleanup_expired_memories(self) -> Dict[str, Any]:
        """清理过期记忆."""
        cleanup_stats = {
            "total_memories": len(self._memories),
            "expired_memories": 0,
            "low_importance_memories": 0,
            "cleaned_memories": 0
        }
        
        now = datetime.now()
        to_delete = []
        
        for memory_id, memory in self._memories.items():
            should_delete = False
            
            # 检查是否过期
            if memory.expires_at and now > memory.expires_at:
                should_delete = True
                cleanup_stats["expired_memories"] += 1
            
            # 检查短期记忆的重要性和访问频率
            elif (memory.memory_type == "short_term" and 
                  memory.importance_score < 0.3 and 
                  memory.access_count < 2 and
                  (now - memory.created_at).days > 7):
                should_delete = True
                cleanup_stats["low_importance_memories"] += 1
            
            if should_delete:
                to_delete.append(memory_id)
        
        # 删除记忆
        for memory_id in to_delete:
            if memory_id in self._memories:
                del self._memories[memory_id]
            
            # 从向量数据库删除
            try:
                self.memory_collection.delete(ids=[memory_id])
            except Exception as e:
                logger.warning(f"删除记忆向量失败: {e}")
        
        cleanup_stats["cleaned_memories"] = len(to_delete)
        
        logger.info(f"清理了 {len(to_delete)} 个过期或低重要性记忆")
        
        return cleanup_stats
    
    async def _check_memory_limits(self, user_id: str, memory_type: str):
        """检查记忆数量限制."""
        user_memories = [
            m for m in self._memories.values() 
            if m.user_id == user_id and m.memory_type == memory_type
        ]
        
        max_limit = (self.max_short_term_memories if memory_type == "short_term" 
                    else self.max_long_term_memories)
        
        if len(user_memories) > max_limit:
            # 按重要性和访问时间排序，删除最不重要的记忆
            user_memories.sort(
                key=lambda m: (m.importance_score, m.last_accessed),
                reverse=False
            )
            
            to_delete = user_memories[:len(user_memories) - max_limit]
            
            for memory in to_delete:
                if memory.memory_id in self._memories:
                    del self._memories[memory.memory_id]
                
                try:
                    self.memory_collection.delete(ids=[memory.memory_id])
                except Exception as e:
                    logger.warning(f"删除记忆向量失败: {e}")
            
            logger.info(f"为用户 {user_id} 清理了 {len(to_delete)} 个 {memory_type} 记忆")
    
    async def get_memory_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """获取记忆统计信息."""
        if user_id:
            user_memories = [m for m in self._memories.values() if m.user_id == user_id]
        else:
            user_memories = list(self._memories.values())
        
        stats = {
            "total_memories": len(user_memories),
            "short_term_memories": len([m for m in user_memories if m.memory_type == "short_term"]),
            "long_term_memories": len([m for m in user_memories if m.memory_type == "long_term"]),
            "avg_importance_score": sum(m.importance_score for m in user_memories) / len(user_memories) if user_memories else 0,
            "total_access_count": sum(m.access_count for m in user_memories),
            "content_types": {}
        }
        
        # 统计内容类型分布
        for memory in user_memories:
            content_type = memory.content_type
            if content_type not in stats["content_types"]:
                stats["content_types"][content_type] = 0
            stats["content_types"][content_type] += 1
        
        return stats


# 使用示例
if __name__ == "__main__":
    import asyncio
    
    async def test_memory_agent():
        # 创建记忆Agent
        memory_agent = MemoryAgent()
        
        # 启动Agent
        await memory_agent.start()
        
        # 测试存储记忆
        user_id = "test_user_001"
        
        # 存储SQL分析记录
        sql_analysis_content = """
        用户查询: SELECT * FROM users WHERE created_at > '2023-01-01'
        
        性能分析:
        - 执行时间: 2.5秒
        - 扫描行数: 1,000,000
        - 使用索引: 否
        
        优化建议:
        1. 在created_at字段上创建索引
        2. 考虑分页查询减少数据量
        3. 添加WHERE条件进一步过滤
        
        用户反馈: 建议很有用，性能提升明显
        """
        
        memory_id = await memory_agent.store_memory(
            user_id=user_id,
            content=sql_analysis_content,
            content_type="sql_analysis",
            metadata={
                "query_type": "SELECT",
                "table": "users",
                "performance_issue": "missing_index"
            },
            importance_score=0.8
        )
        
        print(f"存储记忆ID: {memory_id}")
        
        # 搜索相关记忆
        search_results = await memory_agent.search_memories(
            user_id=user_id,
            query="索引优化建议",
            limit=5
        )
        
        print(f"搜索到 {len(search_results)} 个相关记忆:")
        for result in search_results:
            print(f"- 相似度: {result.similarity_score:.2f}")
            print(f"  内容: {result.memory_entry.content[:100]}...")
            print(f"  原因: {result.relevance_reason}")
            print()
        
        # 获取记忆统计
        stats = await memory_agent.get_memory_statistics(user_id)
        print(f"记忆统计: {stats}")
        
        # 停止Agent
        await memory_agent.stop()
    
    # 运行测试
    asyncio.run(test_memory_agent())