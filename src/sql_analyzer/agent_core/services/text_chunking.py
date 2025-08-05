"""文本分块和向量化处理模块 - 为Agent提供智能文本分块和嵌入服务."""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """文本块数据结构."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    start_pos: int = 0
    end_pos: int = 0
    chunk_type: str = "general"
    parent_id: Optional[str] = None
    overlap_with: List[str] = None


class ChunkingStrategy(ABC):
    """文本分块策略抽象基类."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """将文本分块."""
        pass


class SemanticChunker(ChunkingStrategy):
    """语义分块器 - 基于语义边界进行分块."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap_size: int = 50,
                 min_chunk_size: int = 100):
        """
        初始化语义分块器.
        
        Args:
            chunk_size: 目标块大小(tokens)
            overlap_size: 重叠大小(tokens)
            min_chunk_size: 最小块大小(tokens)
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # 语义边界标识符
        self.sentence_endings = re.compile(r'[.!?。！？]\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.section_breaks = re.compile(r'\n#+\s+')
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """基于语义边界分块."""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # 首先按段落分割
        paragraphs = self.paragraph_breaks.split(text)
        
        current_chunk = ""
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落超过大小限制
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    # 保存当前块
                    chunk = self._create_chunk(
                        current_chunk, 
                        current_pos, 
                        current_pos + len(current_chunk),
                        metadata
                    )
                    chunks.append(chunk)
                    
                    # 处理重叠
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                    current_chunk = overlap_text + paragraph
                    current_pos += len(current_chunk) - len(overlap_text)
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # 处理最后一个块
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk,
                current_pos,
                current_pos + len(current_chunk),
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, start_pos: int, end_pos: int, 
                     metadata: Dict[str, Any]) -> TextChunk:
        """创建文本块."""
        return TextChunk(
            chunk_id=str(uuid4()),
            content=content.strip(),
            metadata=metadata.copy(),
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_type="semantic",
            overlap_with=[]
        )
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """获取重叠文本."""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在句子边界处截取
        sentences = self.sentence_endings.split(text)
        if len(sentences) > 1:
            # 从末尾开始累积句子直到达到重叠大小
            overlap_text = ""
            for sentence in reversed(sentences[-2:]):  # 取最后两个句子
                if len(overlap_text) + len(sentence) <= overlap_size:
                    overlap_text = sentence + overlap_text
                else:
                    break
            if overlap_text:
                return overlap_text
        
        # 如果无法按句子分割，直接截取
        return text[-overlap_size:]


class ConversationChunker(ChunkingStrategy):
    """对话分块器 - 专门处理对话和交互历史."""
    
    def __init__(self, 
                 turns_per_chunk: int = 5,
                 overlap_turns: int = 1,
                 max_chunk_size: int = 1024):
        """
        初始化对话分块器.
        
        Args:
            turns_per_chunk: 每个块包含的对话轮次
            overlap_turns: 重叠的对话轮次
            max_chunk_size: 最大块大小
        """
        self.turns_per_chunk = turns_per_chunk
        self.overlap_turns = overlap_turns
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """按对话轮次分块."""
        metadata = metadata or {}
        
        # 识别对话轮次（假设用特定格式标记）
        turn_pattern = re.compile(r'(用户|User|系统|System|Agent)[:：]\s*')
        turns = turn_pattern.split(text)
        
        if len(turns) < 2:
            # 如果没有明确的对话标记，按段落处理
            return SemanticChunker().chunk_text(text, metadata)
        
        chunks = []
        current_turns = []
        
        # 重新组合对话轮次
        for i in range(1, len(turns), 2):
            if i + 1 < len(turns):
                speaker = turns[i].strip()
                content = turns[i + 1].strip()
                current_turns.append(f"{speaker}: {content}")
                
                # 检查是否需要创建新块
                if (len(current_turns) >= self.turns_per_chunk or 
                    len('\n'.join(current_turns)) > self.max_chunk_size):
                    
                    chunk_content = '\n'.join(current_turns)
                    chunk = TextChunk(
                        chunk_id=str(uuid4()),
                        content=chunk_content,
                        metadata={**metadata, "chunk_type": "conversation"},
                        chunk_type="conversation"
                    )
                    chunks.append(chunk)
                    
                    # 保留重叠轮次
                    if self.overlap_turns > 0:
                        current_turns = current_turns[-self.overlap_turns:]
                    else:
                        current_turns = []
        
        # 处理剩余轮次
        if current_turns:
            chunk_content = '\n'.join(current_turns)
            chunk = TextChunk(
                chunk_id=str(uuid4()),
                content=chunk_content,
                metadata={**metadata, "chunk_type": "conversation"},
                chunk_type="conversation"
            )
            chunks.append(chunk)
        
        return chunks


class SQLAnalysisChunker(ChunkingStrategy):
    """SQL分析结果分块器 - 专门处理SQL分析和优化建议."""
    
    def __init__(self, chunk_by_section: bool = True):
        """
        初始化SQL分析分块器.
        
        Args:
            chunk_by_section: 是否按分析部分分块
        """
        self.chunk_by_section = chunk_by_section
        
        # SQL分析部分标识符
        self.section_patterns = {
            'query': re.compile(r'(SQL查询|Query|查询语句)[:：]\s*', re.IGNORECASE),
            'analysis': re.compile(r'(性能分析|Analysis|分析结果)[:：]\s*', re.IGNORECASE),
            'suggestions': re.compile(r'(优化建议|Suggestions|建议)[:：]\s*', re.IGNORECASE),
            'explanation': re.compile(r'(执行计划|Execution Plan|解释)[:：]\s*', re.IGNORECASE)
        }
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """按SQL分析部分分块."""
        metadata = metadata or {}
        
        if not self.chunk_by_section:
            return SemanticChunker().chunk_text(text, metadata)
        
        chunks = []
        sections = self._identify_sections(text)
        
        for section_type, content in sections.items():
            if content.strip():
                chunk = TextChunk(
                    chunk_id=str(uuid4()),
                    content=content.strip(),
                    metadata={
                        **metadata, 
                        "section_type": section_type,
                        "chunk_type": "sql_analysis"
                    },
                    chunk_type="sql_analysis"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """识别SQL分析的不同部分."""
        sections = {}
        current_section = "general"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # 检查是否是新的部分标题
            section_found = None
            for section_type, pattern in self.section_patterns.items():
                if pattern.search(line):
                    section_found = section_type
                    break
            
            if section_found:
                # 保存当前部分
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # 开始新部分
                current_section = section_found
                current_content = [line]
            else:
                current_content.append(line)
        
        # 保存最后一个部分
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections


class TextEmbedder:
    """文本嵌入器 - 将文本转换为向量表示."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化文本嵌入器.
        
        Args:
            model_name: 使用的嵌入模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"成功加载嵌入模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            # 使用备用模型或简单的词向量
            self.model = None
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """为文本块生成嵌入向量."""
        if not self.model:
            logger.warning("嵌入模型未加载，跳过向量化")
            return chunks
        
        try:
            # 提取文本内容
            texts = [chunk.content for chunk in chunks]
            
            # 生成嵌入向量
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # 将嵌入向量添加到块中
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            logger.info(f"成功为 {len(chunks)} 个文本块生成嵌入向量")
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
        
        return chunks
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """为单个文本生成嵌入向量."""
        if not self.model:
            return None
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"生成文本嵌入失败: {e}")
            return None


class ChunkingManager:
    """分块管理器 - 统一管理不同类型的文本分块."""
    
    def __init__(self):
        """初始化分块管理器."""
        self.strategies = {
            "semantic": SemanticChunker(),
            "conversation": ConversationChunker(),
            "sql_analysis": SQLAnalysisChunker()
        }
        self.embedder = TextEmbedder()
    
    def chunk_and_embed(self, 
                       text: str, 
                       strategy: str = "semantic",
                       metadata: Dict[str, Any] = None,
                       embed: bool = True) -> List[TextChunk]:
        """
        分块并生成嵌入向量.
        
        Args:
            text: 要分块的文本
            strategy: 分块策略
            metadata: 元数据
            embed: 是否生成嵌入向量
            
        Returns:
            分块结果列表
        """
        if strategy not in self.strategies:
            logger.warning(f"未知的分块策略: {strategy}，使用默认策略")
            strategy = "semantic"
        
        # 执行分块
        chunker = self.strategies[strategy]
        chunks = chunker.chunk_text(text, metadata)
        
        # 生成嵌入向量
        if embed and chunks:
            chunks = self.embedder.embed_chunks(chunks)
        
        return chunks
    
    def add_strategy(self, name: str, strategy: ChunkingStrategy):
        """添加自定义分块策略."""
        self.strategies[name] = strategy
        logger.info(f"添加分块策略: {name}")


# 使用示例和测试
if __name__ == "__main__":
    # 测试不同的分块策略
    manager = ChunkingManager()
    
    # 测试语义分块
    knowledge_text = """
    数据库索引优化是提高查询性能的重要手段。
    
    B-Tree索引适用于范围查询和等值查询。它的结构特点是平衡树，
    能够保证查询、插入、删除操作的时间复杂度为O(log n)。
    
    Hash索引适用于等值查询，但不支持范围查询。它通过哈希函数
    将键值映射到存储位置，查询速度非常快。
    
    在选择索引类型时，需要考虑查询模式、数据分布和维护成本。
    """
    
    chunks = manager.chunk_and_embed(
        knowledge_text, 
        strategy="semantic",
        metadata={"type": "knowledge", "category": "index_optimization"}
    )
    
    print(f"语义分块结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"块 {i+1}: {chunk.content[:100]}...")
        print(f"嵌入向量维度: {len(chunk.embedding) if chunk.embedding else 0}")
        print("---")