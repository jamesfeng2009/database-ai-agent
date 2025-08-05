# 多Agent系统中的文本分块和向量化最佳实践

## 概述

在多Agent系统中，文本分块和向量化是实现智能搜索和记忆管理的关键技术。本文档总结了针对不同Agent和内容类型的分块策略和最佳实践。

## 核心原则

### 1. 语义完整性
- 分块时保持语义单元的完整性
- 避免在句子中间或重要概念中间切分
- 使用语义边界（段落、句子、章节）作为分块点

### 2. 上下文连续性
- 设置适当的重叠（overlap）保持上下文连续性
- 重叠大小通常为块大小的10-20%
- 对于对话类内容，重叠应包含完整的对话轮次

### 3. 内容类型适配
- 不同类型的内容使用不同的分块策略
- 结构化内容（如SQL分析）按逻辑部分分块
- 对话内容按轮次或时间窗口分块

## 各Agent的分块策略

### Knowledge Agent

**内容特点：**
- 结构化的知识条目
- 包含标题、描述、解决步骤等
- 相对静态，更新频率低

**推荐策略：**
```python
# 语义分块，保持知识条目的逻辑完整性
chunking_strategy = "semantic"
chunk_size = 512  # tokens
overlap_size = 50  # tokens
min_chunk_size = 100  # tokens
```

**分块要点：**
- 按知识条目的逻辑结构分块
- 标题和描述通常作为一个块
- 解决步骤可以分组分块
- 为每个知识条目生成主要的嵌入向量

### Memory Agent

**内容特点：**
- 多样化的用户交互记录
- 包含SQL分析、对话、反馈等
- 动态增长，需要高效检索

**推荐策略：**
```python
# 根据内容类型选择策略
content_type_strategies = {
    "sql_analysis": "sql_analysis",    # 按分析部分分块
    "conversation": "conversation",    # 按对话轮次分块
    "user_feedback": "semantic",       # 语义分块
    "general": "semantic"              # 默认语义分块
}
```

**分块要点：**
- SQL分析按查询、分析、建议等部分分块
- 对话按轮次分块，保持对话的连贯性
- 用户反馈作为完整单元，避免拆分
- 支持长短期记忆的分层管理

### Learning Agent

**内容特点：**
- 用户反馈和行为数据
- 模式发现结果
- 学习统计信息

**推荐策略：**
```python
# 主要处理结构化数据，分块需求较少
# 重点是数据聚合和模式识别
chunking_strategy = "semantic"  # 用于文本反馈
chunk_size = 256  # 较小的块大小
```

## 分块参数配置

### 通用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| chunk_size | 512-1024 tokens | 根据内容复杂度调整 |
| overlap_size | 50-100 tokens | 保持上下文连续性 |
| min_chunk_size | 100 tokens | 避免过小的无意义块 |

### 内容类型特定参数

#### SQL分析内容
```python
sql_analysis_config = {
    "chunk_by_section": True,
    "sections": ["query", "analysis", "suggestions", "explanation"],
    "preserve_code_blocks": True
}
```

#### 对话内容
```python
conversation_config = {
    "turns_per_chunk": 5,
    "overlap_turns": 1,
    "max_chunk_size": 1024,
    "preserve_speaker_info": True
}
```

#### 知识文档
```python
knowledge_config = {
    "chunk_size": 512,
    "overlap_size": 50,
    "respect_paragraph_boundaries": True,
    "preserve_headings": True
}
```

## 向量化最佳实践

### 模型选择

**推荐模型：**
- `all-MiniLM-L6-v2`: 轻量级，适合大多数场景
- `all-mpnet-base-v2`: 更高质量，计算成本较高
- `paraphrase-multilingual-MiniLM-L12-v2`: 多语言支持

### 嵌入策略

```python
class EmbeddingStrategy:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.batch_size = 32
        self.normalize_embeddings = True
    
    def embed_chunks(self, chunks):
        # 批量处理提高效率
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings
```

### 相似性搜索优化

```python
def optimized_similarity_search(query_embedding, stored_embeddings, top_k=10):
    # 使用向量化操作提高搜索效率
    similarities = np.dot(stored_embeddings, query_embedding)
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
    return top_indices, similarities[top_indices]
```

## 存储和索引策略

### 向量数据库选择

**ChromaDB（推荐）：**
- 轻量级，易于集成
- 支持元数据过滤
- 本地持久化存储

```python
# ChromaDB配置示例
chroma_config = {
    "persist_directory": "./vector_db",
    "collection_metadata": {"hnsw:space": "cosine"},
    "embedding_function": sentence_transformer_ef
}
```

**FAISS（高性能场景）：**
- 更高的搜索性能
- 支持GPU加速
- 适合大规模数据

### 索引策略

```python
class HybridIndexStrategy:
    def __init__(self):
        # 向量索引用于语义搜索
        self.vector_index = ChromaCollection()
        # 关键词索引用于精确匹配
        self.keyword_index = InvertedIndex()
        # 元数据索引用于过滤
        self.metadata_index = BTreeIndex()
    
    def search(self, query, filters=None):
        # 混合搜索：向量 + 关键词 + 元数据过滤
        vector_results = self.vector_index.query(query)
        keyword_results = self.keyword_index.search(query)
        
        # 结果融合和重排序
        return self.merge_and_rerank(vector_results, keyword_results, filters)
```

## 性能优化建议

### 1. 批量处理
- 批量生成嵌入向量，减少模型调用次数
- 批量插入向量数据库，提高写入效率

### 2. 缓存策略
```python
class EmbeddingCache:
    def __init__(self, max_size=10000):
        self.cache = LRUCache(max_size)
    
    def get_embedding(self, text):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        embedding = self.model.encode([text])[0]
        self.cache[text_hash] = embedding
        return embedding
```

### 3. 异步处理
```python
async def async_chunk_and_embed(texts, chunking_manager):
    # 异步分块
    chunk_tasks = [
        chunking_manager.chunk_text(text) 
        for text in texts
    ]
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # 异步向量化
    all_chunks = [chunk for chunks in chunk_results for chunk in chunks]
    embeddings = await async_embed_chunks(all_chunks)
    
    return all_chunks, embeddings
```

## 质量评估和监控

### 分块质量指标

```python
class ChunkingQualityMetrics:
    def evaluate_chunks(self, chunks):
        metrics = {
            "avg_chunk_size": np.mean([len(c.content) for c in chunks]),
            "size_variance": np.var([len(c.content) for c in chunks]),
            "semantic_coherence": self.calculate_coherence(chunks),
            "overlap_effectiveness": self.evaluate_overlap(chunks)
        }
        return metrics
    
    def calculate_coherence(self, chunks):
        # 计算块内语义一致性
        coherence_scores = []
        for chunk in chunks:
            sentences = sent_tokenize(chunk.content)
            if len(sentences) > 1:
                sentence_embeddings = self.embed_sentences(sentences)
                coherence = self.calculate_sentence_similarity(sentence_embeddings)
                coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
```

### 搜索质量监控

```python
class SearchQualityMonitor:
    def __init__(self):
        self.search_logs = []
        self.relevance_feedback = []
    
    def log_search(self, query, results, user_feedback=None):
        log_entry = {
            "timestamp": datetime.now(),
            "query": query,
            "results_count": len(results),
            "top_similarity": results[0].similarity_score if results else 0,
            "user_feedback": user_feedback
        }
        self.search_logs.append(log_entry)
    
    def calculate_search_metrics(self):
        if not self.search_logs:
            return {}
        
        return {
            "avg_results_count": np.mean([log["results_count"] for log in self.search_logs]),
            "avg_top_similarity": np.mean([log["top_similarity"] for log in self.search_logs]),
            "user_satisfaction": self.calculate_satisfaction_rate()
        }
```

## 常见问题和解决方案

### 1. 分块过大或过小
**问题：** 分块大小不合适影响搜索质量
**解决：** 
- 根据内容类型调整chunk_size
- 使用自适应分块策略
- 监控分块质量指标

### 2. 语义边界识别困难
**问题：** 难以准确识别语义边界
**解决：**
- 使用多种边界标识符（段落、句子、标点）
- 结合领域知识定制分块规则
- 使用机器学习方法识别语义边界

### 3. 向量搜索召回率低
**问题：** 相关内容无法被检索到
**解决：**
- 降低相似度阈值
- 使用混合搜索（向量+关键词）
- 优化查询向量生成策略

### 4. 存储空间占用过大
**问题：** 向量存储占用大量空间
**解决：**
- 使用向量压缩技术
- 定期清理低质量向量
- 实施分层存储策略

## 总结

文本分块和向量化是多Agent系统中实现智能搜索和记忆管理的核心技术。通过：

1. **选择合适的分块策略** - 根据内容类型和Agent需求
2. **优化分块参数** - 平衡语义完整性和搜索效率
3. **实施向量化最佳实践** - 选择合适的模型和存储方案
4. **持续监控和优化** - 评估质量指标并持续改进

可以构建高效、准确的智能搜索和记忆系统，为多Agent协作提供强大的知识管理能力。