"""NLP模块的数据模型."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """实体类型枚举."""
    TABLE_NAME = "table_name"
    COLUMN_NAME = "column_name"
    DATABASE_NAME = "database_name"
    INDEX_NAME = "index_name"
    SQL_KEYWORD = "sql_keyword"
    PERFORMANCE_METRIC = "performance_metric"
    TIME_PERIOD = "time_period"
    THRESHOLD_VALUE = "threshold_value"
    OPERATION_TYPE = "operation_type"


class Entity(BaseModel):
    """实体模型."""
    text: str = Field(..., description="实体文本")
    entity_type: EntityType = Field(..., description="实体类型")
    start_pos: int = Field(..., description="在原文中的起始位置")
    end_pos: int = Field(..., description="在原文中的结束位置")
    confidence: float = Field(..., description="置信度")
    normalized_value: Optional[str] = Field(None, description="标准化后的值")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class IntentPattern(BaseModel):
    """意图模式模型."""
    pattern: str = Field(..., description="匹配模式")
    intent_type: str = Field(..., description="意图类型")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    weight: float = Field(default=1.0, description="权重")
    required_entities: List[EntityType] = Field(default_factory=list, description="必需的实体类型")


class ProcessingResult(BaseModel):
    """NLP处理结果模型."""
    original_text: str = Field(..., description="原始文本")
    processed_text: str = Field(..., description="预处理后的文本")
    tokens: List[str] = Field(default_factory=list, description="分词结果")
    entities: List[Entity] = Field(default_factory=list, description="提取的实体")
    intent_scores: Dict[str, float] = Field(default_factory=dict, description="各意图的得分")
    confidence: float = Field(..., description="整体置信度")
    processing_time: float = Field(..., description="处理耗时（秒）")


class DomainTerm(BaseModel):
    """领域术语模型."""
    term: str = Field(..., description="术语")
    category: str = Field(..., description="类别")
    synonyms: List[str] = Field(default_factory=list, description="同义词")
    definition: str = Field(..., description="定义")
    examples: List[str] = Field(default_factory=list, description="使用示例")
    weight: float = Field(default=1.0, description="重要性权重")