"""实体提取器."""

import re
from typing import List, Dict, Any, Optional
import jieba
from .models import Entity, EntityType
from .domain_dictionary import DomainDictionary


class EntityExtractor:
    """实体提取器类."""
    
    def __init__(self, domain_dict: Optional[DomainDictionary] = None):
        """初始化实体提取器."""
        self.domain_dict = domain_dict or DomainDictionary()
        self._init_patterns()
        
        # 初始化jieba分词
        jieba.initialize()
    
    def _init_patterns(self) -> None:
        """初始化实体识别模式."""
        self.patterns = {
            EntityType.TABLE_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*表\b',
                r'\b表\s*([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bupdate\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\binsert\s+into\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            ],
            EntityType.COLUMN_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*字段\b',
                r'\b字段\s*([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bselect\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bwhere\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]',
                r'\border\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bgroup\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            ],
            EntityType.DATABASE_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*数据库\b',
                r'\b数据库\s*([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\buse\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            ],
            EntityType.INDEX_NAME: [
                r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*索引\b',
                r'\b索引\s*([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bcreate\s+index\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bdrop\s+index\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            ],
            EntityType.PERFORMANCE_METRIC: [
                r'\b(响应时间|执行时间|耗时)\b',
                r'\b(吞吐量|TPS|QPS)\b',
                r'\b(CPU使用率|内存使用率|磁盘使用率)\b',
                r'\b(连接数|并发数)\b',
            ],
            EntityType.TIME_PERIOD: [
                r'\b(\d+)\s*(秒|分钟|小时|天|周|月)\b',
                r'\b(今天|昨天|本周|上周|本月|上月)\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',  # 日期格式
                r'\b(\d{2}:\d{2}:\d{2})\b',  # 时间格式
            ],
            EntityType.THRESHOLD_VALUE: [
                r'\b(\d+(?:\.\d+)?)\s*(ms|秒|%|MB|GB|TB)\b',
                r'\b(超过|大于|小于|等于)\s*(\d+(?:\.\d+)?)\b',
            ],
            EntityType.OPERATION_TYPE: [
                r'\b(查询|插入|更新|删除|创建|删除)\b',
                r'\b(select|insert|update|delete|create|drop)\b',
                r'\b(优化|分析|监控|告警)\b',
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取实体."""
        entities = []
        
        # 使用正则表达式提取实体
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # 获取匹配的组
                    groups = match.groups()
                    if groups:
                        entity_text = groups[0] if len(groups) == 1 else groups[-1]
                    else:
                        entity_text = match.group(0)
                    
                    # 计算置信度
                    confidence = self._calculate_confidence(entity_text, entity_type, text)
                    
                    # 标准化实体值
                    normalized_value = self._normalize_entity(entity_text, entity_type)
                    
                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        normalized_value=normalized_value
                    )
                    entities.append(entity)
        
        # 使用领域词典增强实体识别
        entities.extend(self._extract_domain_entities(text))
        
        # 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)
        
        return entities
    
    def _extract_domain_entities(self, text: str) -> List[Entity]:
        """使用领域词典提取实体."""
        entities = []
        
        # 分词
        tokens = list(jieba.cut(text))
        current_pos = 0
        
        for token in tokens:
            # 查找token在原文中的位置
            start_pos = text.find(token, current_pos)
            if start_pos == -1:
                current_pos += len(token)
                continue
            
            end_pos = start_pos + len(token)
            current_pos = end_pos
            
            # 检查是否为领域术语
            domain_term = self.domain_dict.get_term(token)
            if domain_term:
                entity_type = self._map_category_to_entity_type(domain_term.category)
                if entity_type:
                    entity = Entity(
                        text=token,
                        entity_type=entity_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=min(0.9, domain_term.weight * 0.8),
                        normalized_value=domain_term.term,
                        metadata={"category": domain_term.category}
                    )
                    entities.append(entity)
        
        return entities
    
    def _map_category_to_entity_type(self, category: str) -> Optional[EntityType]:
        """将领域词典类别映射到实体类型."""
        mapping = {
            "sql_operation": EntityType.OPERATION_TYPE,
            "performance": EntityType.PERFORMANCE_METRIC,
            "database_object": EntityType.TABLE_NAME,  # 默认映射，可能需要更精确的判断
            "metric": EntityType.PERFORMANCE_METRIC,
            "operation": EntityType.OPERATION_TYPE,
        }
        return mapping.get(category)
    
    def _calculate_confidence(self, entity_text: str, entity_type: EntityType, context: str) -> float:
        """计算实体识别的置信度."""
        base_confidence = 0.7
        
        # 根据实体类型调整置信度
        type_weights = {
            EntityType.TABLE_NAME: 0.8,
            EntityType.COLUMN_NAME: 0.7,
            EntityType.DATABASE_NAME: 0.9,
            EntityType.INDEX_NAME: 0.8,
            EntityType.PERFORMANCE_METRIC: 0.9,
            EntityType.TIME_PERIOD: 0.95,
            EntityType.THRESHOLD_VALUE: 0.9,
            EntityType.OPERATION_TYPE: 0.8,
        }
        
        confidence = base_confidence * type_weights.get(entity_type, 0.7)
        
        # 根据上下文调整置信度
        if entity_type in [EntityType.TABLE_NAME, EntityType.COLUMN_NAME]:
            # 如果在SQL关键词附近，提高置信度
            sql_keywords = ['select', 'from', 'where', 'join', 'update', 'insert', 'delete']
            context_lower = context.lower()
            if any(keyword in context_lower for keyword in sql_keywords):
                confidence = min(0.95, confidence * 1.2)
        
        # 根据实体长度调整置信度
        if len(entity_text) < 2:
            confidence *= 0.8
        elif len(entity_text) > 20:
            confidence *= 0.9
        
        return min(0.99, max(0.1, confidence))
    
    def _normalize_entity(self, entity_text: str, entity_type: EntityType) -> Optional[str]:
        """标准化实体值."""
        if entity_type == EntityType.OPERATION_TYPE:
            # 标准化操作类型
            operation_mapping = {
                "查询": "SELECT",
                "插入": "INSERT", 
                "更新": "UPDATE",
                "删除": "DELETE",
                "创建": "CREATE",
                "select": "SELECT",
                "insert": "INSERT",
                "update": "UPDATE", 
                "delete": "DELETE",
                "create": "CREATE",
                "drop": "DROP"
            }
            return operation_mapping.get(entity_text.lower(), entity_text.upper())
        
        elif entity_type == EntityType.PERFORMANCE_METRIC:
            # 标准化性能指标
            metric_mapping = {
                "响应时间": "response_time",
                "执行时间": "execution_time",
                "耗时": "duration",
                "吞吐量": "throughput",
                "CPU使用率": "cpu_usage",
                "内存使用率": "memory_usage"
            }
            return metric_mapping.get(entity_text, entity_text)
        
        elif entity_type in [EntityType.TABLE_NAME, EntityType.COLUMN_NAME, EntityType.DATABASE_NAME, EntityType.INDEX_NAME]:
            # 数据库对象名称通常保持原样，但可以转换为小写
            return entity_text.lower()
        
        return entity_text
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去除重复的实体."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 创建唯一标识符
            key = (entity.text, entity.entity_type, entity.start_pos, entity.end_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # 如果位置相同但置信度更高，则替换
                for i, existing in enumerate(unique_entities):
                    if (existing.text == entity.text and 
                        existing.entity_type == entity.entity_type and
                        existing.start_pos == entity.start_pos and
                        existing.end_pos == entity.end_pos and
                        entity.confidence > existing.confidence):
                        unique_entities[i] = entity
                        break
        
        return unique_entities
    
    def extract_sql_entities(self, sql: str) -> List[Entity]:
        """专门从SQL语句中提取实体."""
        entities = []
        
        # SQL特定的实体提取模式
        sql_patterns = {
            EntityType.TABLE_NAME: [
                r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
                r'\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            ],
            EntityType.COLUMN_NAME: [
                r'\bSELECT\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s+FROM',
                r'\bWHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]',
                r'\bORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                r'\bGROUP\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            ]
        }
        
        for entity_type, patterns in sql_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sql, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1)
                    
                    # 处理多个列名的情况
                    if entity_type == EntityType.COLUMN_NAME and ',' in entity_text:
                        columns = [col.strip() for col in entity_text.split(',')]
                        for col in columns:
                            if col:
                                entity = Entity(
                                    text=col,
                                    entity_type=entity_type,
                                    start_pos=match.start(1),
                                    end_pos=match.end(1),
                                    confidence=0.9,
                                    normalized_value=col.lower()
                                )
                                entities.append(entity)
                    else:
                        entity = Entity(
                            text=entity_text,
                            entity_type=entity_type,
                            start_pos=match.start(1),
                            end_pos=match.end(1),
                            confidence=0.9,
                            normalized_value=entity_text.lower()
                        )
                        entities.append(entity)
        
        return entities