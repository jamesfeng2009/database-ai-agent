"""自然语言处理器主模块."""

import time
from typing import Dict, List, Optional, Any
import jieba
import re
from ..agent_core.models import UserIntent, IntentType
from .models import ProcessingResult, Entity
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .domain_dictionary import DomainDictionary


class NLPProcessor:
    """自然语言处理器主类."""
    
    def __init__(self):
        """初始化NLP处理器."""
        self.domain_dict = DomainDictionary()
        self.intent_classifier = IntentClassifier(self.domain_dict)
        self.entity_extractor = EntityExtractor(self.domain_dict)
        
        # 初始化jieba分词
        jieba.initialize()
    
    async def process_text(self, text: str) -> ProcessingResult:
        """处理自然语言文本."""
        start_time = time.time()
        
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 分词
        tokens = self._tokenize(processed_text)
        
        # 提取实体
        entities = self.entity_extractor.extract_entities(processed_text)
        
        # 分类意图
        intent_scores = self.intent_classifier.classify_intent(processed_text)
        
        # 计算整体置信度
        confidence = self._calculate_overall_confidence(intent_scores, entities)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            original_text=text,
            processed_text=processed_text,
            tokens=tokens,
            entities=entities,
            intent_scores=intent_scores,
            confidence=confidence,
            processing_time=processing_time
        )
    
    async def extract_intent(self, user_input: str) -> UserIntent:
        """提取用户意图."""
        # 处理文本
        result = await self.process_text(user_input)
        
        # 获取最可能的意图
        top_intent, confidence = self.intent_classifier.get_top_intent(user_input)
        
        # 构建实体字典
        entities_dict = {}
        for entity in result.entities:
            entity_type = entity.entity_type.value
            if entity_type not in entities_dict:
                entities_dict[entity_type] = []
            entities_dict[entity_type].append({
                "text": entity.text,
                "normalized_value": entity.normalized_value,
                "confidence": entity.confidence,
                "position": (entity.start_pos, entity.end_pos)
            })
        
        # 提取参数
        parameters = self._extract_parameters(user_input, result.entities, top_intent)
        
        return UserIntent(
            intent_type=IntentType(top_intent),
            entities=entities_dict,
            confidence=confidence,
            parameters=parameters,
            raw_input=user_input
        )
    
    async def generate_sql_from_natural_language(self, description: str) -> str:
        """从自然语言生成SQL查询（基础实现）."""
        # 提取实体
        entities = self.entity_extractor.extract_entities(description)
        
        # 查找表名和列名
        tables = [e.normalized_value for e in entities if e.entity_type.value == "table_name"]
        columns = [e.normalized_value for e in entities if e.entity_type.value == "column_name"]
        
        # 简单的SQL生成逻辑
        if tables and columns:
            table = tables[0]
            column_list = ", ".join(columns) if columns else "*"
            return f"SELECT {column_list} FROM {table}"
        elif tables:
            return f"SELECT * FROM {tables[0]}"
        else:
            return "-- 无法从描述中提取足够信息生成SQL"
    
    async def explain_technical_terms(self, terms: List[str]) -> Dict[str, str]:
        """解释技术术语."""
        explanations = {}
        
        for term in terms:
            domain_term = self.domain_dict.get_term(term)
            if domain_term:
                explanations[term] = domain_term.definition
            else:
                # 尝试搜索相关术语
                related_terms = self.domain_dict.search_terms(term)
                if related_terms:
                    explanations[term] = related_terms[0].definition
                else:
                    explanations[term] = f"未找到术语 '{term}' 的定义"
        
        return explanations
    
    async def translate_response(self, response: str, target_language: str) -> str:
        """翻译响应（基础实现）."""
        # 这里可以集成翻译API，目前只是简单返回
        if target_language.lower() in ["en", "english"]:
            # 简单的中英文术语映射
            translations = {
                "查询": "query",
                "索引": "index", 
                "优化": "optimization",
                "性能": "performance",
                "数据库": "database",
                "表": "table",
                "字段": "column"
            }
            
            translated = response
            for chinese, english in translations.items():
                translated = translated.replace(chinese, english)
            
            return translated
        
        return response
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本."""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 统一标点符号
        text = text.replace('？', '?').replace('！', '!').replace('，', ',').replace('。', '.')
        
        # 处理SQL关键词大小写
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'UPDATE', 'INSERT', 'DELETE', 'CREATE', 'DROP']
        for keyword in sql_keywords:
            text = re.sub(rf'\b{keyword.lower()}\b', keyword, text, flags=re.IGNORECASE)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """分词."""
        # 使用jieba进行中文分词
        tokens = list(jieba.cut(text))
        
        # 过滤停用词和标点符号
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        tokens = [token.strip() for token in tokens if token.strip() and token not in stop_words and not re.match(r'^[^\w\s]+$', token)]
        
        return tokens
    
    def _calculate_overall_confidence(self, intent_scores: Dict[str, float], entities: List[Entity]) -> float:
        """计算整体置信度."""
        # 获取最高意图分数
        max_intent_score = max(intent_scores.values()) if intent_scores else 0.0
        
        # 计算实体置信度平均值
        entity_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.5
        
        # 综合计算
        overall_confidence = 0.6 * max_intent_score + 0.4 * entity_confidence
        
        return min(0.99, max(0.1, overall_confidence))
    
    def _extract_parameters(self, text: str, entities: List[Entity], intent_type: str) -> Dict[str, Any]:
        """根据意图类型提取参数."""
        parameters = {}
        
        # 根据不同意图类型提取不同参数
        if intent_type == IntentType.QUERY_ANALYSIS.value:
            # 查询分析相关参数
            parameters["analysis_type"] = self._detect_analysis_type(text)
            parameters["sql_statement"] = self._extract_sql_statement(text)
            parameters["performance_focus"] = self._detect_performance_focus(text)
            
        elif intent_type == IntentType.OPTIMIZATION_REQUEST.value:
            # 优化请求相关参数
            parameters["optimization_type"] = self._detect_optimization_type(text, entities)
            parameters["target_objects"] = self._extract_target_objects(entities)
            parameters["urgency_level"] = self._detect_urgency_level(text)
            
        elif intent_type == IntentType.MONITORING_SETUP.value:
            # 监控设置相关参数
            parameters["monitoring_type"] = self._detect_monitoring_type(text)
            parameters["metrics"] = self._extract_metrics(entities)
            parameters["thresholds"] = self._extract_thresholds(entities)
            
        elif intent_type == IntentType.KNOWLEDGE_QUERY.value:
            # 知识查询相关参数
            parameters["query_topic"] = self._extract_query_topic(text, entities)
            parameters["detail_level"] = self._detect_detail_level(text)
        
        return parameters
    
    def _detect_analysis_type(self, text: str) -> str:
        """检测分析类型."""
        if "执行计划" in text or "explain" in text.lower():
            return "execution_plan"
        elif "慢查询" in text or "slow" in text.lower():
            return "slow_query"
        elif "性能" in text or "performance" in text.lower():
            return "performance"
        else:
            return "general"
    
    def _extract_sql_statement(self, text: str) -> Optional[str]:
        """提取SQL语句."""
        # 查找SQL语句模式
        sql_pattern = r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+.*?(?=;|$|\n\n)'
        match = re.search(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(0).strip()
        
        return None
    
    def _detect_optimization_type(self, text: str, entities: List[Entity]) -> str:
        """检测优化类型."""
        if "索引" in text or "index" in text.lower():
            return "index"
        elif "查询" in text or "query" in text.lower():
            return "query"
        elif "配置" in text or "config" in text.lower():
            return "configuration"
        else:
            return "general"
    
    def _extract_target_objects(self, entities: List[Entity]) -> List[str]:
        """提取目标对象."""
        objects = []
        for entity in entities:
            if entity.entity_type.value in ["table_name", "column_name", "index_name"]:
                objects.append(entity.normalized_value or entity.text)
        return objects
    
    def _detect_monitoring_type(self, text: str) -> str:
        """检测监控类型."""
        if "实时" in text or "real-time" in text.lower():
            return "realtime"
        elif "告警" in text or "alert" in text.lower():
            return "alerting"
        elif "性能" in text or "performance" in text.lower():
            return "performance"
        else:
            return "general"
    
    def _extract_metrics(self, entities: List[Entity]) -> List[str]:
        """提取指标."""
        metrics = []
        for entity in entities:
            if entity.entity_type.value == "performance_metric":
                metrics.append(entity.normalized_value or entity.text)
        return metrics
    
    def _extract_thresholds(self, entities: List[Entity]) -> List[Dict[str, Any]]:
        """提取阈值."""
        thresholds = []
        for entity in entities:
            if entity.entity_type.value == "threshold_value":
                thresholds.append({
                    "value": entity.text,
                    "normalized": entity.normalized_value
                })
        return thresholds
    
    def _extract_query_topic(self, text: str, entities: List[Entity]) -> str:
        """提取查询主题."""
        # 从实体中提取主要主题
        topics = []
        for entity in entities:
            if entity.entity_type.value in ["database_object", "performance_metric", "operation_type"]:
                topics.append(entity.normalized_value or entity.text)
        
        if topics:
            return topics[0]
        
        # 从文本中提取关键词
        keywords = ["索引", "查询", "优化", "性能", "数据库", "表", "字段"]
        for keyword in keywords:
            if keyword in text:
                return keyword
        
        return "general"
    
    def _detect_detail_level(self, text: str) -> str:
        """检测详细程度."""
        if "详细" in text or "具体" in text or "深入" in text:
            return "detailed"
        elif "简单" in text or "概述" in text or "简要" in text:
            return "brief"
        else:
            return "normal"
    
    def _detect_performance_focus(self, text: str) -> str:
        """检测性能关注点."""
        if "慢" in text or "slow" in text.lower():
            return "slow_query"
        elif "索引" in text or "index" in text.lower():
            return "index_usage"
        elif "内存" in text or "memory" in text.lower():
            return "memory_usage"
        elif "CPU" in text or "cpu" in text.lower():
            return "cpu_usage"
        else:
            return "general"
    
    def _detect_urgency_level(self, text: str) -> str:
        """检测紧急程度."""
        if "紧急" in text or "urgent" in text.lower() or "立即" in text:
            return "urgent"
        elif "尽快" in text or "asap" in text.lower():
            return "high"
        elif "有时间" in text or "when possible" in text.lower():
            return "low"
        else:
            return "normal"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息."""
        return {
            "domain_dict_stats": self.domain_dict.get_statistics(),
            "supported_intents": [intent.value for intent in IntentType],
            "supported_entities": [entity_type.value for entity_type in self.entity_extractor.patterns.keys()]
        }