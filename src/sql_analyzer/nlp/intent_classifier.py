"""意图分类器."""

import re
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import jieba
from ..agent_core.models import IntentType
from .models import IntentPattern
from .domain_dictionary import DomainDictionary


class IntentClassifier:
    """意图分类器类."""
    
    def __init__(self, domain_dict: DomainDictionary = None):
        """初始化意图分类器."""
        self.domain_dict = domain_dict or DomainDictionary()
        self._init_patterns()
        self._init_ml_classifier()
    
    def _init_patterns(self) -> None:
        """初始化意图识别模式."""
        self.intent_patterns = [
            # 查询分析相关
            IntentPattern(
                pattern=r"(为什么|怎么|如何).*(慢|性能|优化)",
                intent_type=IntentType.QUERY_ANALYSIS,
                keywords=["慢查询", "性能", "优化", "分析", "为什么"],
                weight=1.5
            ),
            IntentPattern(
                pattern=r"(分析|查看|检查).*(查询|SQL|语句)",
                intent_type=IntentType.QUERY_ANALYSIS,
                keywords=["分析", "查询", "SQL", "语句", "检查"],
                weight=1.3
            ),
            IntentPattern(
                pattern=r"(执行计划|explain|性能分析)",
                intent_type=IntentType.QUERY_ANALYSIS,
                keywords=["执行计划", "explain", "性能分析"],
                weight=1.4
            ),
            
            # 优化请求相关
            IntentPattern(
                pattern=r"(优化|改进|提升).*(性能|速度|效率)",
                intent_type=IntentType.OPTIMIZATION_REQUEST,
                keywords=["优化", "改进", "提升", "性能", "速度"],
                weight=1.5
            ),
            IntentPattern(
                pattern=r"(创建|添加|建立).*(索引|index)",
                intent_type=IntentType.OPTIMIZATION_REQUEST,
                keywords=["创建", "索引", "index", "添加"],
                weight=1.4
            ),
            IntentPattern(
                pattern=r"(自动|帮我|请).*(优化|修复|改善)",
                intent_type=IntentType.OPTIMIZATION_REQUEST,
                keywords=["自动", "帮我", "优化", "修复"],
                weight=1.3
            ),
            
            # 监控设置相关
            IntentPattern(
                pattern=r"(监控|监视|观察).*(性能|状态|指标)",
                intent_type=IntentType.MONITORING_SETUP,
                keywords=["监控", "监视", "性能", "状态", "指标"],
                weight=1.4
            ),
            IntentPattern(
                pattern=r"(设置|配置|启动).*(告警|警报|通知)",
                intent_type=IntentType.MONITORING_SETUP,
                keywords=["设置", "告警", "警报", "通知", "配置"],
                weight=1.3
            ),
            IntentPattern(
                pattern=r"(实时|持续).*(监控|检测)",
                intent_type=IntentType.MONITORING_SETUP,
                keywords=["实时", "持续", "监控", "检测"],
                weight=1.2
            ),
            
            # 知识查询相关
            IntentPattern(
                pattern=r"(什么是|解释|说明).*(索引|查询|数据库)",
                intent_type=IntentType.KNOWLEDGE_QUERY,
                keywords=["什么是", "解释", "说明", "索引", "数据库"],
                weight=1.3
            ),
            IntentPattern(
                pattern=r"(如何|怎样|怎么).*(使用|操作|配置)",
                intent_type=IntentType.KNOWLEDGE_QUERY,
                keywords=["如何", "怎样", "使用", "操作", "配置"],
                weight=1.2
            ),
            IntentPattern(
                pattern=r"(最佳实践|建议|推荐)",
                intent_type=IntentType.KNOWLEDGE_QUERY,
                keywords=["最佳实践", "建议", "推荐"],
                weight=1.1
            ),
            
            # 帮助请求相关
            IntentPattern(
                pattern=r"(帮助|help|使用说明|指南)",
                intent_type=IntentType.HELP_REQUEST,
                keywords=["帮助", "help", "使用说明", "指南"],
                weight=1.5
            ),
            IntentPattern(
                pattern=r"(不知道|不会|不懂).*(怎么|如何)",
                intent_type=IntentType.HELP_REQUEST,
                keywords=["不知道", "不会", "怎么", "如何"],
                weight=1.2
            ),
            IntentPattern(
                pattern=r"(可以|能够|支持).*(什么|哪些|功能)",
                intent_type=IntentType.HELP_REQUEST,
                keywords=["可以", "能够", "支持", "功能"],
                weight=1.1
            )
        ]
    
    def _init_ml_classifier(self) -> None:
        """初始化机器学习分类器."""
        # 创建训练数据
        training_data = self._create_training_data()
        
        if training_data:
            texts, labels = zip(*training_data)
            
            # 创建分类管道
            self.ml_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(
                    tokenizer=self._tokenize_chinese,
                    lowercase=True,
                    max_features=1000,
                    ngram_range=(1, 2)
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            # 训练分类器
            self.ml_classifier.fit(texts, labels)
        else:
            self.ml_classifier = None
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词."""
        return list(jieba.cut(text))
    
    def _create_training_data(self) -> List[Tuple[str, str]]:
        """创建训练数据."""
        training_data = [
            # 查询分析
            ("为什么我的查询这么慢？", IntentType.QUERY_ANALYSIS),
            ("这个SQL语句性能怎么样？", IntentType.QUERY_ANALYSIS),
            ("分析一下这个慢查询", IntentType.QUERY_ANALYSIS),
            ("查看执行计划", IntentType.QUERY_ANALYSIS),
            ("检查查询性能", IntentType.QUERY_ANALYSIS),
            ("这个查询有什么问题？", IntentType.QUERY_ANALYSIS),
            ("explain这个SQL", IntentType.QUERY_ANALYSIS),
            
            # 优化请求
            ("帮我优化这个查询", IntentType.OPTIMIZATION_REQUEST),
            ("如何提升查询性能？", IntentType.OPTIMIZATION_REQUEST),
            ("创建索引来优化", IntentType.OPTIMIZATION_REQUEST),
            ("自动优化数据库", IntentType.OPTIMIZATION_REQUEST),
            ("改进查询速度", IntentType.OPTIMIZATION_REQUEST),
            ("请优化这个慢查询", IntentType.OPTIMIZATION_REQUEST),
            ("建议优化方案", IntentType.OPTIMIZATION_REQUEST),
            
            # 监控设置
            ("设置性能监控", IntentType.MONITORING_SETUP),
            ("监控数据库状态", IntentType.MONITORING_SETUP),
            ("配置告警规则", IntentType.MONITORING_SETUP),
            ("启动实时监控", IntentType.MONITORING_SETUP),
            ("观察系统性能", IntentType.MONITORING_SETUP),
            ("设置性能告警", IntentType.MONITORING_SETUP),
            ("持续监控数据库", IntentType.MONITORING_SETUP),
            
            # 知识查询
            ("什么是索引？", IntentType.KNOWLEDGE_QUERY),
            ("解释执行计划", IntentType.KNOWLEDGE_QUERY),
            ("如何使用这个功能？", IntentType.KNOWLEDGE_QUERY),
            ("数据库最佳实践", IntentType.KNOWLEDGE_QUERY),
            ("推荐优化策略", IntentType.KNOWLEDGE_QUERY),
            ("说明查询原理", IntentType.KNOWLEDGE_QUERY),
            ("怎样配置数据库？", IntentType.KNOWLEDGE_QUERY),
            
            # 帮助请求
            ("帮助", IntentType.HELP_REQUEST),
            ("使用说明", IntentType.HELP_REQUEST),
            ("不知道怎么操作", IntentType.HELP_REQUEST),
            ("支持哪些功能？", IntentType.HELP_REQUEST),
            ("可以做什么？", IntentType.HELP_REQUEST),
            ("操作指南", IntentType.HELP_REQUEST),
            ("help", IntentType.HELP_REQUEST),
        ]
        
        return training_data
    
    def classify_intent(self, text: str) -> Dict[str, float]:
        """分类用户意图."""
        intent_scores = {}
        
        # 基于规则的分类
        rule_scores = self._classify_by_rules(text)
        
        # 基于机器学习的分类
        ml_scores = self._classify_by_ml(text)
        
        # 合并分数
        all_intents = set(rule_scores.keys()) | set(ml_scores.keys())
        
        for intent in all_intents:
            rule_score = rule_scores.get(intent, 0.0)
            ml_score = ml_scores.get(intent, 0.0)
            
            # 加权平均，规则权重更高
            combined_score = 0.6 * rule_score + 0.4 * ml_score
            intent_scores[intent] = combined_score
        
        # 确保所有意图类型都有分数
        for intent_type in IntentType:
            if intent_type not in intent_scores:
                intent_scores[intent_type] = 0.0
        
        # 归一化分数
        total_score = sum(intent_scores.values())
        if total_score > 0:
            intent_scores = {k: v / total_score for k, v in intent_scores.items()}
        
        return intent_scores
    
    def _classify_by_rules(self, text: str) -> Dict[str, float]:
        """基于规则的意图分类."""
        scores = {}
        text_lower = text.lower()
        
        for pattern in self.intent_patterns:
            # 正则匹配
            if re.search(pattern.pattern, text, re.IGNORECASE):
                scores[pattern.intent_type] = scores.get(pattern.intent_type, 0) + pattern.weight
            
            # 关键词匹配
            keyword_matches = sum(1 for keyword in pattern.keywords if keyword in text_lower)
            if keyword_matches > 0:
                keyword_score = (keyword_matches / len(pattern.keywords)) * pattern.weight * 0.8
                scores[pattern.intent_type] = scores.get(pattern.intent_type, 0) + keyword_score
        
        # 使用领域词典增强分类
        domain_score = self._get_domain_score(text)
        for intent, score in domain_score.items():
            scores[intent] = scores.get(intent, 0) + score
        
        return scores
    
    def _classify_by_ml(self, text: str) -> Dict[str, float]:
        """基于机器学习的意图分类."""
        if not self.ml_classifier:
            return {}
        
        try:
            # 获取预测概率
            probabilities = self.ml_classifier.predict_proba([text])[0]
            classes = self.ml_classifier.classes_
            
            return dict(zip(classes, probabilities))
        except Exception:
            return {}
    
    def _get_domain_score(self, text: str) -> Dict[str, float]:
        """基于领域词典计算意图分数."""
        scores = {}
        
        # 分词
        tokens = list(jieba.cut(text))
        
        for token in tokens:
            domain_term = self.domain_dict.get_term(token)
            if domain_term:
                # 根据术语类别推断可能的意图
                if domain_term.category == "performance":
                    scores[IntentType.QUERY_ANALYSIS] = scores.get(IntentType.QUERY_ANALYSIS, 0) + 0.3
                    scores[IntentType.MONITORING_SETUP] = scores.get(IntentType.MONITORING_SETUP, 0) + 0.2
                elif domain_term.category == "operation":
                    if "优化" in domain_term.synonyms or "optimize" in domain_term.synonyms:
                        scores[IntentType.OPTIMIZATION_REQUEST] = scores.get(IntentType.OPTIMIZATION_REQUEST, 0) + 0.4
                    elif "监控" in domain_term.synonyms or "monitor" in domain_term.synonyms:
                        scores[IntentType.MONITORING_SETUP] = scores.get(IntentType.MONITORING_SETUP, 0) + 0.4
                    else:
                        scores[IntentType.QUERY_ANALYSIS] = scores.get(IntentType.QUERY_ANALYSIS, 0) + 0.2
                elif domain_term.category == "database_object":
                    scores[IntentType.QUERY_ANALYSIS] = scores.get(IntentType.QUERY_ANALYSIS, 0) + 0.2
                    scores[IntentType.OPTIMIZATION_REQUEST] = scores.get(IntentType.OPTIMIZATION_REQUEST, 0) + 0.1
        
        return scores
    
    def get_top_intent(self, text: str) -> Tuple[str, float]:
        """获取最可能的意图."""
        scores = self.classify_intent(text)
        
        if not scores:
            return IntentType.UNKNOWN, 0.0
        
        # 过滤出有效的意图类型
        valid_scores = {}
        for intent, score in scores.items():
            # 确保intent是有效的IntentType
            if isinstance(intent, str):
                try:
                    # 尝试转换为IntentType
                    if intent.startswith('IntentType.'):
                        # 处理类似 'IntentType.QUERY_ANA' 的情况
                        intent_name = intent.replace('IntentType.', '')
                        # 尝试匹配完整的意图名称
                        for valid_intent in IntentType:
                            if valid_intent.value.upper().startswith(intent_name.upper()):
                                valid_scores[valid_intent.value] = score
                                break
                    else:
                        # 直接检查是否是有效的意图值
                        IntentType(intent)  # 验证是否有效
                        valid_scores[intent] = score
                except ValueError:
                    continue
            elif hasattr(intent, 'value'):
                valid_scores[intent.value] = score
        
        if not valid_scores:
            return IntentType.UNKNOWN, 0.0
        
        top_intent = max(valid_scores.items(), key=lambda x: x[1])
        return top_intent
    
    def add_training_example(self, text: str, intent: str) -> None:
        """添加训练样例（用于在线学习）."""
        # 这里可以实现在线学习逻辑
        # 暂时只是存储，实际应用中可以定期重训练模型
        pass
    
    def get_intent_confidence_threshold(self, intent_type: str) -> float:
        """获取意图的置信度阈值."""
        thresholds = {
            IntentType.QUERY_ANALYSIS: 0.3,
            IntentType.OPTIMIZATION_REQUEST: 0.4,
            IntentType.MONITORING_SETUP: 0.3,
            IntentType.KNOWLEDGE_QUERY: 0.2,
            IntentType.HELP_REQUEST: 0.2,
        }
        return thresholds.get(intent_type, 0.3)
    
    def is_confident_prediction(self, text: str) -> bool:
        """判断预测是否足够可信."""
        top_intent, confidence = self.get_top_intent(text)
        threshold = self.get_intent_confidence_threshold(top_intent)
        return confidence >= threshold