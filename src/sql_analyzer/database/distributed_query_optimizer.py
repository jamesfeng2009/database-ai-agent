"""分布式查询优化器 - 优化跨数据库查询的执行策略."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .cross_database_analyzer import (
    CrossDatabaseQuery, 
    CrossDatabaseAnalyzer, 
    PerformanceImpactAnalysis,
    QueryType
)
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """优化策略枚举."""
    PUSH_DOWN = "push_down"           # 谓词下推
    JOIN_REORDER = "join_reorder"     # 连接重排序
    PARALLEL_EXECUTION = "parallel"   # 并行执行
    DATA_LOCALITY = "data_locality"   # 数据本地化
    CACHING = "caching"               # 缓存策略
    MATERIALIZATION = "materialization"  # 物化视图


@dataclass
class OptimizationRule:
    """优化规则."""
    rule_id: str
    name: str
    strategy: OptimizationStrategy
    condition: str  # 应用条件
    priority: int   # 优先级
    estimated_improvement: float  # 预期改进百分比
    description: str


@dataclass
class QueryOptimizationPlan:
    """查询优化计划."""
    plan_id: str
    original_query: CrossDatabaseQuery
    applied_rules: List[OptimizationRule]
    optimized_execution_steps: List[Dict[str, Any]]
    estimated_cost_reduction: float
    estimated_time_reduction: float
    risk_assessment: Dict[str, float]
    implementation_complexity: str  # LOW, MEDIUM, HIGH


@dataclass
class OptimizationResult:
    """优化结果."""
    optimization_id: str
    original_cost: float
    optimized_cost: float
    actual_improvement: float
    execution_time_before: float
    execution_time_after: float
    success: bool
    error_message: Optional[str] = None


class DistributedQueryOptimizer:
    """分布式查询优化器."""
    
    def __init__(self, database_manager: DatabaseManager, cross_db_analyzer: CrossDatabaseAnalyzer):
        self.database_manager = database_manager
        self.cross_db_analyzer = cross_db_analyzer
        self.optimization_rules = []
        self.optimization_history = {}
        self.performance_cache = {}
        self._initialized = False
    
    async def initialize(self):
        """初始化优化器."""
        if self._initialized:
            return
        
        try:
            # 初始化依赖组件
            await self.database_manager.initialize()
            await self.cross_db_analyzer.initialize()
            
            # 加载优化规则
            await self._load_optimization_rules()
            
            self._initialized = True
            logger.info("分布式查询优化器初始化完成")
            
        except Exception as e:
            logger.error(f"分布式查询优化器初始化失败: {e}")
            raise
    
    async def optimize_query(self, query: CrossDatabaseQuery) -> QueryOptimizationPlan:
        """优化跨数据库查询."""
        try:
            # 分析查询特征
            query_features = await self._analyze_query_features(query)
            
            # 选择适用的优化规则
            applicable_rules = await self._select_optimization_rules(query, query_features)
            
            # 生成优化计划
            optimization_plan = await self._generate_optimization_plan(query, applicable_rules)
            
            # 评估优化效果
            await self._evaluate_optimization_plan(optimization_plan)
            
            return optimization_plan
            
        except Exception as e:
            logger.error(f"查询优化失败: {e}")
            raise
    
    async def execute_optimization_plan(self, plan: QueryOptimizationPlan) -> OptimizationResult:
        """执行优化计划."""
        try:
            optimization_id = f"opt_{plan.plan_id}_{datetime.now().timestamp()}"
            
            # 记录原始性能
            original_performance = await self._measure_query_performance(plan.original_query)
            
            # 应用优化规则
            optimized_query = await self._apply_optimization_rules(plan.original_query, plan.applied_rules)
            
            # 测量优化后性能
            optimized_performance = await self._measure_query_performance(optimized_query)
            
            # 计算实际改进
            actual_improvement = self._calculate_improvement(original_performance, optimized_performance)
            
            result = OptimizationResult(
                optimization_id=optimization_id,
                original_cost=original_performance.get("cost", 0.0),
                optimized_cost=optimized_performance.get("cost", 0.0),
                actual_improvement=actual_improvement,
                execution_time_before=original_performance.get("execution_time", 0.0),
                execution_time_after=optimized_performance.get("execution_time", 0.0),
                success=True
            )
            
            # 记录优化历史
            self.optimization_history[optimization_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"执行优化计划失败: {e}")
            return OptimizationResult(
                optimization_id=f"failed_{datetime.now().timestamp()}",
                original_cost=0.0,
                optimized_cost=0.0,
                actual_improvement=0.0,
                execution_time_before=0.0,
                execution_time_after=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def generate_optimization_suggestions(self, query: CrossDatabaseQuery) -> List[Dict[str, Any]]:
        """生成优化建议."""
        try:
            suggestions = []
            
            # 分析查询特征
            query_features = await self._analyze_query_features(query)
            
            # 基于查询类型的建议
            if query.query_type == QueryType.CROSS_DATABASE:
                suggestions.extend(await self._generate_cross_db_suggestions(query, query_features))
            elif query.query_type == QueryType.DISTRIBUTED:
                suggestions.extend(await self._generate_distributed_suggestions(query, query_features))
            
            # 基于性能分析的建议
            performance_analysis = await self.cross_db_analyzer.analyze_performance_impact(query)
            suggestions.extend(await self._generate_performance_based_suggestions(query, performance_analysis))
            
            # 基于依赖关系的建议
            suggestions.extend(await self._generate_dependency_based_suggestions(query))
            
            # 排序建议（按预期改进排序）
            suggestions.sort(key=lambda x: x.get("expected_improvement", 0), reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            return []
    
    async def _load_optimization_rules(self):
        """加载优化规则."""
        self.optimization_rules = [
            OptimizationRule(
                rule_id="push_down_predicates",
                name="谓词下推",
                strategy=OptimizationStrategy.PUSH_DOWN,
                condition="has_where_clause_and_cross_db_join",
                priority=1,
                estimated_improvement=0.3,
                description="将过滤条件推送到数据源，减少网络传输"
            ),
            OptimizationRule(
                rule_id="optimize_join_order",
                name="连接顺序优化",
                strategy=OptimizationStrategy.JOIN_REORDER,
                condition="multiple_joins_with_different_selectivity",
                priority=2,
                estimated_improvement=0.25,
                description="重新排序连接操作，优先执行高选择性的连接"
            ),
            OptimizationRule(
                rule_id="parallel_execution",
                name="并行执行",
                strategy=OptimizationStrategy.PARALLEL_EXECUTION,
                condition="independent_subqueries",
                priority=3,
                estimated_improvement=0.4,
                description="并行执行独立的子查询"
            ),
            OptimizationRule(
                rule_id="data_locality_optimization",
                name="数据本地化",
                strategy=OptimizationStrategy.DATA_LOCALITY,
                condition="frequent_cross_db_access",
                priority=4,
                estimated_improvement=0.5,
                description="将频繁访问的数据复制到本地"
            ),
            OptimizationRule(
                rule_id="result_caching",
                name="结果缓存",
                strategy=OptimizationStrategy.CACHING,
                condition="repeated_expensive_operations",
                priority=5,
                estimated_improvement=0.6,
                description="缓存昂贵操作的结果"
            ),
            OptimizationRule(
                rule_id="materialized_views",
                name="物化视图",
                strategy=OptimizationStrategy.MATERIALIZATION,
                condition="complex_aggregations_on_large_datasets",
                priority=6,
                estimated_improvement=0.7,
                description="为复杂聚合创建物化视图"
            )
        ]
    
    async def _analyze_query_features(self, query: CrossDatabaseQuery) -> Dict[str, Any]:
        """分析查询特征."""
        features = {
            "query_type": query.query_type.value,
            "database_count": len(query.involved_databases),
            "dependency_count": len(query.dependencies),
            "estimated_cost": query.estimated_cost,
            "has_joins": self._has_joins(query.sql_statement),
            "has_aggregations": self._has_aggregations(query.sql_statement),
            "has_subqueries": self._has_subqueries(query.sql_statement),
            "has_where_clause": self._has_where_clause(query.sql_statement),
            "complexity_score": self._calculate_complexity_score(query)
        }
        
        # 分析数据库类型分布
        db_types = [db.database_type.value for db in query.involved_databases]
        features["database_types"] = list(set(db_types))
        features["homogeneous_databases"] = len(set(db_types)) == 1
        
        # 分析依赖关系特征
        if query.dependencies:
            features["avg_dependency_strength"] = sum(dep.strength for dep in query.dependencies) / len(query.dependencies)
            features["max_dependency_impact"] = max(dep.performance_impact for dep in query.dependencies)
            features["high_impact_dependencies"] = len([dep for dep in query.dependencies if dep.performance_impact > 0.7])
        
        return features
    
    async def _select_optimization_rules(self, query: CrossDatabaseQuery, features: Dict[str, Any]) -> List[OptimizationRule]:
        """选择适用的优化规则."""
        applicable_rules = []
        
        for rule in self.optimization_rules:
            if await self._rule_applies(rule, query, features):
                applicable_rules.append(rule)
        
        # 按优先级排序
        applicable_rules.sort(key=lambda r: r.priority)
        
        return applicable_rules
    
    async def _rule_applies(self, rule: OptimizationRule, query: CrossDatabaseQuery, features: Dict[str, Any]) -> bool:
        """检查规则是否适用."""
        condition = rule.condition
        
        # 简化的条件检查逻辑
        if condition == "has_where_clause_and_cross_db_join":
            return features.get("has_where_clause", False) and features.get("database_count", 0) > 1
        
        elif condition == "multiple_joins_with_different_selectivity":
            return features.get("has_joins", False) and features.get("database_count", 0) > 1
        
        elif condition == "independent_subqueries":
            return features.get("has_subqueries", False) and features.get("database_count", 0) > 1
        
        elif condition == "frequent_cross_db_access":
            return features.get("high_impact_dependencies", 0) > 0
        
        elif condition == "repeated_expensive_operations":
            return features.get("estimated_cost", 0) > 500
        
        elif condition == "complex_aggregations_on_large_datasets":
            return features.get("has_aggregations", False) and features.get("complexity_score", 0) > 0.7
        
        return False
    
    async def _generate_optimization_plan(self, query: CrossDatabaseQuery, rules: List[OptimizationRule]) -> QueryOptimizationPlan:
        """生成优化计划."""
        plan_id = f"plan_{hash(query.sql_statement)}_{datetime.now().timestamp()}"
        
        # 生成优化执行步骤
        execution_steps = []
        total_cost_reduction = 0.0
        total_time_reduction = 0.0
        
        for i, rule in enumerate(rules):
            step = {
                "step_id": i + 1,
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "strategy": rule.strategy.value,
                "description": rule.description,
                "estimated_improvement": rule.estimated_improvement,
                "execution_order": i + 1
            }
            execution_steps.append(step)
            
            total_cost_reduction += rule.estimated_improvement * 0.1  # 简化计算
            total_time_reduction += rule.estimated_improvement * 0.15
        
        # 风险评估
        risk_assessment = await self._assess_optimization_risks(query, rules)
        
        # 实现复杂度评估
        complexity = self._assess_implementation_complexity(rules)
        
        return QueryOptimizationPlan(
            plan_id=plan_id,
            original_query=query,
            applied_rules=rules,
            optimized_execution_steps=execution_steps,
            estimated_cost_reduction=min(total_cost_reduction, 0.8),  # 最大80%改进
            estimated_time_reduction=min(total_time_reduction, 0.8),
            risk_assessment=risk_assessment,
            implementation_complexity=complexity
        )
    
    async def _evaluate_optimization_plan(self, plan: QueryOptimizationPlan):
        """评估优化计划."""
        # 这里可以添加更详细的计划评估逻辑
        # 例如：成本效益分析、风险评估、可行性检查等
        pass
    
    async def _apply_optimization_rules(self, query: CrossDatabaseQuery, rules: List[OptimizationRule]) -> CrossDatabaseQuery:
        """应用优化规则."""
        optimized_query = query
        
        for rule in rules:
            if rule.strategy == OptimizationStrategy.PUSH_DOWN:
                optimized_query = await self._apply_predicate_pushdown(optimized_query)
            elif rule.strategy == OptimizationStrategy.JOIN_REORDER:
                optimized_query = await self._apply_join_reordering(optimized_query)
            elif rule.strategy == OptimizationStrategy.PARALLEL_EXECUTION:
                optimized_query = await self._apply_parallel_execution(optimized_query)
            # 其他策略的实现...
        
        return optimized_query
    
    async def _measure_query_performance(self, query: CrossDatabaseQuery) -> Dict[str, float]:
        """测量查询性能."""
        # 检查缓存
        cache_key = f"perf_{hash(query.sql_statement)}"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # 执行性能测量
        performance_analysis = await self.cross_db_analyzer.analyze_performance_impact(query)
        
        performance_metrics = {
            "cost": query.estimated_cost,
            "execution_time": performance_analysis.total_execution_time,
            "network_latency": performance_analysis.network_latency,
            "data_transfer_size": performance_analysis.data_transfer_size,
            "connection_overhead": performance_analysis.connection_overhead
        }
        
        # 缓存结果
        self.performance_cache[cache_key] = performance_metrics
        
        return performance_metrics
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """计算性能改进百分比."""
        before_time = before.get("execution_time", 1.0)
        after_time = after.get("execution_time", 1.0)
        
        if before_time == 0:
            return 0.0
        
        improvement = (before_time - after_time) / before_time
        return max(0.0, improvement)  # 确保不为负数
    
    async def _generate_cross_db_suggestions(self, query: CrossDatabaseQuery, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成跨数据库查询的优化建议."""
        suggestions = []
        
        if features.get("has_where_clause", False):
            suggestions.append({
                "type": "predicate_pushdown",
                "title": "谓词下推优化",
                "description": "将过滤条件推送到数据源，减少网络传输量",
                "expected_improvement": 0.3,
                "implementation_effort": "LOW",
                "risk_level": "LOW"
            })
        
        if features.get("database_count", 0) > 2:
            suggestions.append({
                "type": "query_decomposition",
                "title": "查询分解",
                "description": "将复杂的多数据库查询分解为多个简单查询",
                "expected_improvement": 0.25,
                "implementation_effort": "MEDIUM",
                "risk_level": "MEDIUM"
            })
        
        return suggestions
    
    async def _generate_distributed_suggestions(self, query: CrossDatabaseQuery, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成分布式查询的优化建议."""
        suggestions = []
        
        suggestions.append({
            "type": "parallel_execution",
            "title": "并行执行",
            "description": "并行执行独立的子查询以提高整体性能",
            "expected_improvement": 0.4,
            "implementation_effort": "MEDIUM",
            "risk_level": "MEDIUM"
        })
        
        if features.get("homogeneous_databases", False):
            suggestions.append({
                "type": "federated_query",
                "title": "联邦查询",
                "description": "使用联邦查询引擎统一处理同类型数据库",
                "expected_improvement": 0.35,
                "implementation_effort": "HIGH",
                "risk_level": "LOW"
            })
        
        return suggestions
    
    async def _generate_performance_based_suggestions(self, query: CrossDatabaseQuery, analysis: PerformanceImpactAnalysis) -> List[Dict[str, Any]]:
        """基于性能分析生成建议."""
        suggestions = []
        
        if analysis.network_latency > 100:  # 网络延迟超过100ms
            suggestions.append({
                "type": "network_optimization",
                "title": "网络优化",
                "description": "优化网络连接或使用数据缓存减少网络延迟",
                "expected_improvement": 0.2,
                "implementation_effort": "MEDIUM",
                "risk_level": "LOW"
            })
        
        if analysis.data_transfer_size > 1000000:  # 数据传输超过1MB
            suggestions.append({
                "type": "data_compression",
                "title": "数据压缩",
                "description": "启用数据压缩以减少网络传输量",
                "expected_improvement": 0.15,
                "implementation_effort": "LOW",
                "risk_level": "LOW"
            })
        
        if analysis.bottleneck_databases:
            suggestions.append({
                "type": "bottleneck_optimization",
                "title": "瓶颈优化",
                "description": f"优化瓶颈数据库 {', '.join(analysis.bottleneck_databases)} 的性能",
                "expected_improvement": 0.3,
                "implementation_effort": "HIGH",
                "risk_level": "MEDIUM"
            })
        
        return suggestions
    
    async def _generate_dependency_based_suggestions(self, query: CrossDatabaseQuery) -> List[Dict[str, Any]]:
        """基于依赖关系生成建议."""
        suggestions = []
        
        high_impact_deps = [dep for dep in query.dependencies if dep.performance_impact > 0.7]
        if high_impact_deps:
            suggestions.append({
                "type": "dependency_optimization",
                "title": "依赖关系优化",
                "description": "优化高影响的数据库依赖关系",
                "expected_improvement": 0.4,
                "implementation_effort": "HIGH",
                "risk_level": "MEDIUM"
            })
        
        frequent_deps = [dep for dep in query.dependencies if dep.frequency > 100]
        if frequent_deps:
            suggestions.append({
                "type": "data_replication",
                "title": "数据复制",
                "description": "复制频繁访问的数据以减少跨数据库查询",
                "expected_improvement": 0.5,
                "implementation_effort": "HIGH",
                "risk_level": "HIGH"
            })
        
        return suggestions
    
    async def _assess_optimization_risks(self, query: CrossDatabaseQuery, rules: List[OptimizationRule]) -> Dict[str, float]:
        """评估优化风险."""
        risks = {
            "data_consistency": 0.0,
            "performance_regression": 0.0,
            "implementation_complexity": 0.0,
            "system_stability": 0.0
        }
        
        # 基于规则类型评估风险
        for rule in rules:
            if rule.strategy == OptimizationStrategy.DATA_LOCALITY:
                risks["data_consistency"] += 0.3
            elif rule.strategy == OptimizationStrategy.PARALLEL_EXECUTION:
                risks["system_stability"] += 0.2
            elif rule.strategy == OptimizationStrategy.MATERIALIZATION:
                risks["implementation_complexity"] += 0.4
        
        # 基于查询复杂度评估风险
        complexity = self._calculate_complexity_score(query)
        risks["performance_regression"] = complexity * 0.3
        
        # 限制风险值在0-1范围内
        for key in risks:
            risks[key] = min(risks[key], 1.0)
        
        return risks
    
    def _assess_implementation_complexity(self, rules: List[OptimizationRule]) -> str:
        """评估实现复杂度."""
        complexity_scores = {
            OptimizationStrategy.PUSH_DOWN: 1,
            OptimizationStrategy.JOIN_REORDER: 2,
            OptimizationStrategy.PARALLEL_EXECUTION: 3,
            OptimizationStrategy.DATA_LOCALITY: 4,
            OptimizationStrategy.CACHING: 2,
            OptimizationStrategy.MATERIALIZATION: 5
        }
        
        total_complexity = sum(complexity_scores.get(rule.strategy, 3) for rule in rules)
        avg_complexity = total_complexity / len(rules) if rules else 0
        
        if avg_complexity <= 2:
            return "LOW"
        elif avg_complexity <= 3.5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    # SQL分析辅助方法
    def _has_joins(self, sql: str) -> bool:
        """检查SQL是否包含连接操作."""
        return any(keyword in sql.upper() for keyword in ["JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"])
    
    def _has_aggregations(self, sql: str) -> bool:
        """检查SQL是否包含聚合操作."""
        return any(keyword in sql.upper() for keyword in ["GROUP BY", "COUNT", "SUM", "AVG", "MAX", "MIN"])
    
    def _has_subqueries(self, sql: str) -> bool:
        """检查SQL是否包含子查询."""
        return "(" in sql and "SELECT" in sql.upper()
    
    def _has_where_clause(self, sql: str) -> bool:
        """检查SQL是否包含WHERE子句."""
        return "WHERE" in sql.upper()
    
    def _calculate_complexity_score(self, query: CrossDatabaseQuery) -> float:
        """计算查询复杂度评分."""
        score = 0.0
        
        # 基于涉及的数据库数量
        score += len(query.involved_databases) * 0.1
        
        # 基于依赖关系数量
        score += len(query.dependencies) * 0.05
        
        # 基于估算成本
        score += min(query.estimated_cost / 1000, 0.5)
        
        # 基于SQL复杂度
        sql = query.sql_statement.upper()
        if "JOIN" in sql:
            score += 0.1
        if "GROUP BY" in sql:
            score += 0.1
        if "ORDER BY" in sql:
            score += 0.05
        if "HAVING" in sql:
            score += 0.1
        
        return min(score, 1.0)
    
    # 优化策略实现（简化版本）
    async def _apply_predicate_pushdown(self, query: CrossDatabaseQuery) -> CrossDatabaseQuery:
        """应用谓词下推优化."""
        # 这里应该实现实际的谓词下推逻辑
        # 简化实现：降低估算成本
        optimized_query = query
        optimized_query.estimated_cost *= 0.7
        return optimized_query
    
    async def _apply_join_reordering(self, query: CrossDatabaseQuery) -> CrossDatabaseQuery:
        """应用连接重排序优化."""
        # 这里应该实现实际的连接重排序逻辑
        optimized_query = query
        optimized_query.estimated_cost *= 0.75
        return optimized_query
    
    async def _apply_parallel_execution(self, query: CrossDatabaseQuery) -> CrossDatabaseQuery:
        """应用并行执行优化."""
        # 这里应该实现实际的并行执行逻辑
        optimized_query = query
        optimized_query.estimated_cost *= 0.6
        return optimized_query