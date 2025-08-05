"""智能化工作流 - 实现增强的SQL分析、反馈学习、知识发现和自动优化工作流."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .workflow_templates import Workflow, WorkflowStep
from ..models.models import TaskStatus, TaskPriority
from ..communication.a2a_protocol import A2AMessage, MessageType, Priority

logger = logging.getLogger(__name__)


class IntelligentSQLAnalysisWorkflow:
    """智能SQL分析工作流 - 集成记忆检索、知识查询和个性化响应生成."""
    
    def __init__(self, orchestrator):
        """初始化智能SQL分析工作流.
        
        Args:
            orchestrator: Agent编排器实例
        """
        self.orchestrator = orchestrator
        self.workflow_type = "intelligent_sql_analysis"
    
    async def create_workflow(self, parameters: Dict[str, Any]) -> Workflow:
        """创建智能SQL分析工作流.
        
        Args:
            parameters: 工作流参数，包含：
                - sql_statement: SQL语句
                - user_id: 用户ID
                - session_id: 会话ID
                - database: 数据库信息
                - context: 上下文信息
                - user_preferences: 用户偏好
        
        Returns:
            Workflow: 创建的工作流对象
        """
        workflow = Workflow(
            name="智能SQL分析工作流",
            description="集成记忆检索和知识查询的增强SQL分析",
            workflow_type=self.workflow_type,
            user_id=parameters.get("user_id"),
            session_id=parameters.get("session_id"),
            priority=TaskPriority.HIGH
        )
        
        sql_statement = parameters.get("sql_statement")
        user_id = parameters.get("user_id")
        session_id = parameters.get("session_id")
        database = parameters.get("database")
        
        if not sql_statement or not user_id:
            raise ValueError("缺少必要参数: sql_statement 和 user_id")
        
        # 步骤1: 检索相似历史分析记录
        workflow.add_step(WorkflowStep(
            name="检索历史记忆",
            agent_id="memory_agent",
            action="retrieve_similar_analyses",
            payload={
                "sql_statement": sql_statement,
                "user_id": user_id,
                "similarity_threshold": 0.7,
                "max_results": 5
            },
            timeout=30
        ))
        
        # 步骤2: 查询相关知识和最佳实践
        workflow.add_step(WorkflowStep(
            name="查询知识库",
            agent_id="knowledge_agent",
            action="query_relevant_knowledge",
            payload={
                "sql_statement": sql_statement,
                "database_type": database.get("type") if database else "unknown",
                "problem_context": parameters.get("context", {}),
                "max_results": 10
            },
            timeout=45
        ))
        
        # 步骤3: 获取用户画像和偏好
        workflow.add_step(WorkflowStep(
            name="获取用户画像",
            agent_id="memory_agent",
            action="get_user_profile",
            payload={
                "user_id": user_id,
                "include_preferences": True,
                "include_history_summary": True
            },
            timeout=20
        ))
        
        # 步骤4: 执行基础SQL分析
        workflow.add_step(WorkflowStep(
            name="基础SQL分析",
            agent_id="sql_analysis_agent",
            action="analyze_sql_comprehensive",
            payload={
                "sql_statement": sql_statement,
                "database": database,
                "analysis_depth": "detailed",
                "include_execution_plan": True
            },
            dependencies=["retrieve_historical_memory"],
            timeout=90
        ))
        
        # 步骤5: 合并分析结果和上下文信息
        workflow.add_step(WorkflowStep(
            name="合并上下文信息",
            agent_id="sql_analysis_agent",
            action="merge_contextual_analysis",
            payload={
                "base_analysis": "from_basic_sql_analysis",
                "historical_context": "from_retrieve_historical_memory",
                "knowledge_context": "from_query_knowledge_base",
                "user_profile": "from_get_user_profile",
                "merge_strategy": "intelligent_weighted"
            },
            dependencies=["basic_sql_analysis", "query_knowledge_base", "get_user_profile"],
            timeout=60
        ))
        
        # 步骤6: 生成个性化优化建议
        workflow.add_step(WorkflowStep(
            name="生成个性化建议",
            agent_id="sql_analysis_agent",
            action="generate_personalized_suggestions",
            payload={
                "merged_analysis": "from_merge_contextual_analysis",
                "user_profile": "from_get_user_profile",
                "personalization_factors": [
                    "skill_level",
                    "preferred_optimization_types",
                    "risk_tolerance",
                    "performance_priorities"
                ]
            },
            dependencies=["merge_contextual_analysis"],
            timeout=45
        ))
        
        # 步骤7: 排序和过滤建议
        workflow.add_step(WorkflowStep(
            name="优化建议排序",
            agent_id="knowledge_agent",
            action="rank_suggestions_by_effectiveness",
            payload={
                "suggestions": "from_generate_personalized_suggestions",
                "ranking_criteria": {
                    "effectiveness_score": 0.4,
                    "user_preference_match": 0.3,
                    "implementation_difficulty": 0.2,
                    "risk_level": 0.1
                },
                "max_suggestions": 8
            },
            dependencies=["generate_personalized_suggestions"],
            timeout=30
        ))
        
        # 步骤8: 生成解释和教育内容
        workflow.add_step(WorkflowStep(
            name="生成解释内容",
            agent_id="knowledge_agent",
            action="generate_educational_content",
            payload={
                "analysis_results": "from_merge_contextual_analysis",
                "suggestions": "from_rank_suggestions",
                "user_skill_level": "from_get_user_profile",
                "explanation_depth": "adaptive"
            },
            dependencies=["rank_suggestions"],
            timeout=40
        ))
        
        # 步骤9: 存储分析记录到记忆系统
        workflow.add_step(WorkflowStep(
            name="存储分析记录",
            agent_id="memory_agent",
            action="store_analysis_memory",
            payload={
                "user_id": user_id,
                "session_id": session_id,
                "sql_statement": sql_statement,
                "analysis_results": "from_merge_contextual_analysis",
                "suggestions": "from_rank_suggestions",
                "context": {
                    "database": database,
                    "timestamp": datetime.now().isoformat(),
                    "workflow_id": workflow.workflow_id
                }
            },
            dependencies=["rank_suggestions"],
            timeout=30
        ))
        
        # 步骤10: 更新知识使用统计
        workflow.add_step(WorkflowStep(
            name="更新知识统计",
            agent_id="knowledge_agent",
            action="update_knowledge_usage_stats",
            payload={
                "used_knowledge_items": "from_query_knowledge_base",
                "analysis_context": "from_merge_contextual_analysis",
                "effectiveness_prediction": "from_rank_suggestions"
            },
            dependencies=["store_analysis_memory"],
            timeout=20
        ))
        
        # 步骤11: 生成最终响应
        workflow.add_step(WorkflowStep(
            name="生成最终响应",
            agent_id="sql_analysis_agent",
            action="generate_final_response",
            payload={
                "analysis_results": "from_merge_contextual_analysis",
                "ranked_suggestions": "from_rank_suggestions",
                "educational_content": "from_generate_educational_content",
                "response_format": parameters.get("response_format", "comprehensive"),
                "include_confidence_scores": True
            },
            dependencies=["generate_educational_content", "update_knowledge_statistics"],
            timeout=30
        ))
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行智能SQL分析工作流.
        
        Args:
            workflow: 要执行的工作流
        
        Returns:
            Dict[str, Any]: 工作流执行结果
        """
        try:
            workflow.status = "running"
            workflow.started_at = datetime.now()
            
            # 执行工作流
            result = await self.orchestrator.execute_workflow(workflow)
            
            # 记录执行日志
            await self._log_workflow_execution(workflow, result)
            
            return result
            
        except Exception as e:
            logger.error(f"智能SQL分析工作流执行失败: {e}")
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            
            # 记录失败日志
            await self._log_workflow_failure(workflow, e)
            
            raise
    
    async def _log_workflow_execution(self, workflow: Workflow, result: Dict[str, Any]):
        """记录工作流执行日志.
        
        Args:
            workflow: 执行的工作流
            result: 执行结果
        """
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "user_id": workflow.user_id,
            "session_id": workflow.session_id,
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else None,
            "status": workflow.status,
            "steps_executed": len([s for s in workflow.steps.values() if s.status == TaskStatus.COMPLETED]),
            "total_steps": len(workflow.steps),
            "result_summary": {
                "suggestions_count": len(result.get("suggestions", [])),
                "confidence_score": result.get("confidence_score", 0.0),
                "personalization_applied": result.get("personalization_applied", False)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"智能SQL分析工作流执行完成: {log_entry}")
    
    async def _log_workflow_failure(self, workflow: Workflow, error: Exception):
        """记录工作流失败日志.
        
        Args:
            workflow: 失败的工作流
            error: 错误信息
        """
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "user_id": workflow.user_id,
            "session_id": workflow.session_id,
            "error": str(error),
            "failed_step": self._get_failed_step(workflow),
            "steps_completed": len([s for s in workflow.steps.values() if s.status == TaskStatus.COMPLETED]),
            "total_steps": len(workflow.steps),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"智能SQL分析工作流执行失败: {log_entry}")
    
    def _get_failed_step(self, workflow: Workflow) -> Optional[str]:
        """获取失败的步骤名称.
        
        Args:
            workflow: 工作流对象
        
        Returns:
            Optional[str]: 失败步骤的名称
        """
        for step in workflow.steps.values():
            if step.status == TaskStatus.FAILED:
                return step.name
        return None


class FeedbackLearningWorkflow:
    """反馈学习工作流 - 处理用户反馈并更新用户画像和知识有效性."""
    
    def __init__(self, orchestrator):
        """初始化反馈学习工作流.
        
        Args:
            orchestrator: Agent编排器实例
        """
        self.orchestrator = orchestrator
        self.workflow_type = "feedback_learning"
    
    async def create_workflow(self, parameters: Dict[str, Any]) -> Workflow:
        """创建反馈学习工作流.
        
        Args:
            parameters: 工作流参数，包含：
                - user_id: 用户ID
                - feedback_data: 反馈数据
                - analysis_id: 相关分析ID
                - suggestion_id: 相关建议ID
        
        Returns:
            Workflow: 创建的工作流对象
        """
        workflow = Workflow(
            name="反馈学习工作流",
            description="处理用户反馈并更新用户画像和知识有效性",
            workflow_type=self.workflow_type,
            user_id=parameters.get("user_id"),
            priority=TaskPriority.MEDIUM
        )
        
        user_id = parameters.get("user_id")
        feedback_data = parameters.get("feedback_data")
        
        if not user_id or not feedback_data:
            raise ValueError("缺少必要参数: user_id 和 feedback_data")
        
        # 步骤1: 验证和预处理反馈数据
        workflow.add_step(WorkflowStep(
            name="预处理反馈数据",
            agent_id="learning_agent",
            action="preprocess_feedback",
            payload={
                "feedback_data": feedback_data,
                "user_id": user_id,
                "validation_rules": [
                    "completeness_check",
                    "consistency_check",
                    "validity_check"
                ]
            },
            timeout=30
        ))
        
        # 步骤2: 更新用户画像
        workflow.add_step(WorkflowStep(
            name="更新用户画像",
            agent_id="learning_agent",
            action="update_user_profile",
            payload={
                "user_id": user_id,
                "feedback_data": "from_preprocess_feedback",
                "update_strategy": "incremental_learning",
                "confidence_adjustment": True
            },
            dependencies=["preprocess_feedback"],
            timeout=45
        ))
        
        # 步骤3: 更新知识有效性评分
        workflow.add_step(WorkflowStep(
            name="更新知识有效性",
            agent_id="knowledge_agent",
            action="update_knowledge_effectiveness",
            payload={
                "feedback_data": "from_preprocess_feedback",
                "knowledge_items": parameters.get("related_knowledge_items", []),
                "effectiveness_algorithm": "weighted_feedback",
                "decay_factor": 0.95
            },
            dependencies=["preprocess_feedback"],
            timeout=30
        ))
        
        # 步骤4: 分析反馈模式
        workflow.add_step(WorkflowStep(
            name="分析反馈模式",
            agent_id="learning_agent",
            action="analyze_feedback_patterns",
            payload={
                "user_id": user_id,
                "current_feedback": "from_preprocess_feedback",
                "historical_window": "30d",
                "pattern_types": [
                    "preference_patterns",
                    "effectiveness_patterns",
                    "difficulty_patterns"
                ]
            },
            dependencies=["update_user_profile"],
            timeout=60
        ))
        
        # 步骤5: 触发模式发现（如果需要）
        workflow.add_step(WorkflowStep(
            name="触发模式发现",
            agent_id="learning_agent",
            action="trigger_pattern_discovery",
            payload={
                "feedback_patterns": "from_analyze_feedback_patterns",
                "discovery_threshold": 0.7,
                "min_feedback_count": 5,
                "discovery_scope": "user_specific"
            },
            dependencies=["analyze_feedback_patterns"],
            timeout=30
        ))
        
        # 步骤6: 更新建议排序权重
        workflow.add_step(WorkflowStep(
            name="更新排序权重",
            agent_id="knowledge_agent",
            action="update_suggestion_weights",
            payload={
                "user_id": user_id,
                "feedback_insights": "from_analyze_feedback_patterns",
                "weight_adjustment_strategy": "gradient_descent",
                "learning_rate": 0.1
            },
            dependencies=["analyze_feedback_patterns"],
            timeout=25
        ))
        
        # 步骤7: 存储反馈记录
        workflow.add_step(WorkflowStep(
            name="存储反馈记录",
            agent_id="memory_agent",
            action="store_feedback_record",
            payload={
                "user_id": user_id,
                "processed_feedback": "from_preprocess_feedback",
                "profile_updates": "from_update_user_profile",
                "pattern_insights": "from_analyze_feedback_patterns",
                "timestamp": datetime.now().isoformat()
            },
            dependencies=["update_suggestion_weights"],
            timeout=20
        ))
        
        # 步骤8: 生成学习报告
        workflow.add_step(WorkflowStep(
            name="生成学习报告",
            agent_id="learning_agent",
            action="generate_learning_report",
            payload={
                "user_id": user_id,
                "feedback_processing_results": "from_all_previous_steps",
                "report_type": "feedback_processing",
                "include_recommendations": True
            },
            dependencies=["store_feedback_record"],
            timeout=30
        ))
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行反馈学习工作流.
        
        Args:
            workflow: 要执行的工作流
        
        Returns:
            Dict[str, Any]: 工作流执行结果
        """
        try:
            workflow.status = "running"
            workflow.started_at = datetime.now()
            
            # 执行工作流
            result = await self.orchestrator.execute_workflow(workflow)
            
            # 记录执行日志
            await self._log_workflow_execution(workflow, result)
            
            return result
            
        except Exception as e:
            logger.error(f"反馈学习工作流执行失败: {e}")
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            
            # 记录失败日志
            await self._log_workflow_failure(workflow, e)
            
            raise
    
    async def _log_workflow_execution(self, workflow: Workflow, result: Dict[str, Any]):
        """记录工作流执行日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "user_id": workflow.user_id,
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else None,
            "status": workflow.status,
            "feedback_processed": result.get("feedback_processed", False),
            "profile_updated": result.get("profile_updated", False),
            "patterns_discovered": len(result.get("discovered_patterns", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"反馈学习工作流执行完成: {log_entry}")
    
    async def _log_workflow_failure(self, workflow: Workflow, error: Exception):
        """记录工作流失败日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "user_id": workflow.user_id,
            "error": str(error),
            "failed_step": self._get_failed_step(workflow),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"反馈学习工作流执行失败: {log_entry}")
    
    def _get_failed_step(self, workflow: Workflow) -> Optional[str]:
        """获取失败的步骤名称."""
        for step in workflow.steps.values():
            if step.status == TaskStatus.FAILED:
                return step.name
        return None


class KnowledgeDiscoveryWorkflow:
    """知识发现工作流 - 定期分析历史数据发现新的优化模式."""
    
    def __init__(self, orchestrator):
        """初始化知识发现工作流.
        
        Args:
            orchestrator: Agent编排器实例
        """
        self.orchestrator = orchestrator
        self.workflow_type = "knowledge_discovery"
    
    async def create_workflow(self, parameters: Dict[str, Any]) -> Workflow:
        """创建知识发现工作流.
        
        Args:
            parameters: 工作流参数，包含：
                - time_window: 分析时间窗口
                - discovery_type: 发现类型
                - min_pattern_support: 最小模式支持度
        
        Returns:
            Workflow: 创建的工作流对象
        """
        workflow = Workflow(
            name="知识发现工作流",
            description="定期分析历史数据发现新的优化模式",
            workflow_type=self.workflow_type,
            priority=TaskPriority.LOW
        )
        
        time_window = parameters.get("time_window", "7d")
        discovery_type = parameters.get("discovery_type", "optimization_patterns")
        
        # 步骤1: 收集历史分析数据
        workflow.add_step(WorkflowStep(
            name="收集历史数据",
            agent_id="memory_agent",
            action="collect_historical_analysis_data",
            payload={
                "time_window": time_window,
                "data_types": [
                    "sql_analyses",
                    "optimization_results",
                    "user_feedback",
                    "performance_metrics"
                ],
                "include_metadata": True,
                "min_records": 50
            },
            timeout=180
        ))
        
        # 步骤2: 数据预处理和特征提取
        workflow.add_step(WorkflowStep(
            name="数据预处理",
            agent_id="learning_agent",
            action="preprocess_discovery_data",
            payload={
                "raw_data": "from_collect_historical_data",
                "preprocessing_steps": [
                    "data_cleaning",
                    "feature_extraction",
                    "normalization",
                    "dimensionality_reduction"
                ],
                "feature_types": [
                    "sql_patterns",
                    "performance_metrics",
                    "optimization_outcomes",
                    "user_characteristics"
                ]
            },
            dependencies=["collect_historical_data"],
            timeout=240
        ))
        
        # 步骤3: 聚类分析发现模式
        workflow.add_step(WorkflowStep(
            name="聚类分析",
            agent_id="learning_agent",
            action="perform_clustering_analysis",
            payload={
                "processed_data": "from_preprocess_discovery_data",
                "clustering_algorithms": [
                    "kmeans",
                    "dbscan",
                    "hierarchical"
                ],
                "cluster_validation": True,
                "min_cluster_size": parameters.get("min_cluster_size", 5),
                "max_clusters": parameters.get("max_clusters", 20)
            },
            dependencies=["preprocess_discovery_data"],
            timeout=300
        ))
        
        # 步骤4: 模式提取和特征化
        workflow.add_step(WorkflowStep(
            name="模式提取",
            agent_id="learning_agent",
            action="extract_optimization_patterns",
            payload={
                "clustering_results": "from_clustering_analysis",
                "pattern_extraction_methods": [
                    "frequent_itemsets",
                    "association_rules",
                    "sequential_patterns"
                ],
                "min_support": parameters.get("min_pattern_support", 0.1),
                "min_confidence": parameters.get("min_confidence", 0.6)
            },
            dependencies=["clustering_analysis"],
            timeout=180
        ))
        
        # 步骤5: 模式有效性验证
        workflow.add_step(WorkflowStep(
            name="模式验证",
            agent_id="learning_agent",
            action="validate_discovered_patterns",
            payload={
                "discovered_patterns": "from_extract_patterns",
                "validation_methods": [
                    "cross_validation",
                    "statistical_significance",
                    "domain_expert_rules"
                ],
                "validation_data": "from_collect_historical_data",
                "significance_threshold": 0.05
            },
            dependencies=["extract_patterns"],
            timeout=120
        ))
        
        # 步骤6: 模式质量评估
        workflow.add_step(WorkflowStep(
            name="质量评估",
            agent_id="knowledge_agent",
            action="assess_pattern_quality",
            payload={
                "validated_patterns": "from_validate_patterns",
                "quality_metrics": [
                    "novelty_score",
                    "utility_score",
                    "generalizability_score",
                    "actionability_score"
                ],
                "existing_knowledge": "from_knowledge_base",
                "quality_threshold": 0.7
            },
            dependencies=["validate_patterns"],
            timeout=90
        ))
        
        # 步骤7: 更新知识库
        workflow.add_step(WorkflowStep(
            name="更新知识库",
            agent_id="knowledge_agent",
            action="update_knowledge_base_with_patterns",
            payload={
                "quality_assessed_patterns": "from_assess_quality",
                "update_strategy": "incremental_merge",
                "conflict_resolution": "evidence_based",
                "version_control": True
            },
            dependencies=["assess_quality"],
            timeout=60
        ))
        
        # 步骤8: 更新向量索引
        workflow.add_step(WorkflowStep(
            name="更新向量索引",
            agent_id="memory_agent",
            action="update_vector_indices",
            payload={
                "new_knowledge_items": "from_update_knowledge_base",
                "index_types": [
                    "semantic_similarity",
                    "pattern_matching",
                    "contextual_relevance"
                ],
                "batch_update": True,
                "optimize_indices": True
            },
            dependencies=["update_knowledge_base"],
            timeout=120
        ))
        
        # 步骤9: 生成发现报告
        workflow.add_step(WorkflowStep(
            name="生成发现报告",
            agent_id="learning_agent",
            action="generate_discovery_report",
            payload={
                "discovery_results": "from_all_previous_steps",
                "report_format": "comprehensive",
                "include_visualizations": True,
                "include_recommendations": True,
                "audience": "technical"
            },
            dependencies=["update_vector_indices"],
            timeout=45
        ))
        
        # 步骤10: 通知相关系统
        workflow.add_step(WorkflowStep(
            name="通知系统更新",
            agent_id="coordinator_agent",
            action="notify_knowledge_update",
            payload={
                "update_summary": "from_generate_discovery_report",
                "affected_agents": [
                    "sql_analysis_agent",
                    "knowledge_agent",
                    "memory_agent"
                ],
                "notification_type": "knowledge_base_update"
            },
            dependencies=["generate_discovery_report"],
            timeout=30
        ))
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行知识发现工作流."""
        try:
            workflow.status = "running"
            workflow.started_at = datetime.now()
            
            # 执行工作流
            result = await self.orchestrator.execute_workflow(workflow)
            
            # 记录执行日志
            await self._log_workflow_execution(workflow, result)
            
            return result
            
        except Exception as e:
            logger.error(f"知识发现工作流执行失败: {e}")
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            
            # 记录失败日志
            await self._log_workflow_failure(workflow, e)
            
            raise
    
    async def _log_workflow_execution(self, workflow: Workflow, result: Dict[str, Any]):
        """记录工作流执行日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else None,
            "status": workflow.status,
            "patterns_discovered": len(result.get("discovered_patterns", [])),
            "patterns_validated": len(result.get("validated_patterns", [])),
            "knowledge_items_added": result.get("knowledge_items_added", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"知识发现工作流执行完成: {log_entry}")
    
    async def _log_workflow_failure(self, workflow: Workflow, error: Exception):
        """记录工作流失败日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "error": str(error),
            "failed_step": self._get_failed_step(workflow),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"知识发现工作流执行失败: {log_entry}")
    
    def _get_failed_step(self, workflow: Workflow) -> Optional[str]:
        """获取失败的步骤名称."""
        for step in workflow.steps.values():
            if step.status == TaskStatus.FAILED:
                return step.name
        return None


class AutoOptimizationWorkflow:
    """自动优化工作流 - 集成监控、安全、优化和回滚Agent的完整自动化流程."""
    
    def __init__(self, orchestrator):
        """初始化自动优化工作流.
        
        Args:
            orchestrator: Agent编排器实例
        """
        self.orchestrator = orchestrator
        self.workflow_type = "auto_optimization"
    
    async def create_workflow(self, parameters: Dict[str, Any]) -> Workflow:
        """创建自动优化工作流.
        
        Args:
            parameters: 工作流参数，包含：
                - trigger_event: 触发事件
                - optimization_target: 优化目标
                - risk_tolerance: 风险容忍度
                - auto_execute: 是否自动执行
        
        Returns:
            Workflow: 创建的工作流对象
        """
        workflow = Workflow(
            name="自动优化工作流",
            description="异常检测到自动修复的完整流程",
            workflow_type=self.workflow_type,
            priority=TaskPriority.HIGH
        )
        
        trigger_event = parameters.get("trigger_event")
        optimization_target = parameters.get("optimization_target")
        
        if not trigger_event or not optimization_target:
            raise ValueError("缺少必要参数: trigger_event 和 optimization_target")
        
        # 步骤1: 异常检测和分析
        workflow.add_step(WorkflowStep(
            name="异常检测分析",
            agent_id="monitoring_agent",
            action="analyze_performance_anomaly",
            payload={
                "trigger_event": trigger_event,
                "analysis_depth": "comprehensive",
                "include_root_cause": True,
                "historical_comparison": True,
                "anomaly_threshold": parameters.get("anomaly_threshold", 2.0)
            },
            timeout=120
        ))
        
        # 步骤2: 安全风险评估
        workflow.add_step(WorkflowStep(
            name="安全风险评估",
            agent_id="safety_agent",
            action="assess_optimization_risks",
            payload={
                "anomaly_analysis": "from_anomaly_detection",
                "optimization_target": optimization_target,
                "risk_factors": [
                    "data_integrity",
                    "system_availability",
                    "performance_impact",
                    "rollback_complexity"
                ],
                "risk_tolerance": parameters.get("risk_tolerance", "medium")
            },
            dependencies=["anomaly_detection_analysis"],
            timeout=90
        ))
        
        # 步骤3: 生成优化方案
        workflow.add_step(WorkflowStep(
            name="生成优化方案",
            agent_id="auto_optimizer",
            action="generate_optimization_plan",
            payload={
                "anomaly_analysis": "from_anomaly_detection",
                "risk_assessment": "from_security_risk_assessment",
                "optimization_target": optimization_target,
                "plan_types": [
                    "immediate_fixes",
                    "preventive_measures",
                    "long_term_optimizations"
                ],
                "include_alternatives": True
            },
            dependencies=["security_risk_assessment"],
            timeout=150
        ))
        
        # 步骤4: 方案安全验证
        workflow.add_step(WorkflowStep(
            name="方案安全验证",
            agent_id="safety_agent",
            action="validate_optimization_plan_safety",
            payload={
                "optimization_plan": "from_generate_optimization_plan",
                "validation_criteria": [
                    "sql_injection_check",
                    "privilege_escalation_check",
                    "data_corruption_risk",
                    "system_stability_impact"
                ],
                "approval_required": not parameters.get("auto_execute", False)
            },
            dependencies=["generate_optimization_plan"],
            timeout=60
        ))
        
        # 步骤5: 创建回滚计划
        workflow.add_step(WorkflowStep(
            name="创建回滚计划",
            agent_id="rollback_manager",
            action="create_comprehensive_rollback_plan",
            payload={
                "optimization_plan": "from_generate_optimization_plan",
                "safety_validation": "from_validate_plan_safety",
                "rollback_strategies": [
                    "immediate_rollback",
                    "partial_rollback",
                    "staged_rollback"
                ],
                "backup_requirements": True
            },
            dependencies=["validate_plan_safety"],
            timeout=90
        ))
        
        # 步骤6: 执行预优化快照
        workflow.add_step(WorkflowStep(
            name="执行预优化快照",
            agent_id="rollback_manager",
            action="create_pre_optimization_snapshot",
            payload={
                "optimization_plan": "from_generate_optimization_plan",
                "snapshot_scope": "comprehensive",
                "include_performance_baseline": True,
                "verification_required": True
            },
            dependencies=["create_rollback_plan"],
            timeout=180
        ))
        
        # 步骤7: 执行优化操作
        workflow.add_step(WorkflowStep(
            name="执行优化操作",
            agent_id="auto_optimizer",
            action="execute_optimization_plan",
            payload={
                "optimization_plan": "from_generate_optimization_plan",
                "rollback_plan": "from_create_rollback_plan",
                "execution_mode": "staged",
                "monitoring_enabled": True,
                "auto_rollback_triggers": [
                    "performance_degradation",
                    "error_rate_increase",
                    "timeout_exceeded"
                ]
            },
            dependencies=["create_pre_optimization_snapshot"],
            timeout=600,
            max_retries=1
        ))
        
        # 步骤8: 实时效果监控
        workflow.add_step(WorkflowStep(
            name="实时效果监控",
            agent_id="monitoring_agent",
            action="monitor_optimization_effects",
            payload={
                "optimization_execution": "from_execute_optimization",
                "monitoring_duration": parameters.get("monitoring_duration", 300),
                "performance_metrics": [
                    "query_response_time",
                    "throughput",
                    "resource_utilization",
                    "error_rates"
                ],
                "alert_thresholds": parameters.get("alert_thresholds", {})
            },
            dependencies=["execute_optimization"],
            timeout=360
        ))
        
        # 步骤9: 优化效果验证
        workflow.add_step(WorkflowStep(
            name="优化效果验证",
            agent_id="monitoring_agent",
            action="validate_optimization_effectiveness",
            payload={
                "monitoring_results": "from_real_time_monitoring",
                "baseline_metrics": "from_create_pre_optimization_snapshot",
                "success_criteria": parameters.get("success_criteria", {}),
                "validation_tests": [
                    "performance_improvement",
                    "stability_check",
                    "regression_test"
                ]
            },
            dependencies=["real_time_monitoring"],
            timeout=120
        ))
        
        # 步骤10: 学习和反馈更新
        workflow.add_step(WorkflowStep(
            name="学习反馈更新",
            agent_id="learning_agent",
            action="update_optimization_knowledge",
            payload={
                "optimization_results": "from_validate_effectiveness",
                "execution_context": "from_all_previous_steps",
                "learning_updates": [
                    "pattern_effectiveness",
                    "risk_assessment_accuracy",
                    "execution_time_prediction"
                ],
                "knowledge_base_update": True
            },
            dependencies=["validate_effectiveness"],
            timeout=60
        ))
        
        # 步骤11: 生成优化报告
        workflow.add_step(WorkflowStep(
            name="生成优化报告",
            agent_id="auto_optimizer",
            action="generate_optimization_report",
            payload={
                "complete_workflow_results": "from_all_previous_steps",
                "report_format": "comprehensive",
                "include_metrics": True,
                "include_recommendations": True,
                "audience": ["technical", "management"]
            },
            dependencies=["learning_feedback_update"],
            timeout=45
        ))
        
        return workflow
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """执行自动优化工作流."""
        try:
            workflow.status = "running"
            workflow.started_at = datetime.now()
            
            # 执行工作流
            result = await self.orchestrator.execute_workflow(workflow)
            
            # 记录执行日志
            await self._log_workflow_execution(workflow, result)
            
            return result
            
        except Exception as e:
            logger.error(f"自动优化工作流执行失败: {e}")
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            
            # 记录失败日志和触发回滚
            await self._log_workflow_failure(workflow, e)
            await self._trigger_emergency_rollback(workflow, e)
            
            raise
    
    async def _trigger_emergency_rollback(self, workflow: Workflow, error: Exception):
        """触发紧急回滚.
        
        Args:
            workflow: 失败的工作流
            error: 错误信息
        """
        try:
            # 检查是否已执行优化操作
            optimization_step = workflow.steps.get("execute_optimization")
            if optimization_step and optimization_step.status == TaskStatus.COMPLETED:
                # 创建紧急回滚工作流
                rollback_workflow = Workflow(
                    name="紧急回滚工作流",
                    description=f"自动优化失败后的紧急回滚: {error}",
                    workflow_type="emergency_rollback",
                    priority=TaskPriority.CRITICAL
                )
                
                rollback_workflow.add_step(WorkflowStep(
                    name="紧急回滚",
                    agent_id="rollback_manager",
                    action="execute_emergency_rollback",
                    payload={
                        "failed_workflow_id": workflow.workflow_id,
                        "rollback_reason": str(error),
                        "rollback_priority": "immediate"
                    },
                    timeout=300
                ))
                
                # 执行紧急回滚
                await self.orchestrator.execute_workflow(rollback_workflow)
                logger.info(f"紧急回滚已执行，工作流ID: {rollback_workflow.workflow_id}")
                
        except Exception as rollback_error:
            logger.critical(f"紧急回滚执行失败: {rollback_error}")
    
    async def _log_workflow_execution(self, workflow: Workflow, result: Dict[str, Any]):
        """记录工作流执行日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else None,
            "status": workflow.status,
            "optimization_success": result.get("optimization_success", False),
            "performance_improvement": result.get("performance_improvement", 0.0),
            "rollback_required": result.get("rollback_required", False),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"自动优化工作流执行完成: {log_entry}")
    
    async def _log_workflow_failure(self, workflow: Workflow, error: Exception):
        """记录工作流失败日志."""
        log_entry = {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "error": str(error),
            "failed_step": self._get_failed_step(workflow),
            "emergency_rollback_triggered": True,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.error(f"自动优化工作流执行失败: {log_entry}")
    
    def _get_failed_step(self, workflow: Workflow) -> Optional[str]:
        """获取失败的步骤名称."""
        for step in workflow.steps.values():
            if step.status == TaskStatus.FAILED:
                return step.name
        return None


# 工作流管理器
class IntelligentWorkflowManager:
    """智能工作流管理器 - 统一管理所有智能化工作流."""
    
    def __init__(self, orchestrator):
        """初始化智能工作流管理器.
        
        Args:
            orchestrator: Agent编排器实例
        """
        self.orchestrator = orchestrator
        self.workflows = {
            "intelligent_sql_analysis": IntelligentSQLAnalysisWorkflow(orchestrator),
            "feedback_learning": FeedbackLearningWorkflow(orchestrator),
            "knowledge_discovery": KnowledgeDiscoveryWorkflow(orchestrator),
            "auto_optimization": AutoOptimizationWorkflow(orchestrator)
        }
        
        # 操作日志和审计追踪
        self.operation_logs = []
        self.audit_trail = []
    
    async def execute_intelligent_workflow(
        self, 
        workflow_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行智能工作流.
        
        Args:
            workflow_type: 工作流类型
            parameters: 工作流参数
        
        Returns:
            Dict[str, Any]: 执行结果
        """
        if workflow_type not in self.workflows:
            raise ValueError(f"不支持的工作流类型: {workflow_type}")
        
        workflow_handler = self.workflows[workflow_type]
        
        try:
            # 记录操作开始
            operation_id = str(uuid4())
            await self._log_operation_start(operation_id, workflow_type, parameters)
            
            # 创建工作流
            workflow = await workflow_handler.create_workflow(parameters)
            workflow.metadata["operation_id"] = operation_id
            
            # 执行工作流
            result = await workflow_handler.execute_workflow(workflow)
            
            # 记录操作成功
            await self._log_operation_success(operation_id, workflow, result)
            
            return result
            
        except Exception as e:
            # 记录操作失败
            await self._log_operation_failure(operation_id, workflow_type, e)
            
            # 触发失败恢复流程
            await self._handle_workflow_failure(workflow_type, parameters, e)
            
            raise
    
    async def _log_operation_start(
        self, 
        operation_id: str, 
        workflow_type: str, 
        parameters: Dict[str, Any]
    ):
        """记录操作开始日志.
        
        Args:
            operation_id: 操作ID
            workflow_type: 工作流类型
            parameters: 参数
        """
        log_entry = {
            "operation_id": operation_id,
            "workflow_type": workflow_type,
            "action": "workflow_start",
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "user_id": parameters.get("user_id"),
            "session_id": parameters.get("session_id")
        }
        
        self.operation_logs.append(log_entry)
        self.audit_trail.append({
            **log_entry,
            "audit_type": "operation_start",
            "severity": "info"
        })
        
        logger.info(f"工作流开始执行: {log_entry}")
    
    async def _log_operation_success(
        self, 
        operation_id: str, 
        workflow: Workflow, 
        result: Dict[str, Any]
    ):
        """记录操作成功日志.
        
        Args:
            operation_id: 操作ID
            workflow: 工作流对象
            result: 执行结果
        """
        log_entry = {
            "operation_id": operation_id,
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type,
            "action": "workflow_success",
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at else None,
            "result_summary": {
                "status": "success",
                "steps_completed": len([s for s in workflow.steps.values() if s.status == TaskStatus.COMPLETED]),
                "total_steps": len(workflow.steps)
            },
            "timestamp": datetime.now().isoformat(),
            "user_id": workflow.user_id,
            "session_id": workflow.session_id
        }
        
        self.operation_logs.append(log_entry)
        self.audit_trail.append({
            **log_entry,
            "audit_type": "operation_success",
            "severity": "info"
        })
        
        logger.info(f"工作流执行成功: {log_entry}")
    
    async def _log_operation_failure(
        self, 
        operation_id: str, 
        workflow_type: str, 
        error: Exception
    ):
        """记录操作失败日志.
        
        Args:
            operation_id: 操作ID
            workflow_type: 工作流类型
            error: 错误信息
        """
        log_entry = {
            "operation_id": operation_id,
            "workflow_type": workflow_type,
            "action": "workflow_failure",
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        self.operation_logs.append(log_entry)
        self.audit_trail.append({
            **log_entry,
            "audit_type": "operation_failure",
            "severity": "error"
        })
        
        logger.error(f"工作流执行失败: {log_entry}")
    
    async def _handle_workflow_failure(
        self, 
        workflow_type: str, 
        parameters: Dict[str, Any], 
        error: Exception
    ):
        """处理工作流失败的恢复流程.
        
        Args:
            workflow_type: 失败的工作流类型
            parameters: 原始参数
            error: 错误信息
        """
        try:
            # 根据工作流类型和错误类型决定恢复策略
            recovery_strategy = self._determine_recovery_strategy(workflow_type, error)
            
            if recovery_strategy == "retry":
                # 延迟重试
                await asyncio.sleep(5)
                logger.info(f"重试工作流: {workflow_type}")
                # 这里可以实现重试逻辑
                
            elif recovery_strategy == "rollback":
                # 触发回滚
                logger.info(f"触发回滚恢复: {workflow_type}")
                # 这里可以实现回滚逻辑
                
            elif recovery_strategy == "alert":
                # 发送告警
                logger.warning(f"发送失败告警: {workflow_type}")
                # 这里可以实现告警逻辑
                
        except Exception as recovery_error:
            logger.critical(f"恢复流程执行失败: {recovery_error}")
    
    def _determine_recovery_strategy(self, workflow_type: str, error: Exception) -> str:
        """确定恢复策略.
        
        Args:
            workflow_type: 工作流类型
            error: 错误信息
        
        Returns:
            str: 恢复策略
        """
        # 根据错误类型和工作流类型确定恢复策略
        if isinstance(error, TimeoutError):
            return "retry"
        elif isinstance(error, ValueError):
            return "alert"
        elif workflow_type == "auto_optimization":
            return "rollback"
        else:
            return "alert"
    
    def get_operation_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取操作日志.
        
        Args:
            limit: 返回记录数限制
        
        Returns:
            List[Dict[str, Any]]: 操作日志列表
        """
        return self.operation_logs[-limit:]
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计追踪记录.
        
        Args:
            limit: 返回记录数限制
        
        Returns:
            List[Dict[str, Any]]: 审计追踪记录列表
        """
        return self.audit_trail[-limit:]