"""工作流模板定义 - 预定义的常用工作流模板."""

import logging
from typing import Any, Dict

# 避免复杂的导入依赖，直接在这里定义必要的类型
try:
    from .agent_orchestrator import Workflow, WorkflowStep
    from ..models.models import TaskPriority
except ImportError:
    # 如果导入失败，使用本地定义
    from enum import Enum
    from pydantic import BaseModel, Field
    from uuid import uuid4
    from datetime import datetime
    from typing import Any, Dict, List, Optional
    
    class TaskPriority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class TaskStatus(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class WorkflowStatus(str, Enum):
        CREATED = "created"
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        PAUSED = "paused"
    
    class WorkflowStep(BaseModel):
        step_id: str = Field(default_factory=lambda: str(uuid4()))
        name: str
        agent_id: str
        action: str
        payload: Dict[str, Any] = Field(default_factory=dict)
        dependencies: List[str] = Field(default_factory=list)
        timeout: int = Field(default=60)
        retry_count: int = Field(default=0)
        max_retries: int = Field(default=3)
        status: TaskStatus = Field(default=TaskStatus.PENDING)
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        execution_time: Optional[float] = None
    
    class Workflow(BaseModel):
        workflow_id: str = Field(default_factory=lambda: str(uuid4()))
        name: str
        description: str = ""
        workflow_type: str
        steps: Dict[str, WorkflowStep] = Field(default_factory=dict)
        status: WorkflowStatus = Field(default=WorkflowStatus.CREATED)
        priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
        created_at: datetime = Field(default_factory=datetime.now)
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        session_id: Optional[str] = None
        user_id: Optional[str] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)
        
        def add_step(self, step: WorkflowStep):
            self.steps[step.step_id] = step

logger = logging.getLogger(__name__)


async def create_sql_analysis_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建SQL分析工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    sql_statement = parameters.get("sql_statement")
    database = parameters.get("database")
    
    if not sql_statement:
        raise ValueError("缺少SQL语句参数")
    
    # 步骤1: NLP处理（如果有自然语言查询）
    if parameters.get("natural_language_query"):
        workflow.add_step(WorkflowStep(
            name="自然语言处理",
            agent_id="nlp_agent",
            action="process_natural_language",
            payload={
                "query": parameters.get("natural_language_query"),
                "context": parameters.get("context", {})
            },
            timeout=30
        ))
    
    # 步骤2: SQL语法分析
    workflow.add_step(WorkflowStep(
        name="SQL语法分析",
        agent_id="sql_analysis_agent",
        action="analyze_sql_syntax",
        payload={
            "sql_statement": sql_statement,
            "database": database
        },
        dependencies=["nlp_processing"] if parameters.get("natural_language_query") else [],
        timeout=60
    ))
    
    # 步骤3: 执行计划分析
    workflow.add_step(WorkflowStep(
        name="执行计划分析",
        agent_id="sql_analysis_agent",
        action="analyze_execution_plan",
        payload={
            "sql_statement": sql_statement,
            "database": database,
            "explain_results": parameters.get("explain_results", [])
        },
        dependencies=["sql_syntax_analysis"],
        timeout=90
    ))
    
    # 步骤4: 安全检查
    workflow.add_step(WorkflowStep(
        name="安全检查",
        agent_id="safety_agent",
        action="validate_sql_safety",
        payload={
            "sql_statement": sql_statement,
            "user_info": parameters.get("user_info", {})
        },
        dependencies=["sql_syntax_analysis"],
        timeout=30
    ))
    
    # 步骤5: 性能分析
    workflow.add_step(WorkflowStep(
        name="性能分析",
        agent_id="sql_analysis_agent",
        action="analyze_performance",
        payload={
            "sql_statement": sql_statement,
            "execution_plan": "from_execution_plan_analysis"
        },
        dependencies=["execution_plan_analysis", "safety_check"],
        timeout=120
    ))
    
    # 步骤6: 优化建议生成
    workflow.add_step(WorkflowStep(
        name="优化建议生成",
        agent_id="optimization_agent",
        action="generate_optimization_suggestions",
        payload={
            "sql_statement": sql_statement,
            "performance_analysis": "from_performance_analysis",
            "user_preferences": parameters.get("user_preferences", {})
        },
        dependencies=["performance_analysis"],
        timeout=60
    ))
    
    # 步骤7: 记忆存储
    workflow.add_step(WorkflowStep(
        name="记忆存储",
        agent_id="memory_agent",
        action="store_analysis_memory",
        payload={
            "sql_statement": sql_statement,
            "analysis_results": "from_all_previous_steps",
            "user_id": parameters.get("user_id"),
            "session_id": parameters.get("session_id")
        },
        dependencies=["optimization_suggestions"],
        timeout=30
    ))


async def create_optimization_execution_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建优化执行工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    optimization_type = parameters.get("optimization_type")
    target_table = parameters.get("target_table")
    
    if not optimization_type or not target_table:
        raise ValueError("缺少优化类型或目标表参数")
    
    # 步骤1: 创建操作快照
    workflow.add_step(WorkflowStep(
        name="创建操作快照",
        agent_id="rollback_agent",
        action="create_operation_snapshot",
        payload={
            "operation_id": workflow.workflow_id,
            "target_objects": [target_table],
            "operation_type": optimization_type
        },
        timeout=120
    ))
    
    # 步骤2: 安全验证
    workflow.add_step(WorkflowStep(
        name="安全验证",
        agent_id="safety_agent",
        action="validate_optimization_safety",
        payload={
            "optimization_type": optimization_type,
            "target_table": target_table,
            "user_info": parameters.get("user_info", {})
        },
        dependencies=["create_snapshot"],
        timeout=60
    ))
    
    # 步骤3: 执行优化
    workflow.add_step(WorkflowStep(
        name="执行优化",
        agent_id="optimization_agent",
        action="execute_optimization",
        payload={
            "optimization_type": optimization_type,
            "target_table": target_table,
            "optimization_params": parameters.get("optimization_params", {})
        },
        dependencies=["safety_validation"],
        timeout=300,
        max_retries=1  # 优化操作只重试一次
    ))
    
    # 步骤4: 验证优化结果
    workflow.add_step(WorkflowStep(
        name="验证优化结果",
        agent_id="monitoring_agent",
        action="validate_optimization_result",
        payload={
            "optimization_id": workflow.workflow_id,
            "target_table": target_table,
            "expected_improvement": parameters.get("expected_improvement")
        },
        dependencies=["execute_optimization"],
        timeout=180
    ))
    
    # 步骤5: 更新知识库
    workflow.add_step(WorkflowStep(
        name="更新知识库",
        agent_id="knowledge_agent",
        action="update_optimization_knowledge",
        payload={
            "optimization_type": optimization_type,
            "target_table": target_table,
            "optimization_result": "from_validation",
            "effectiveness": "from_validation"
        },
        dependencies=["validate_result"],
        timeout=30
    ))


async def create_safety_check_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建安全检查工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    sql_statement = parameters.get("sql_statement")
    user_info = parameters.get("user_info", {})
    
    if not sql_statement:
        raise ValueError("缺少SQL语句参数")
    
    # 步骤1: SQL注入检查
    workflow.add_step(WorkflowStep(
        name="SQL注入检查",
        agent_id="safety_agent",
        action="check_sql_injection",
        payload={
            "sql_statement": sql_statement
        },
        timeout=30
    ))
    
    # 步骤2: 权限验证
    workflow.add_step(WorkflowStep(
        name="权限验证",
        agent_id="safety_agent",
        action="validate_user_permissions",
        payload={
            "sql_statement": sql_statement,
            "user_info": user_info
        },
        dependencies=["sql_injection_check"],
        timeout=30
    ))
    
    # 步骤3: 操作风险评估
    workflow.add_step(WorkflowStep(
        name="操作风险评估",
        agent_id="safety_agent",
        action="assess_operation_risk",
        payload={
            "sql_statement": sql_statement,
            "user_info": user_info,
            "operation_context": parameters.get("context", {})
        },
        dependencies=["permission_validation"],
        timeout=45
    ))
    
    # 步骤4: 数据敏感性检查
    workflow.add_step(WorkflowStep(
        name="数据敏感性检查",
        agent_id="safety_agent",
        action="check_data_sensitivity",
        payload={
            "sql_statement": sql_statement,
            "affected_tables": parameters.get("affected_tables", [])
        },
        dependencies=["risk_assessment"],
        timeout=30
    ))


async def create_rollback_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建回滚工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    operation_id = parameters.get("operation_id")
    rollback_type = parameters.get("rollback_type", "automatic")
    
    if not operation_id:
        raise ValueError("缺少操作ID参数")
    
    # 步骤1: 获取回滚计划
    workflow.add_step(WorkflowStep(
        name="获取回滚计划",
        agent_id="rollback_agent",
        action="get_rollback_plan",
        payload={
            "operation_id": operation_id
        },
        timeout=60
    ))
    
    # 步骤2: 验证回滚可行性
    workflow.add_step(WorkflowStep(
        name="验证回滚可行性",
        agent_id="rollback_agent",
        action="validate_rollback_feasibility",
        payload={
            "operation_id": operation_id,
            "rollback_plan": "from_get_rollback_plan"
        },
        dependencies=["get_rollback_plan"],
        timeout=30
    ))
    
    # 步骤3: 安全验证
    workflow.add_step(WorkflowStep(
        name="回滚安全验证",
        agent_id="safety_agent",
        action="validate_rollback_safety",
        payload={
            "operation_id": operation_id,
            "rollback_plan": "from_get_rollback_plan"
        },
        dependencies=["validate_feasibility"],
        timeout=30
    ))
    
    # 步骤4: 执行回滚
    workflow.add_step(WorkflowStep(
        name="执行回滚",
        agent_id="rollback_agent",
        action="execute_rollback",
        payload={
            "operation_id": operation_id,
            "rollback_type": rollback_type
        },
        dependencies=["rollback_safety_validation"],
        timeout=300,
        max_retries=2
    ))
    
    # 步骤5: 验证回滚结果
    workflow.add_step(WorkflowStep(
        name="验证回滚结果",
        agent_id="monitoring_agent",
        action="validate_rollback_result",
        payload={
            "operation_id": operation_id
        },
        dependencies=["execute_rollback"],
        timeout=120
    ))


async def create_knowledge_discovery_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建知识发现工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    time_window = parameters.get("time_window", "7d")
    analysis_type = parameters.get("analysis_type", "pattern_discovery")
    
    # 步骤1: 收集历史数据
    workflow.add_step(WorkflowStep(
        name="收集历史数据",
        agent_id="memory_agent",
        action="collect_historical_data",
        payload={
            "time_window": time_window,
            "data_types": ["sql_analysis", "optimization_results", "user_feedback"]
        },
        timeout=120
    ))
    
    # 步骤2: 数据预处理
    workflow.add_step(WorkflowStep(
        name="数据预处理",
        agent_id="learning_agent",
        action="preprocess_data",
        payload={
            "raw_data": "from_collect_historical_data",
            "preprocessing_options": parameters.get("preprocessing_options", {})
        },
        dependencies=["collect_historical_data"],
        timeout=180
    ))
    
    # 步骤3: 模式发现
    workflow.add_step(WorkflowStep(
        name="模式发现",
        agent_id="learning_agent",
        action="discover_patterns",
        payload={
            "processed_data": "from_preprocess_data",
            "analysis_type": analysis_type,
            "discovery_params": parameters.get("discovery_params", {})
        },
        dependencies=["preprocess_data"],
        timeout=300
    ))
    
    # 步骤4: 模式验证
    workflow.add_step(WorkflowStep(
        name="模式验证",
        agent_id="learning_agent",
        action="validate_patterns",
        payload={
            "discovered_patterns": "from_discover_patterns",
            "validation_criteria": parameters.get("validation_criteria", {})
        },
        dependencies=["discover_patterns"],
        timeout=120
    ))
    
    # 步骤5: 更新知识库
    workflow.add_step(WorkflowStep(
        name="更新知识库",
        agent_id="knowledge_agent",
        action="update_knowledge_base",
        payload={
            "validated_patterns": "from_validate_patterns",
            "knowledge_type": "optimization_patterns"
        },
        dependencies=["validate_patterns"],
        timeout=60
    ))


async def create_monitoring_setup_workflow(workflow: Workflow, parameters: Dict[str, Any]):
    """创建监控设置工作流模板.
    
    Args:
        workflow: 工作流对象
        parameters: 工作流参数
    """
    database = parameters.get("database")
    monitoring_type = parameters.get("monitoring_type", "performance")
    
    if not database:
        raise ValueError("缺少数据库参数")
    
    # 步骤1: 配置监控指标
    workflow.add_step(WorkflowStep(
        name="配置监控指标",
        agent_id="monitoring_agent",
        action="configure_metrics",
        payload={
            "database": database,
            "monitoring_type": monitoring_type,
            "metrics": parameters.get("metrics", [])
        },
        timeout=60
    ))
    
    # 步骤2: 设置告警阈值
    workflow.add_step(WorkflowStep(
        name="设置告警阈值",
        agent_id="monitoring_agent",
        action="configure_alert_thresholds",
        payload={
            "database": database,
            "thresholds": parameters.get("thresholds", {}),
            "alert_channels": parameters.get("alert_channels", [])
        },
        dependencies=["configure_metrics"],
        timeout=30
    ))
    
    # 步骤3: 启动监控
    workflow.add_step(WorkflowStep(
        name="启动监控",
        agent_id="monitoring_agent",
        action="start_monitoring",
        payload={
            "database": database,
            "monitoring_config": "from_previous_steps"
        },
        dependencies=["configure_thresholds"],
        timeout=30
    ))
    
    # 步骤4: 验证监控状态
    workflow.add_step(WorkflowStep(
        name="验证监控状态",
        agent_id="monitoring_agent",
        action="validate_monitoring_status",
        payload={
            "database": database
        },
        dependencies=["start_monitoring"],
        timeout=30
    ))


# 导入智能工作流
try:
    from .intelligent_workflows import (
        IntelligentSQLAnalysisWorkflow,
        FeedbackLearningWorkflow,
        KnowledgeDiscoveryWorkflow,
        AutoOptimizationWorkflow,
        IntelligentWorkflowManager
    )
    INTELLIGENT_WORKFLOWS_AVAILABLE = True
except ImportError:
    INTELLIGENT_WORKFLOWS_AVAILABLE = False

# 工作流模板注册映射
WORKFLOW_TEMPLATES = {
    "sql_analysis": create_sql_analysis_workflow,
    "optimization_execution": create_optimization_execution_workflow,
    "safety_check": create_safety_check_workflow,
    "rollback_operation": create_rollback_workflow,
    "knowledge_discovery": create_knowledge_discovery_workflow,
    "monitoring_setup": create_monitoring_setup_workflow
}


def register_workflow_templates(orchestrator):
    """注册所有工作流模板到编排器.
    
    Args:
        orchestrator: Agent编排器实例
    """
    for workflow_type, template_func in WORKFLOW_TEMPLATES.items():
        orchestrator.register_workflow_template(workflow_type, template_func)
    
    # 注册智能工作流管理器
    if INTELLIGENT_WORKFLOWS_AVAILABLE:
        intelligent_manager = IntelligentWorkflowManager(orchestrator)
        orchestrator.intelligent_workflow_manager = intelligent_manager
        logger.info("已注册智能工作流管理器")
    
    logger.info(f"已注册 {len(WORKFLOW_TEMPLATES)} 个工作流模板")