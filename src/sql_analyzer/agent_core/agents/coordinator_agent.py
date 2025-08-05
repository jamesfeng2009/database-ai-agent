"""协调器Agent - 负责系统总协调、任务分发和结果聚合."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..communication.a2a_protocol import A2AMessage
from .base_agent import BaseAgent
from ..models.models import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class WorkflowStep:
    """工作流步骤定义."""
    
    def __init__(
        self,
        step_id: str,
        agent_id: str,
        action: str,
        payload: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """初始化工作流步骤.
        
        Args:
            step_id: 步骤ID
            agent_id: 执行Agent ID
            action: 操作名称
            payload: 操作参数
            dependencies: 依赖的步骤ID列表
            timeout: 超时时间
        """
        self.step_id = step_id
        self.agent_id = agent_id
        self.action = action
        self.payload = payload
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.status = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


class Workflow:
    """工作流定义."""
    
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        """初始化工作流.
        
        Args:
            workflow_id: 工作流ID
            name: 工作流名称
            description: 工作流描述
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.status = "created"
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
    
    def add_step(self, step: WorkflowStep):
        """添加工作流步骤.
        
        Args:
            step: 工作流步骤
        """
        self.steps[step.step_id] = step
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """获取可以执行的步骤（依赖已完成）.
        
        Returns:
            可执行的步骤列表
        """
        ready_steps = []
        
        for step in self.steps.values():
            if step.status != "pending":
                continue
            
            # 检查依赖是否都已完成
            dependencies_completed = all(
                self.steps[dep_id].status == "completed"
                for dep_id in step.dependencies
                if dep_id in self.steps
            )
            
            if dependencies_completed:
                ready_steps.append(step)
        
        return ready_steps
    
    def is_completed(self) -> bool:
        """检查工作流是否完成.
        
        Returns:
            是否完成
        """
        return all(
            step.status in ["completed", "failed", "skipped"]
            for step in self.steps.values()
        )
    
    def has_failed_steps(self) -> bool:
        """检查是否有失败的步骤.
        
        Returns:
            是否有失败步骤
        """
        return any(step.status == "failed" for step in self.steps.values())


class CoordinatorAgent(BaseAgent):
    """协调器Agent - 负责系统总协调、任务分发和结果聚合."""
    
    def __init__(self):
        """初始化协调器Agent."""
        super().__init__(
            agent_id="coordinator",
            agent_name="System Coordinator",
            agent_type="coordinator",
            capabilities=[
                "task_orchestration",
                "workflow_management",
                "result_aggregation",
                "system_coordination"
            ]
        )
        
        # 工作流管理
        self._workflows: Dict[str, Workflow] = {}
        self._active_workflows: Dict[str, asyncio.Task] = {}
        
        # 任务管理
        self._tasks: Dict[str, Task] = {}
        self._task_workflows: Dict[str, str] = {}  # task_id -> workflow_id
        
        # Agent状态跟踪
        self._agent_status: Dict[str, Dict[str, Any]] = {}
        
        # 预定义工作流模板
        self._workflow_templates = {
            "sql_analysis": self._create_sql_analysis_workflow,
            "optimization_execution": self._create_optimization_workflow,
            "safety_check": self._create_safety_check_workflow,
            "rollback_operation": self._create_rollback_workflow
        }
    
    async def _initialize(self):
        """初始化协调器Agent."""
        logger.info("协调器Agent初始化完成")
        
        # 订阅系统事件
        await self.subscribe_action("agent_status_update")
        await self.subscribe_action("task_completion")
        await self.subscribe_action("workflow_request")
    
    async def _cleanup(self):
        """清理协调器Agent."""
        # 取消所有活跃的工作流
        for workflow_task in self._active_workflows.values():
            workflow_task.cancel()
        
        # 等待所有任务完成
        if self._active_workflows:
            await asyncio.gather(*self._active_workflows.values(), return_exceptions=True)
        
        logger.info("协调器Agent清理完成")
    
    async def _register_custom_handlers(self):
        """注册协调器特定的消息处理器."""
        handlers = {
            "create_workflow": self._handle_create_workflow,
            "execute_task": self._handle_execute_task,
            "get_workflow_status": self._handle_get_workflow_status,
            "cancel_workflow": self._handle_cancel_workflow,
            "list_workflows": self._handle_list_workflows,
            "analyze_sql": self._handle_analyze_sql,
            "execute_optimization": self._handle_execute_optimization,
            "check_safety": self._handle_check_safety,
            "execute_rollback": self._handle_execute_rollback
        }
        
        for action, handler in handlers.items():
            self._message_handler.register_handler(action, handler)
    
    async def _handle_create_workflow(self, message: A2AMessage) -> Dict[str, Any]:
        """处理创建工作流请求.
        
        Args:
            message: 消息对象
            
        Returns:
            创建结果
        """
        try:
            workflow_type = message.payload.get("workflow_type")
            workflow_params = message.payload.get("parameters", {})
            
            if workflow_type not in self._workflow_templates:
                return {
                    "success": False,
                    "error": f"Unknown workflow type: {workflow_type}"
                }
            
            # 创建工作流
            workflow = await self._workflow_templates[workflow_type](workflow_params)
            self._workflows[workflow.workflow_id] = workflow
            
            # 启动工作流执行
            workflow_task = asyncio.create_task(self._execute_workflow(workflow))
            self._active_workflows[workflow.workflow_id] = workflow_task
            
            return {
                "success": True,
                "workflow_id": workflow.workflow_id,
                "workflow_name": workflow.name
            }
            
        except Exception as e:
            logger.error(f"创建工作流失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_execute_task(self, message: A2AMessage) -> Dict[str, Any]:
        """处理执行任务请求.
        
        Args:
            message: 消息对象
            
        Returns:
            执行结果
        """
        try:
            task_type = message.payload.get("task_type")
            task_params = message.payload.get("parameters", {})
            
            # 根据任务类型创建相应的工作流
            workflow_type = self._get_workflow_type_for_task(task_type)
            if not workflow_type:
                return {
                    "success": False,
                    "error": f"No workflow available for task type: {task_type}"
                }
            
            # 创建并执行工作流
            workflow = await self._workflow_templates[workflow_type](task_params)
            self._workflows[workflow.workflow_id] = workflow
            
            workflow_task = asyncio.create_task(self._execute_workflow(workflow))
            self._active_workflows[workflow.workflow_id] = workflow_task
            
            # 等待工作流完成
            await workflow_task
            
            return {
                "success": workflow.status == "completed",
                "workflow_id": workflow.workflow_id,
                "result": workflow.result,
                "error": workflow.error
            }
            
        except Exception as e:
            logger.error(f"执行任务失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_get_workflow_status(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取工作流状态请求.
        
        Args:
            message: 消息对象
            
        Returns:
            工作流状态
        """
        workflow_id = message.payload.get("workflow_id")
        workflow = self._workflows.get(workflow_id)
        
        if not workflow:
            return {
                "success": False,
                "error": f"Workflow not found: {workflow_id}"
            }
        
        return {
            "success": True,
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "steps": {
                step_id: {
                    "status": step.status,
                    "agent_id": step.agent_id,
                    "action": step.action,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error": step.error
                }
                for step_id, step in workflow.steps.items()
            }
        }
    
    async def _handle_cancel_workflow(self, message: A2AMessage) -> Dict[str, Any]:
        """处理取消工作流请求.
        
        Args:
            message: 消息对象
            
        Returns:
            取消结果
        """
        workflow_id = message.payload.get("workflow_id")
        
        if workflow_id in self._active_workflows:
            self._active_workflows[workflow_id].cancel()
            del self._active_workflows[workflow_id]
        
        if workflow_id in self._workflows:
            self._workflows[workflow_id].status = "cancelled"
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": "cancelled"
        }
    
    async def _handle_list_workflows(self, message: A2AMessage) -> Dict[str, Any]:
        """处理列出工作流请求.
        
        Args:
            message: 消息对象
            
        Returns:
            工作流列表
        """
        workflows = []
        
        for workflow in self._workflows.values():
            workflows.append({
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "status": workflow.status,
                "created_at": workflow.created_at.isoformat(),
                "step_count": len(workflow.steps)
            })
        
        return {
            "success": True,
            "workflows": workflows
        }
    
    # 高级业务处理器
    
    async def _handle_analyze_sql(self, message: A2AMessage) -> Dict[str, Any]:
        """处理SQL分析请求.
        
        Args:
            message: 消息对象
            
        Returns:
            分析结果
        """
        try:
            sql_statement = message.payload.get("sql_statement")
            database = message.payload.get("database")
            
            if not sql_statement:
                return {
                    "success": False,
                    "error": "Missing sql_statement parameter"
                }
            
            # 创建SQL分析工作流
            workflow = await self._create_sql_analysis_workflow({
                "sql_statement": sql_statement,
                "database": database
            })
            
            self._workflows[workflow.workflow_id] = workflow
            
            # 执行工作流
            await self._execute_workflow(workflow)
            
            return {
                "success": workflow.status == "completed",
                "workflow_id": workflow.workflow_id,
                "result": workflow.result,
                "error": workflow.error
            }
            
        except Exception as e:
            logger.error(f"SQL分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_execute_optimization(self, message: A2AMessage) -> Dict[str, Any]:
        """处理执行优化请求.
        
        Args:
            message: 消息对象
            
        Returns:
            优化结果
        """
        try:
            optimization_type = message.payload.get("optimization_type")
            target_table = message.payload.get("target_table")
            
            # 创建优化工作流
            workflow = await self._create_optimization_workflow({
                "optimization_type": optimization_type,
                "target_table": target_table
            })
            
            self._workflows[workflow.workflow_id] = workflow
            
            # 执行工作流
            await self._execute_workflow(workflow)
            
            return {
                "success": workflow.status == "completed",
                "workflow_id": workflow.workflow_id,
                "result": workflow.result,
                "error": workflow.error
            }
            
        except Exception as e:
            logger.error(f"执行优化失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_check_safety(self, message: A2AMessage) -> Dict[str, Any]:
        """处理安全检查请求.
        
        Args:
            message: 消息对象
            
        Returns:
            安全检查结果
        """
        try:
            sql_statement = message.payload.get("sql_statement")
            user_info = message.payload.get("user_info")
            
            # 创建安全检查工作流
            workflow = await self._create_safety_check_workflow({
                "sql_statement": sql_statement,
                "user_info": user_info
            })
            
            self._workflows[workflow.workflow_id] = workflow
            
            # 执行工作流
            await self._execute_workflow(workflow)
            
            return {
                "success": workflow.status == "completed",
                "workflow_id": workflow.workflow_id,
                "result": workflow.result,
                "error": workflow.error
            }
            
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_execute_rollback(self, message: A2AMessage) -> Dict[str, Any]:
        """处理执行回滚请求.
        
        Args:
            message: 消息对象
            
        Returns:
            回滚结果
        """
        try:
            operation_id = message.payload.get("operation_id")
            rollback_type = message.payload.get("rollback_type", "automatic")
            
            # 创建回滚工作流
            workflow = await self._create_rollback_workflow({
                "operation_id": operation_id,
                "rollback_type": rollback_type
            })
            
            self._workflows[workflow.workflow_id] = workflow
            
            # 执行工作流
            await self._execute_workflow(workflow)
            
            return {
                "success": workflow.status == "completed",
                "workflow_id": workflow.workflow_id,
                "result": workflow.result,
                "error": workflow.error
            }
            
        except Exception as e:
            logger.error(f"执行回滚失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # 工作流模板创建方法
    
    async def _create_sql_analysis_workflow(self, params: Dict[str, Any]) -> Workflow:
        """创建SQL分析工作流.
        
        Args:
            params: 工作流参数
            
        Returns:
            工作流对象
        """
        workflow_id = str(uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name="SQL Analysis Workflow",
            description="Comprehensive SQL analysis workflow"
        )
        
        # 步骤1: NLP处理（如果需要）
        if params.get("natural_language_query"):
            workflow.add_step(WorkflowStep(
                step_id="nlp_processing",
                agent_id="nlp_agent",
                action="process_natural_language",
                payload={
                    "query": params.get("natural_language_query"),
                    "context": params.get("context", {})
                }
            ))
        
        # 步骤2: SQL分析
        workflow.add_step(WorkflowStep(
            step_id="sql_analysis",
            agent_id="sql_analysis_agent",
            action="analyze_sql",
            payload={
                "sql_statement": params.get("sql_statement"),
                "database": params.get("database"),
                "explain_results": params.get("explain_results", [])
            },
            dependencies=["nlp_processing"] if params.get("natural_language_query") else []
        ))
        
        # 步骤3: 安全检查
        workflow.add_step(WorkflowStep(
            step_id="safety_check",
            agent_id="safety_agent",
            action="validate_sql",
            payload={
                "sql_statement": params.get("sql_statement"),
                "user_info": params.get("user_info", {})
            },
            dependencies=["sql_analysis"]
        ))
        
        # 步骤4: 优化建议生成
        workflow.add_step(WorkflowStep(
            step_id="optimization_suggestions",
            agent_id="optimization_agent",
            action="generate_suggestions",
            payload={
                "sql_statement": params.get("sql_statement"),
                "analysis_result": "from_sql_analysis"
            },
            dependencies=["sql_analysis", "safety_check"]
        ))
        
        return workflow
    
    async def _create_optimization_workflow(self, params: Dict[str, Any]) -> Workflow:
        """创建优化执行工作流.
        
        Args:
            params: 工作流参数
            
        Returns:
            工作流对象
        """
        workflow_id = str(uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name="Optimization Execution Workflow",
            description="Database optimization execution workflow"
        )
        
        # 步骤1: 创建快照
        workflow.add_step(WorkflowStep(
            step_id="create_snapshot",
            agent_id="rollback_agent",
            action="create_snapshot",
            payload={
                "operation_id": workflow_id,
                "target_objects": [params.get("target_table")]
            }
        ))
        
        # 步骤2: 安全验证
        workflow.add_step(WorkflowStep(
            step_id="safety_validation",
            agent_id="safety_agent",
            action="validate_optimization",
            payload={
                "optimization_type": params.get("optimization_type"),
                "target_table": params.get("target_table")
            },
            dependencies=["create_snapshot"]
        ))
        
        # 步骤3: 执行优化
        workflow.add_step(WorkflowStep(
            step_id="execute_optimization",
            agent_id="optimization_agent",
            action="execute_optimization",
            payload=params,
            dependencies=["safety_validation"]
        ))
        
        # 步骤4: 验证结果
        workflow.add_step(WorkflowStep(
            step_id="validate_result",
            agent_id="monitoring_agent",
            action="validate_optimization_result",
            payload={
                "optimization_id": workflow_id,
                "target_table": params.get("target_table")
            },
            dependencies=["execute_optimization"]
        ))
        
        return workflow
    
    async def _create_safety_check_workflow(self, params: Dict[str, Any]) -> Workflow:
        """创建安全检查工作流.
        
        Args:
            params: 工作流参数
            
        Returns:
            工作流对象
        """
        workflow_id = str(uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name="Safety Check Workflow",
            description="Comprehensive safety validation workflow"
        )
        
        # 步骤1: SQL安全检查
        workflow.add_step(WorkflowStep(
            step_id="sql_safety_check",
            agent_id="safety_agent",
            action="check_sql_safety",
            payload={
                "sql_statement": params.get("sql_statement"),
                "user_info": params.get("user_info")
            }
        ))
        
        # 步骤2: 权限验证
        workflow.add_step(WorkflowStep(
            step_id="permission_check",
            agent_id="safety_agent",
            action="check_permissions",
            payload={
                "user_info": params.get("user_info"),
                "required_permissions": params.get("required_permissions", [])
            },
            dependencies=["sql_safety_check"]
        ))
        
        # 步骤3: 风险评估
        workflow.add_step(WorkflowStep(
            step_id="risk_assessment",
            agent_id="safety_agent",
            action="assess_risk",
            payload={
                "sql_statement": params.get("sql_statement"),
                "operation_context": params.get("context", {})
            },
            dependencies=["permission_check"]
        ))
        
        return workflow
    
    async def _create_rollback_workflow(self, params: Dict[str, Any]) -> Workflow:
        """创建回滚工作流.
        
        Args:
            params: 工作流参数
            
        Returns:
            工作流对象
        """
        workflow_id = str(uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name="Rollback Operation Workflow",
            description="Database operation rollback workflow"
        )
        
        # 步骤1: 获取回滚计划
        workflow.add_step(WorkflowStep(
            step_id="get_rollback_plan",
            agent_id="rollback_agent",
            action="get_rollback_plan",
            payload={
                "operation_id": params.get("operation_id")
            }
        ))
        
        # 步骤2: 安全验证
        workflow.add_step(WorkflowStep(
            step_id="validate_rollback",
            agent_id="safety_agent",
            action="validate_rollback",
            payload={
                "rollback_plan": "from_get_rollback_plan"
            },
            dependencies=["get_rollback_plan"]
        ))
        
        # 步骤3: 执行回滚
        workflow.add_step(WorkflowStep(
            step_id="execute_rollback",
            agent_id="rollback_agent",
            action="execute_rollback",
            payload={
                "operation_id": params.get("operation_id"),
                "rollback_type": params.get("rollback_type")
            },
            dependencies=["validate_rollback"]
        ))
        
        # 步骤4: 验证回滚结果
        workflow.add_step(WorkflowStep(
            step_id="validate_rollback_result",
            agent_id="monitoring_agent",
            action="validate_rollback_result",
            payload={
                "operation_id": params.get("operation_id")
            },
            dependencies=["execute_rollback"]
        ))
        
        return workflow
    
    async def _execute_workflow(self, workflow: Workflow):
        """执行工作流.
        
        Args:
            workflow: 工作流对象
        """
        try:
            workflow.status = "running"
            workflow.started_at = datetime.now()
            
            logger.info(f"开始执行工作流: {workflow.workflow_id} ({workflow.name})")
            
            while not workflow.is_completed():
                # 获取可以执行的步骤
                ready_steps = workflow.get_ready_steps()
                
                if not ready_steps:
                    # 没有可执行的步骤，检查是否有失败的步骤
                    if workflow.has_failed_steps():
                        workflow.status = "failed"
                        workflow.error = "Workflow has failed steps"
                        break
                    else:
                        # 等待其他步骤完成
                        await asyncio.sleep(1)
                        continue
                
                # 并行执行所有可执行的步骤
                step_tasks = []
                for step in ready_steps:
                    step_task = asyncio.create_task(self._execute_workflow_step(step))
                    step_tasks.append(step_task)
                
                # 等待所有步骤完成
                if step_tasks:
                    await asyncio.gather(*step_tasks, return_exceptions=True)
            
            # 聚合结果
            if workflow.status != "failed":
                workflow.result = self._aggregate_workflow_results(workflow)
                workflow.status = "completed"
            
            workflow.completed_at = datetime.now()
            
            logger.info(f"工作流执行完成: {workflow.workflow_id}, 状态: {workflow.status}")
            
        except Exception as e:
            workflow.status = "failed"
            workflow.error = str(e)
            workflow.completed_at = datetime.now()
            logger.error(f"工作流执行失败: {workflow.workflow_id}, 错误: {e}")
        finally:
            # 清理活跃工作流记录
            if workflow.workflow_id in self._active_workflows:
                del self._active_workflows[workflow.workflow_id]
    
    async def _execute_workflow_step(self, step: WorkflowStep):
        """执行工作流步骤.
        
        Args:
            step: 工作流步骤
        """
        try:
            step.status = "running"
            step.started_at = datetime.now()
            
            logger.debug(f"执行工作流步骤: {step.step_id} -> {step.agent_id}.{step.action}")
            
            # 发送请求给目标Agent
            response = await self.send_request(
                step.agent_id,
                step.action,
                step.payload,
                step.timeout
            )
            
            if response and response.payload.get("result"):
                step.result = response.payload["result"]
                step.status = "completed"
            else:
                step.status = "failed"
                step.error = response.payload.get("error", "Unknown error") if response else "No response"
            
            step.completed_at = datetime.now()
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"工作流步骤执行失败: {step.step_id}, 错误: {e}")
    
    def _aggregate_workflow_results(self, workflow: Workflow) -> Dict[str, Any]:
        """聚合工作流结果.
        
        Args:
            workflow: 工作流对象
            
        Returns:
            聚合的结果
        """
        results = {}
        
        for step_id, step in workflow.steps.items():
            if step.result:
                results[step_id] = step.result
        
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "execution_time": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at and workflow.started_at else 0,
            "step_results": results,
            "summary": self._generate_workflow_summary(workflow)
        }
    
    def _generate_workflow_summary(self, workflow: Workflow) -> str:
        """生成工作流摘要.
        
        Args:
            workflow: 工作流对象
            
        Returns:
            工作流摘要
        """
        completed_steps = sum(1 for step in workflow.steps.values() if step.status == "completed")
        failed_steps = sum(1 for step in workflow.steps.values() if step.status == "failed")
        total_steps = len(workflow.steps)
        
        return f"工作流 {workflow.name} 执行完成。总步骤: {total_steps}, 成功: {completed_steps}, 失败: {failed_steps}"
    
    def _get_workflow_type_for_task(self, task_type: str) -> Optional[str]:
        """根据任务类型获取对应的工作流类型.
        
        Args:
            task_type: 任务类型
            
        Returns:
            工作流类型
        """
        task_workflow_mapping = {
            "sql_analysis": "sql_analysis",
            "optimization_execution": "optimization_execution",
            "safety_check": "safety_check",
            "rollback_operation": "rollback_operation"
        }
        
        return task_workflow_mapping.get(task_type)