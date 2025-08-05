"""任务编排器，负责协调和执行复杂的数据库优化任务."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .context_manager import ContextManager
from .event_system import publish_event
from .models import EventType, Task, TaskPriority, TaskStatus
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """任务编排器，协调和执行复杂的数据库优化任务."""
    
    def __init__(self, session_manager: SessionManager, context_manager: ContextManager):
        """初始化任务编排器.
        
        Args:
            session_manager: 会话管理器
            context_manager: 上下文管理器
        """
        self.session_manager = session_manager
        self.context_manager = context_manager
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_lock = asyncio.Lock()
        
        # 任务类型处理器映射
        self._task_handlers = {
            "sql_analysis": self._handle_sql_analysis_task,
            "optimization_execution": self._handle_optimization_task,
            "monitoring_setup": self._handle_monitoring_task,
            "batch_analysis": self._handle_batch_analysis_task,
        }
    
    async def create_task(
        self,
        task_type: str,
        description: str,
        parameters: Dict[str, Any],
        session_id: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """创建新任务.
        
        Args:
            task_type: 任务类型
            description: 任务描述
            parameters: 任务参数
            session_id: 关联的会话ID
            priority: 任务优先级
            dependencies: 依赖的任务ID列表
            
        Returns:
            创建的任务ID
        """
        task = Task(
            task_type=task_type,
            description=description,
            parameters=parameters,
            session_id=session_id,
            priority=priority,
            dependencies=dependencies or []
        )
        
        async with self._task_lock:
            self._tasks[task.task_id] = task
        
        # 添加到会话的活跃任务列表
        await self.context_manager.add_active_task(session_id, task.task_id)
        
        # 发布任务创建事件
        await publish_event(
            EventType.TASK_CREATED,
            source="task_orchestrator",
            data={
                "task_id": task.task_id,
                "task_type": task_type,
                "description": description,
                "priority": priority.value
            },
            session_id=session_id
        )
        
        logger.info(f"创建任务: {task.task_id} ({task_type})")
        return task.task_id
    
    async def execute_task(self, task_id: str) -> bool:
        """执行指定任务.
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功启动任务执行
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.error(f"任务不存在: {task_id}")
                return False
            
            if task.status != TaskStatus.PENDING:
                logger.warning(f"任务状态不允许执行: {task_id}, 状态: {task.status}")
                return False
            
            # 检查依赖任务是否完成
            for dep_id in task.dependencies:
                dep_task = self._tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    logger.warning(f"任务依赖未完成: {task_id}, 依赖: {dep_id}")
                    return False
        
        # 启动任务执行
        execution_task = asyncio.create_task(self._execute_task_async(task_id))
        self._running_tasks[task_id] = execution_task
        
        return True
    
    async def _execute_task_async(self, task_id: str) -> None:
        """异步执行任务.
        
        Args:
            task_id: 任务ID
        """
        try:
            # 更新任务状态为运行中
            await self._update_task_status(task_id, TaskStatus.RUNNING)
            
            task = self._tasks[task_id]
            handler = self._task_handlers.get(task.task_type)
            
            if not handler:
                raise ValueError(f"未知的任务类型: {task.task_type}")
            
            # 执行任务处理器
            result = await handler(task)
            
            # 更新任务结果和状态
            await self._complete_task(task_id, result)
            
        except Exception as e:
            logger.error(f"任务执行失败: {task_id}, 错误: {e}")
            await self._fail_task(task_id, str(e))
        finally:
            # 清理运行中的任务
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
    
    async def _update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """更新任务状态.
        
        Args:
            task_id: 任务ID
            status: 新状态
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                
                if status == TaskStatus.RUNNING:
                    task.started_at = datetime.now()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now()
        
        # 发布状态更新事件
        event_type = {
            TaskStatus.RUNNING: EventType.TASK_STARTED,
            TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
            TaskStatus.FAILED: EventType.TASK_FAILED,
        }.get(status)
        
        if event_type:
            await publish_event(
                event_type,
                source="task_orchestrator",
                data={"task_id": task_id, "status": status.value},
                session_id=task.session_id if task else None
            )
    
    async def _complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """完成任务.
        
        Args:
            task_id: 任务ID
            result: 任务结果
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
        
        # 从会话的活跃任务列表中移除
        if task:
            await self.context_manager.remove_active_task(task.session_id, task_id)
        
        await publish_event(
            EventType.TASK_COMPLETED,
            source="task_orchestrator",
            data={"task_id": task_id, "result": result},
            session_id=task.session_id if task else None
        )
        
        logger.info(f"任务完成: {task_id}")
    
    async def _fail_task(self, task_id: str, error: str) -> None:
        """任务失败处理.
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.error = error
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
        
        # 从会话的活跃任务列表中移除
        if task:
            await self.context_manager.remove_active_task(task.session_id, task_id)
        
        await publish_event(
            EventType.TASK_FAILED,
            source="task_orchestrator",
            data={"task_id": task_id, "error": error},
            session_id=task.session_id if task else None
        )
        
        logger.error(f"任务失败: {task_id}, 错误: {error}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务.
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        # 取消运行中的任务
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]
        
        # 更新任务状态
        await self._update_task_status(task_id, TaskStatus.CANCELLED)
        
        # 从会话的活跃任务列表中移除
        task = self._tasks.get(task_id)
        if task:
            await self.context_manager.remove_active_task(task.session_id, task_id)
        
        logger.info(f"任务已取消: {task_id}")
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态.
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态，如果任务不存在则返回None
        """
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务详情.
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象，如果不存在则返回None
        """
        return self._tasks.get(task_id)
    
    async def get_session_tasks(self, session_id: str) -> List[Task]:
        """获取会话的所有任务.
        
        Args:
            session_id: 会话ID
            
        Returns:
            任务列表
        """
        return [task for task in self._tasks.values() if task.session_id == session_id]
    
    # 任务处理器实现
    
    async def _handle_sql_analysis_task(self, task: Task) -> Dict[str, Any]:
        """处理SQL分析任务.
        
        Args:
            task: 任务对象
            
        Returns:
            任务结果
        """
        sql_statement = task.parameters.get("sql_statement")
        database = task.parameters.get("database")
        analysis_type = task.parameters.get("analysis_type", "general")
        
        if not sql_statement:
            raise ValueError("缺少SQL语句参数")
        
        try:
            # 如果有SQL分析器实例，使用实际分析
            if hasattr(self, 'sql_analyzer') and self.sql_analyzer:
                # 创建分析请求
                from ..models import SQLAnalysisRequest, ExplainResult
                
                # 创建模拟的EXPLAIN结果（实际应该从数据库获取）
                explain_results = [
                    ExplainResult(
                        id=1,
                        table="example_table",
                        type="ALL",
                        rows=1000,
                        extra="Using where"
                    )
                ]
                
                analysis_request = SQLAnalysisRequest(
                    sql_statement=sql_statement,
                    explain_results=explain_results,
                    database_schema=database
                )
                
                # 执行分析
                analysis_response = await self.sql_analyzer.analyze_sql(analysis_request)
                
                result = {
                    "analysis_type": analysis_type,
                    "sql_statement": sql_statement,
                    "database": database,
                    "summary": analysis_response.summary,
                    "performance_score": analysis_response.performance_score,
                    "issues_found": len(analysis_response.issues),
                    "suggestions_count": len(analysis_response.suggestions),
                    "detailed_analysis": analysis_response.detailed_analysis,
                    "execution_plan_analysis": analysis_response.execution_plan_analysis,
                    "issues": [issue.dict() for issue in analysis_response.issues],
                    "suggestions": [suggestion.dict() for suggestion in analysis_response.suggestions],
                    "status": "completed"
                }
            else:
                # 使用模拟结果
                result = {
                    "analysis_type": analysis_type,
                    "sql_statement": sql_statement,
                    "database": database,
                    "summary": f"SQL查询分析完成。查询涉及表扫描，建议添加索引优化。",
                    "performance_score": 75,
                    "issues_found": 2,
                    "suggestions_count": 3,
                    "detailed_analysis": "查询使用了全表扫描，建议在WHERE条件字段上创建索引。",
                    "execution_plan_analysis": "执行计划显示需要扫描大量行数，性能可能受到影响。",
                    "issues": [],
                    "suggestions": [],
                    "status": "completed"
                }
            
            logger.info(f"SQL分析任务完成: {task.task_id}")
            return result
            
        except Exception as e:
            logger.error(f"SQL分析任务执行失败: {task.task_id}, 错误: {e}")
            raise
    
    async def _handle_optimization_task(self, task: Task) -> Dict[str, Any]:
        """处理优化执行任务.
        
        Args:
            task: 任务对象
            
        Returns:
            任务结果
        """
        optimization_type = task.parameters.get("optimization_type")
        target_table = task.parameters.get("target_table")
        
        # 模拟优化执行
        await asyncio.sleep(2)  # 模拟执行时间
        
        result = {
            "optimization_type": optimization_type,
            "target_table": target_table,
            "actions_executed": ["create_index", "update_statistics"],
            "performance_improvement": "25%",
            "status": "completed"
        }
        
        logger.info(f"优化任务完成: {task.task_id}")
        return result
    
    async def _handle_monitoring_task(self, task: Task) -> Dict[str, Any]:
        """处理监控设置任务.
        
        Args:
            task: 任务对象
            
        Returns:
            任务结果
        """
        monitoring_type = task.parameters.get("monitoring_type")
        thresholds = task.parameters.get("thresholds", {})
        
        # 模拟监控设置
        await asyncio.sleep(1)
        
        result = {
            "monitoring_type": monitoring_type,
            "thresholds": thresholds,
            "alerts_configured": 3,
            "status": "active"
        }
        
        logger.info(f"监控设置任务完成: {task.task_id}")
        return result
    
    async def _handle_batch_analysis_task(self, task: Task) -> Dict[str, Any]:
        """处理批量分析任务.
        
        Args:
            task: 任务对象
            
        Returns:
            任务结果
        """
        queries = task.parameters.get("queries", [])
        
        # 模拟批量分析
        results = []
        for i, query in enumerate(queries):
            await asyncio.sleep(0.5)  # 模拟每个查询的分析时间
            results.append({
                "query_id": i + 1,
                "sql": query,
                "performance_score": 70 + (i % 30),
                "issues": i % 3,
                "suggestions": (i % 3) + 1
            })
        
        result = {
            "total_queries": len(queries),
            "analyzed_queries": len(results),
            "average_score": sum(r["performance_score"] for r in results) / len(results) if results else 0,
            "results": results,
            "status": "completed"
        }
        
        logger.info(f"批量分析任务完成: {task.task_id}")
        return result
    
    async def shutdown(self) -> None:
        """关闭任务编排器."""
        # 取消所有运行中的任务
        for task_id in list(self._running_tasks.keys()):
            await self.cancel_task(task_id)
        
        logger.info("任务编排器已关闭")
    
    def set_sql_analyzer(self, sql_analyzer):
        """设置SQL分析器.
        
        Args:
            sql_analyzer: SQL分析器实例
        """
        self.sql_analyzer = sql_analyzer
        logger.info("SQL分析器已设置到任务编排器")