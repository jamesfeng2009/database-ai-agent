"""SQL分析能力集成模块，将现有SQL分析器集成到对话系统中."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ...agent import SQLAnalyzerAgent
from ...models import (
    ExplainResult,
    OptimizationSuggestion,
    PerformanceIssue,
    SQLAnalysisRequest,
    SQLAnalysisResponse,
)
from ..management.context_manager import ContextManager
from ..models.models import AgentResponse, IntentType, UserIntent
from ..management.task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)


class SQLAnalysisIntegrator:
    """SQL分析集成器，负责将现有SQL分析能力集成到对话系统中."""
    
    def __init__(
        self,
        sql_analyzer: SQLAnalyzerAgent,
        task_orchestrator: TaskOrchestrator,
        context_manager: ContextManager
    ):
        """初始化SQL分析集成器.
        
        Args:
            sql_analyzer: SQL分析器实例
            task_orchestrator: 任务编排器
            context_manager: 上下文管理器
        """
        self.sql_analyzer = sql_analyzer
        self.task_orchestrator = task_orchestrator
        self.context_manager = context_manager
        
        # 自然语言到SQL分析的转换模式
        self._nl_patterns = {
            "performance_analysis": [
                r"分析.*性能", r"analyze.*performance", r"性能.*分析",
                r"为什么.*慢", r"why.*slow", r"慢.*查询", r"slow.*query"
            ],
            "execution_plan": [
                r"执行计划", r"execution.*plan", r"explain", r"计划.*分析"
            ],
            "optimization": [
                r"优化.*建议", r"optimization.*suggest", r"如何.*优化", r"how.*optimize"
            ],
            "index_analysis": [
                r"索引.*分析", r"index.*analysis", r"索引.*建议", r"index.*suggest"
            ]
        }
    
    async def process_sql_analysis_intent(
        self,
        intent: UserIntent,
        session_id: str
    ) -> AgentResponse:
        """处理SQL分析意图.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            
        Returns:
            Agent响应
        """
        try:
            # 从用户输入中提取SQL语句
            sql_statement = await self._extract_sql_from_intent(intent)
            
            if not sql_statement:
                return await self._request_sql_statement(intent, session_id)
            
            # 检测分析类型
            analysis_type = await self._detect_analysis_type(intent.raw_input)
            
            # 创建SQL分析任务
            task_id = await self.task_orchestrator.create_task(
                task_type="sql_analysis",
                description=f"分析SQL查询性能: {sql_statement[:50]}...",
                parameters={
                    "sql_statement": sql_statement,
                    "analysis_type": analysis_type,
                    "session_id": session_id,
                    "user_input": intent.raw_input
                },
                session_id=session_id
            )
            
            # 执行分析任务
            success = await self.task_orchestrator.execute_task(task_id)
            
            if success:
                # 等待任务完成并获取结果
                analysis_result = await self._wait_for_analysis_result(task_id)
                
                if analysis_result:
                    # 生成自然语言解释
                    explanation = await self._generate_natural_language_explanation(
                        analysis_result, analysis_type
                    )
                    
                    # 生成交互式优化建议
                    interactive_suggestions = await self._generate_interactive_suggestions(
                        analysis_result, session_id
                    )
                    
                    return AgentResponse(
                        content=explanation,
                        intent_handled=IntentType.QUERY_ANALYSIS,
                        suggested_actions=interactive_suggestions,
                        requires_followup=len(analysis_result.suggestions) > 0,
                        metadata={
                            "task_id": task_id,
                            "analysis_result": analysis_result.dict(),
                            "sql_statement": sql_statement,
                            "analysis_type": analysis_type
                        }
                    )
                else:
                    return AgentResponse(
                        content="抱歉，SQL分析任务执行失败。请检查SQL语句是否正确。",
                        intent_handled=IntentType.QUERY_ANALYSIS,
                        suggested_actions=["重新提供SQL语句", "检查SQL语法", "联系技术支持"]
                    )
            else:
                return AgentResponse(
                    content="无法启动SQL分析任务。请稍后重试。",
                    intent_handled=IntentType.QUERY_ANALYSIS,
                    suggested_actions=["重新尝试分析", "检查系统状态"]
                )
                
        except Exception as e:
            logger.error(f"处理SQL分析意图失败: {e}")
            return AgentResponse(
                content=f"处理SQL分析请求时出现错误: {str(e)}",
                intent_handled=IntentType.QUERY_ANALYSIS,
                suggested_actions=["重新尝试", "简化SQL语句", "联系技术支持"]
            )
    
    async def _extract_sql_from_intent(self, intent: UserIntent) -> Optional[str]:
        """从用户意图中提取SQL语句.
        
        Args:
            intent: 用户意图
            
        Returns:
            提取的SQL语句，如果没有找到则返回None
        """
        # 首先检查实体中是否有SQL语句
        if "sql_statement" in intent.entities:
            return intent.entities["sql_statement"]
        
        # 从原始输入中提取SQL语句
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'(SELECT\s+.*?(?:FROM|;))',
            r'(UPDATE\s+.*?(?:SET|;))',
            r'(INSERT\s+.*?(?:VALUES|;))',
            r'(DELETE\s+.*?(?:FROM|;))',
            r'(CREATE\s+.*?(?:TABLE|INDEX|VIEW).*?(?:;|$))',
            r'(ALTER\s+.*?(?:TABLE|INDEX).*?(?:;|$))',
            r'(DROP\s+.*?(?:TABLE|INDEX|VIEW).*?(?:;|$))'
        ]
        
        for pattern in sql_patterns:
            matches = re.findall(pattern, intent.raw_input, re.IGNORECASE | re.DOTALL)
            if matches:
                sql = matches[0].strip()
                # 清理SQL语句
                sql = re.sub(r'\s+', ' ', sql)
                return sql
        
        return None
    
    async def _detect_analysis_type(self, user_input: str) -> str:
        """检测分析类型.
        
        Args:
            user_input: 用户输入
            
        Returns:
            分析类型
        """
        user_input_lower = user_input.lower()
        
        for analysis_type, patterns in self._nl_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return analysis_type
        
        return "general"
    
    async def _request_sql_statement(
        self,
        intent: UserIntent,
        session_id: str
    ) -> AgentResponse:
        """请求用户提供SQL语句.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            
        Returns:
            请求SQL语句的响应
        """
        # 检查是否提到了表名或数据库对象
        mentioned_objects = []
        if "table_name" in intent.entities:
            mentioned_objects.extend(intent.entities["table_name"])
        if "database_name" in intent.entities:
            mentioned_objects.extend(intent.entities["database_name"])
        
        content = "我需要您提供具体的SQL语句来进行分析。\n\n"
        
        if mentioned_objects:
            content += f"我注意到您提到了: {', '.join(mentioned_objects)}\n"
            content += "请提供涉及这些对象的完整SQL语句。\n\n"
        
        content += """请将SQL语句放在代码块中，例如：
```sql
SELECT * FROM your_table WHERE condition;
```

我可以帮您分析：
• 查询性能和执行计划
• 索引使用情况
• 潜在的性能问题
• 优化建议"""
        
        return AgentResponse(
            content=content,
            intent_handled=IntentType.QUERY_ANALYSIS,
            suggested_actions=[
                "提供完整的SQL语句",
                "指定要分析的表名",
                "描述遇到的性能问题",
                "查看SQL分析示例"
            ],
            requires_followup=True
        )
    
    async def _wait_for_analysis_result(
        self,
        task_id: str,
        timeout: int = 30
    ) -> Optional[SQLAnalysisResponse]:
        """等待分析结果.
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            分析结果，如果超时或失败则返回None
        """
        import asyncio
        
        for _ in range(timeout):
            task = await self.task_orchestrator.get_task(task_id)
            if not task:
                return None
            
            if task.status.value == "completed" and task.result:
                # 从任务结果中重建SQLAnalysisResponse
                return await self._rebuild_analysis_response(task.result)
            elif task.status.value == "failed":
                logger.error(f"SQL分析任务失败: {task.error}")
                return None
            
            await asyncio.sleep(1)
        
        logger.warning(f"SQL分析任务超时: {task_id}")
        return None
    
    async def _rebuild_analysis_response(
        self,
        task_result: Dict[str, Any]
    ) -> Optional[SQLAnalysisResponse]:
        """从任务结果重建SQLAnalysisResponse.
        
        Args:
            task_result: 任务结果
            
        Returns:
            重建的分析响应
        """
        try:
            # 这里需要根据实际的任务结果结构来重建
            # 暂时返回一个模拟的结果
            return SQLAnalysisResponse(
                summary=task_result.get("summary", "分析完成"),
                performance_score=task_result.get("performance_score", 75),
                issues=[],
                suggestions=[],
                detailed_analysis=task_result.get("detailed_analysis", "详细分析结果"),
                execution_plan_analysis=task_result.get("execution_plan_analysis", "执行计划分析"),
                explain_results=[]
            )
        except Exception as e:
            logger.error(f"重建分析响应失败: {e}")
            return None
    
    async def _generate_natural_language_explanation(
        self,
        analysis_result: SQLAnalysisResponse,
        analysis_type: str
    ) -> str:
        """生成自然语言解释.
        
        Args:
            analysis_result: 分析结果
            analysis_type: 分析类型
            
        Returns:
            自然语言解释
        """
        explanation = f"## SQL查询分析结果\n\n"
        
        # 性能评分解释
        score = analysis_result.performance_score
        if score >= 80:
            score_desc = "优秀"
            score_emoji = "🟢"
        elif score >= 60:
            score_desc = "良好"
            score_emoji = "🟡"
        else:
            score_desc = "需要优化"
            score_emoji = "🔴"
        
        explanation += f"### 性能评分: {score_emoji} {score}/100 ({score_desc})\n\n"
        
        # 总结
        explanation += f"**分析总结:** {analysis_result.summary}\n\n"
        
        # 根据分析类型提供不同的解释
        if analysis_type == "execution_plan":
            explanation += "### 执行计划分析\n"
            explanation += f"{analysis_result.execution_plan_analysis}\n\n"
        
        # 性能问题解释
        if analysis_result.issues:
            explanation += "### 🚨 发现的性能问题\n\n"
            for i, issue in enumerate(analysis_result.issues, 1):
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }.get(issue.severity, "⚪")
                
                explanation += f"{i}. {severity_emoji} **{issue.issue_type}** ({issue.severity})\n"
                explanation += f"   - 问题描述: {issue.description}\n"
                explanation += f"   - 性能影响: {issue.impact}\n"
                if issue.affected_tables:
                    explanation += f"   - 影响表: {', '.join(issue.affected_tables)}\n"
                explanation += "\n"
        
        # 优化建议解释
        if analysis_result.suggestions:
            explanation += "### 💡 优化建议\n\n"
            for i, suggestion in enumerate(analysis_result.suggestions, 1):
                priority_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(suggestion.priority, "⚪")
                
                explanation += f"{i}. {priority_emoji} **{suggestion.category}** (优先级: {suggestion.priority})\n"
                explanation += f"   - 建议: {suggestion.suggestion}\n"
                explanation += f"   - 预期改善: {suggestion.expected_improvement}\n"
                explanation += f"   - 实施难度: {suggestion.implementation_difficulty}\n"
                
                if suggestion.sql_example:
                    explanation += f"   - 示例SQL:\n```sql\n{suggestion.sql_example}\n```\n"
                explanation += "\n"
        
        # 详细分析
        if analysis_result.detailed_analysis:
            explanation += "### 📊 详细分析\n\n"
            explanation += f"{analysis_result.detailed_analysis}\n\n"
        
        return explanation
    
    async def _generate_interactive_suggestions(
        self,
        analysis_result: SQLAnalysisResponse,
        session_id: str
    ) -> List[str]:
        """生成交互式优化建议.
        
        Args:
            analysis_result: 分析结果
            session_id: 会话ID
            
        Returns:
            交互式建议列表
        """
        suggestions = []
        
        # 基于分析结果生成交互式建议
        if analysis_result.suggestions:
            high_priority_suggestions = [
                s for s in analysis_result.suggestions 
                if s.priority == "high"
            ]
            
            if high_priority_suggestions:
                suggestions.append("🔴 执行高优先级优化建议")
                suggestions.append("📋 查看详细的优化步骤")
            
            suggestions.append("💡 获取所有优化建议的详细说明")
            suggestions.append("🤖 让我帮您自动执行安全的优化操作")
        
        # 基于性能问题生成建议
        if analysis_result.issues:
            critical_issues = [
                i for i in analysis_result.issues 
                if i.severity == "critical"
            ]
            
            if critical_issues:
                suggestions.append("🚨 立即处理严重性能问题")
            
            suggestions.append("🔍 深入分析性能问题原因")
        
        # 通用建议
        suggestions.extend([
            "📈 设置性能监控",
            "🔄 重新分析其他SQL查询",
            "📚 学习SQL优化最佳实践",
            "❓ 询问具体的优化问题"
        ])
        
        return suggestions[:6]  # 限制建议数量
    
    async def handle_optimization_confirmation(
        self,
        intent: UserIntent,
        session_id: str
    ) -> AgentResponse:
        """处理优化建议确认.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            
        Returns:
            确认响应
        """
        # 检查用户是否确认执行优化
        user_input_lower = intent.raw_input.lower()
        
        confirmation_patterns = ["是", "yes", "好的", "ok", "确认", "执行", "同意"]
        rejection_patterns = ["不", "no", "否", "取消", "不要", "拒绝"]
        
        is_confirmed = any(pattern in user_input_lower for pattern in confirmation_patterns)
        is_rejected = any(pattern in user_input_lower for pattern in rejection_patterns)
        
        if is_confirmed:
            return await self._execute_optimization_suggestions(intent, session_id)
        elif is_rejected:
            return AgentResponse(
                content="好的，我不会执行优化操作。您可以：\n\n• 查看其他优化建议\n• 重新分析SQL查询\n• 询问具体的优化问题",
                intent_handled=IntentType.OPTIMIZATION_REQUEST,
                suggested_actions=[
                    "查看其他建议",
                    "重新分析查询", 
                    "询问优化问题",
                    "设置性能监控"
                ]
            )
        else:
            return AgentResponse(
                content="请明确告诉我是否要执行优化建议：\n\n• 回答'是'或'确认'来执行优化\n• 回答'不'或'取消'来跳过优化\n\n我会确保只执行安全的优化操作。",
                intent_handled=IntentType.OPTIMIZATION_REQUEST,
                suggested_actions=["确认执行", "取消操作", "查看详细说明"],
                requires_followup=True
            )
    
    async def _execute_optimization_suggestions(
        self,
        intent: UserIntent,
        session_id: str
    ) -> AgentResponse:
        """执行优化建议.
        
        Args:
            intent: 用户意图
            session_id: 会话ID
            
        Returns:
            执行结果响应
        """
        # 创建优化执行任务
        task_id = await self.task_orchestrator.create_task(
            task_type="optimization_execution",
            description="执行SQL优化建议",
            parameters={
                "session_id": session_id,
                "user_confirmed": True,
                "optimization_type": "sql_optimization"
            },
            session_id=session_id
        )
        
        # 启动任务执行
        success = await self.task_orchestrator.execute_task(task_id)
        
        if success:
            return AgentResponse(
                content="✅ 优化任务已启动！\n\n我正在执行以下安全的优化操作：\n• 分析索引使用情况\n• 检查查询重写机会\n• 验证优化安全性\n\n执行完成后我会向您报告结果。",
                intent_handled=IntentType.OPTIMIZATION_REQUEST,
                suggested_actions=[
                    "查看执行进度",
                    "等待执行完成",
                    "分析其他查询"
                ],
                metadata={"task_id": task_id}
            )
        else:
            return AgentResponse(
                content="❌ 无法启动优化任务。请稍后重试或联系技术支持。",
                intent_handled=IntentType.OPTIMIZATION_REQUEST,
                suggested_actions=["重新尝试", "检查系统状态", "联系支持"]
            )