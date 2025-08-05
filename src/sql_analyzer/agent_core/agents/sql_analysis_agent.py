"""SQL分析Agent - 负责SQL查询分析和性能评估."""

import logging
from typing import Any, Dict, List, Optional

from ..communication.a2a_protocol import A2AMessage
from .base_agent import BaseAgent
from ...models import SQLAnalysisRequest, SQLAnalysisResponse, ExplainResult
from ...tools import (
    detect_performance_issues,
    generate_optimization_suggestions,
    calculate_performance_score,
    format_analysis_request
)

logger = logging.getLogger(__name__)


class SQLAnalysisAgent(BaseAgent):
    """SQL分析Agent - 负责SQL查询分析和性能评估."""
    
    def __init__(self):
        """初始化SQL分析Agent."""
        super().__init__(
            agent_id="sql_analysis_agent",
            agent_name="SQL Analysis Agent",
            agent_type="sql_analysis",
            capabilities=[
                "sql_parsing",
                "performance_analysis",
                "execution_plan_analysis",
                "query_complexity_assessment",
                "issue_detection"
            ]
        )
        
        # 分析统计
        self._analysis_count = 0
        self._total_analysis_time = 0.0
        self._issue_statistics = {}
    
    async def _initialize(self):
        """初始化SQL分析Agent."""
        logger.info("SQL分析Agent初始化完成")
    
    async def _cleanup(self):
        """清理SQL分析Agent."""
        logger.info("SQL分析Agent清理完成")
    
    async def _register_custom_handlers(self):
        """注册SQL分析Agent特定的消息处理器."""
        handlers = {
            "analyze_sql": self._handle_analyze_sql,
            "parse_sql": self._handle_parse_sql,
            "analyze_execution_plan": self._handle_analyze_execution_plan,
            "detect_issues": self._handle_detect_issues,
            "calculate_complexity": self._handle_calculate_complexity,
            "get_analysis_stats": self._handle_get_analysis_stats
        }
        
        for action, handler in handlers.items():
            self._message_handler.register_handler(action, handler)
    
    async def _handle_analyze_sql(self, message: A2AMessage) -> Dict[str, Any]:
        """处理SQL分析请求.
        
        Args:
            message: 消息对象
            
        Returns:
            分析结果
        """
        try:
            start_time = message.timestamp
            
            # 提取参数
            sql_statement = message.payload.get("sql_statement", "")
            database = message.payload.get("database", "")
            explain_results = message.payload.get("explain_results", [])
            
            if not sql_statement:
                return {
                    "success": False,
                    "error": "Missing sql_statement parameter"
                }
            
            # 创建分析请求
            analysis_request = self._create_analysis_request(
                sql_statement, database, explain_results
            )
            
            # 执行分析
            analysis_result = await self._perform_analysis(analysis_request)
            
            # 更新统计信息
            self._analysis_count += 1
            analysis_time = (message.timestamp - start_time).total_seconds()
            self._total_analysis_time += analysis_time
            
            # 更新问题统计
            for issue in analysis_result.get("issues", []):
                issue_type = issue.get("type", "unknown")
                self._issue_statistics[issue_type] = self._issue_statistics.get(issue_type, 0) + 1
            
            return {
                "success": True,
                "analysis_result": analysis_result,
                "analysis_time": analysis_time
            }
            
        except Exception as e:
            logger.error(f"SQL分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_parse_sql(self, message: A2AMessage) -> Dict[str, Any]:
        """处理SQL解析请求.
        
        Args:
            message: 消息对象
            
        Returns:
            解析结果
        """
        try:
            sql_statement = message.payload.get("sql_statement", "")
            
            if not sql_statement:
                return {
                    "success": False,
                    "error": "Missing sql_statement parameter"
                }
            
            # 解析SQL语句
            parsed_result = await self._parse_sql_statement(sql_statement)
            
            return {
                "success": True,
                "parsed_result": parsed_result
            }
            
        except Exception as e:
            logger.error(f"SQL解析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_analyze_execution_plan(self, message: A2AMessage) -> Dict[str, Any]:
        """处理执行计划分析请求.
        
        Args:
            message: 消息对象
            
        Returns:
            执行计划分析结果
        """
        try:
            explain_results = message.payload.get("explain_results", [])
            
            if not explain_results:
                return {
                    "success": False,
                    "error": "Missing explain_results parameter"
                }
            
            # 分析执行计划
            plan_analysis = await self._analyze_execution_plan(explain_results)
            
            return {
                "success": True,
                "plan_analysis": plan_analysis
            }
            
        except Exception as e:
            logger.error(f"执行计划分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_detect_issues(self, message: A2AMessage) -> Dict[str, Any]:
        """处理问题检测请求.
        
        Args:
            message: 消息对象
            
        Returns:
            问题检测结果
        """
        try:
            sql_statement = message.payload.get("sql_statement", "")
            explain_results = message.payload.get("explain_results", [])
            
            if not sql_statement:
                return {
                    "success": False,
                    "error": "Missing sql_statement parameter"
                }
            
            # 创建分析请求
            analysis_request = self._create_analysis_request(sql_statement, "", explain_results)
            
            # 检测性能问题
            issues = detect_performance_issues(analysis_request)
            
            return {
                "success": True,
                "issues": [issue.dict() for issue in issues]
            }
            
        except Exception as e:
            logger.error(f"问题检测失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_calculate_complexity(self, message: A2AMessage) -> Dict[str, Any]:
        """处理复杂度计算请求.
        
        Args:
            message: 消息对象
            
        Returns:
            复杂度计算结果
        """
        try:
            sql_statement = message.payload.get("sql_statement", "")
            
            if not sql_statement:
                return {
                    "success": False,
                    "error": "Missing sql_statement parameter"
                }
            
            # 计算查询复杂度
            complexity = await self._calculate_query_complexity(sql_statement)
            
            return {
                "success": True,
                "complexity": complexity
            }
            
        except Exception as e:
            logger.error(f"复杂度计算失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_get_analysis_stats(self, message: A2AMessage) -> Dict[str, Any]:
        """处理获取分析统计请求.
        
        Args:
            message: 消息对象
            
        Returns:
            分析统计信息
        """
        try:
            avg_analysis_time = (
                self._total_analysis_time / self._analysis_count 
                if self._analysis_count > 0 else 0
            )
            
            return {
                "success": True,
                "stats": {
                    "total_analyses": self._analysis_count,
                    "total_analysis_time": self._total_analysis_time,
                    "average_analysis_time": avg_analysis_time,
                    "issue_statistics": self._issue_statistics.copy()
                }
            }
            
        except Exception as e:
            logger.error(f"获取分析统计失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_analysis_request(
        self, 
        sql_statement: str, 
        database: str, 
        explain_results: List[Dict[str, Any]]
    ) -> SQLAnalysisRequest:
        """创建SQL分析请求.
        
        Args:
            sql_statement: SQL语句
            database: 数据库名
            explain_results: 执行计划结果
            
        Returns:
            SQL分析请求对象
        """
        # 转换执行计划结果
        explain_objects = []
        for result in explain_results:
            explain_objects.append(ExplainResult(**result))
        
        return SQLAnalysisRequest(
            sql_statement=sql_statement,
            database_schema=database,
            explain_results=explain_objects
        )
    
    async def _perform_analysis(self, request: SQLAnalysisRequest) -> Dict[str, Any]:
        """执行SQL分析.
        
        Args:
            request: 分析请求
            
        Returns:
            分析结果
        """
        # 检测性能问题
        issues = detect_performance_issues(request)
        
        # 生成优化建议
        suggestions = generate_optimization_suggestions(request, issues)
        
        # 计算性能得分
        score = calculate_performance_score(request, issues)
        
        # 分析执行计划
        execution_plan_analysis = await self._analyze_execution_plan(request.explain_results)
        
        # 生成总结
        summary = self._generate_analysis_summary(issues, suggestions, score)
        
        return {
            "summary": summary,
            "performance_score": score,
            "issues": [issue.dict() for issue in issues],
            "suggestions": [suggestion.dict() for suggestion in suggestions],
            "execution_plan_analysis": execution_plan_analysis,
            "query_complexity": await self._calculate_query_complexity(request.sql_statement),
            "analysis_metadata": {
                "sql_statement": request.sql_statement,
                "database_schema": request.database_schema,
                "analysis_timestamp": message.timestamp.isoformat() if hasattr(self, 'message') else None
            }
        }
    
    async def _parse_sql_statement(self, sql_statement: str) -> Dict[str, Any]:
        """解析SQL语句.
        
        Args:
            sql_statement: SQL语句
            
        Returns:
            解析结果
        """
        # 简单的SQL解析逻辑
        sql_lower = sql_statement.lower().strip()
        
        # 识别SQL类型
        sql_type = "unknown"
        if sql_lower.startswith("select"):
            sql_type = "select"
        elif sql_lower.startswith("insert"):
            sql_type = "insert"
        elif sql_lower.startswith("update"):
            sql_type = "update"
        elif sql_lower.startswith("delete"):
            sql_type = "delete"
        elif sql_lower.startswith("create"):
            sql_type = "create"
        elif sql_lower.startswith("drop"):
            sql_type = "drop"
        elif sql_lower.startswith("alter"):
            sql_type = "alter"
        
        # 提取表名
        tables = self._extract_table_names(sql_statement)
        
        # 提取列名
        columns = self._extract_column_names(sql_statement)
        
        # 检查是否有子查询
        has_subquery = "(" in sql_statement and "select" in sql_lower
        
        # 检查是否有JOIN
        has_join = any(join_type in sql_lower for join_type in ["join", "inner join", "left join", "right join", "full join"])
        
        return {
            "sql_type": sql_type,
            "tables": tables,
            "columns": columns,
            "has_subquery": has_subquery,
            "has_join": has_join,
            "statement_length": len(sql_statement),
            "word_count": len(sql_statement.split())
        }
    
    async def _analyze_execution_plan(self, explain_results: List[ExplainResult]) -> Dict[str, Any]:
        """分析执行计划.
        
        Args:
            explain_results: 执行计划结果
            
        Returns:
            执行计划分析结果
        """
        if not explain_results:
            return {"analysis": "No execution plan available"}
        
        analysis = {
            "total_steps": len(explain_results),
            "scan_types": [],
            "total_rows": 0,
            "has_full_table_scan": False,
            "has_index_usage": False,
            "performance_concerns": []
        }
        
        for result in explain_results:
            # 分析扫描类型
            if result.type:
                analysis["scan_types"].append(result.type)
                
                if result.type.upper() == "ALL":
                    analysis["has_full_table_scan"] = True
                    analysis["performance_concerns"].append("Full table scan detected")
                elif "INDEX" in result.type.upper():
                    analysis["has_index_usage"] = True
            
            # 累计行数
            if result.rows:
                analysis["total_rows"] += result.rows
            
            # 检查额外信息
            if result.extra:
                if "Using temporary" in result.extra:
                    analysis["performance_concerns"].append("Using temporary table")
                if "Using filesort" in result.extra:
                    analysis["performance_concerns"].append("Using filesort")
        
        # 生成分析总结
        if analysis["has_full_table_scan"]:
            analysis["recommendation"] = "Consider adding indexes to avoid full table scans"
        elif analysis["has_index_usage"]:
            analysis["recommendation"] = "Query is using indexes effectively"
        else:
            analysis["recommendation"] = "Review query structure and indexing strategy"
        
        return analysis
    
    async def _calculate_query_complexity(self, sql_statement: str) -> Dict[str, Any]:
        """计算查询复杂度.
        
        Args:
            sql_statement: SQL语句
            
        Returns:
            复杂度信息
        """
        sql_lower = sql_statement.lower()
        
        # 基础复杂度指标
        complexity_score = 0
        factors = []
        
        # 语句长度
        length_score = min(len(sql_statement) // 100, 10)
        complexity_score += length_score
        if length_score > 5:
            factors.append(f"Long statement ({len(sql_statement)} characters)")
        
        # JOIN数量
        join_count = sql_lower.count("join")
        complexity_score += join_count * 2
        if join_count > 0:
            factors.append(f"{join_count} JOIN operations")
        
        # 子查询数量
        subquery_count = sql_lower.count("select") - 1  # 减去主查询
        complexity_score += subquery_count * 3
        if subquery_count > 0:
            factors.append(f"{subquery_count} subqueries")
        
        # 聚合函数
        aggregate_functions = ["count", "sum", "avg", "max", "min", "group_concat"]
        aggregate_count = sum(sql_lower.count(func) for func in aggregate_functions)
        complexity_score += aggregate_count
        if aggregate_count > 0:
            factors.append(f"{aggregate_count} aggregate functions")
        
        # UNION操作
        union_count = sql_lower.count("union")
        complexity_score += union_count * 2
        if union_count > 0:
            factors.append(f"{union_count} UNION operations")
        
        # 确定复杂度等级
        if complexity_score <= 5:
            complexity_level = "Low"
        elif complexity_score <= 15:
            complexity_level = "Medium"
        elif complexity_score <= 30:
            complexity_level = "High"
        else:
            complexity_level = "Very High"
        
        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "contributing_factors": factors,
            "recommendations": self._get_complexity_recommendations(complexity_level)
        }
    
    def _extract_table_names(self, sql_statement: str) -> List[str]:
        """提取SQL语句中的表名.
        
        Args:
            sql_statement: SQL语句
            
        Returns:
            表名列表
        """
        # 简单的表名提取逻辑
        import re
        
        # 匹配FROM和JOIN后的表名
        pattern = r'\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(pattern, sql_statement, re.IGNORECASE)
        
        return list(set(matches))  # 去重
    
    def _extract_column_names(self, sql_statement: str) -> List[str]:
        """提取SQL语句中的列名.
        
        Args:
            sql_statement: SQL语句
            
        Returns:
            列名列表
        """
        # 简单的列名提取逻辑
        import re
        
        columns = []
        
        # 匹配SELECT后的列名
        select_match = re.search(r'select\s+(.*?)\s+from', sql_statement, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_part = select_match.group(1)
            if select_part.strip() != "*":
                # 简单分割，实际应该更复杂的解析
                column_parts = select_part.split(",")
                for part in column_parts:
                    part = part.strip()
                    # 提取列名（去掉别名）
                    if " as " in part.lower():
                        column = part.split(" as ")[0].strip()
                    else:
                        column = part.strip()
                    
                    if column and not column.startswith("("):  # 排除函数
                        columns.append(column)
        
        return columns
    
    def _generate_analysis_summary(
        self, 
        issues: List[Any], 
        suggestions: List[Any], 
        score: int
    ) -> str:
        """生成分析总结.
        
        Args:
            issues: 发现的问题
            suggestions: 优化建议
            score: 性能得分
            
        Returns:
            分析总结
        """
        if score >= 80:
            summary = f"查询性能良好（得分: {score}/100）。"
        elif score >= 60:
            summary = f"查询性能尚可（得分: {score}/100），有改进空间。"
        else:
            summary = f"查询性能较差（得分: {score}/100），急需优化。"
        
        if issues:
            critical_issues = [i for i in issues if getattr(i, 'severity', '') == "critical"]
            high_issues = [i for i in issues if getattr(i, 'severity', '') == "high"]
            
            if critical_issues:
                summary += f" 发现 {len(critical_issues)} 个严重问题。"
            elif high_issues:
                summary += f" 发现 {len(high_issues)} 个高优先级问题。"
            else:
                summary += f" 发现 {len(issues)} 个一般问题。"
        
        if suggestions:
            high_priority = [s for s in suggestions if getattr(s, 'priority', '') == "high"]
            if high_priority:
                summary += f" 提供了 {len(high_priority)} 个高优先级优化建议。"
            else:
                summary += f" 提供了 {len(suggestions)} 个优化建议。"
        
        return summary
    
    def _get_complexity_recommendations(self, complexity_level: str) -> List[str]:
        """获取复杂度相关的建议.
        
        Args:
            complexity_level: 复杂度等级
            
        Returns:
            建议列表
        """
        recommendations = {
            "Low": [
                "查询复杂度较低，性能应该良好",
                "继续保持简洁的查询结构"
            ],
            "Medium": [
                "查询复杂度适中，注意索引优化",
                "考虑是否可以简化部分逻辑"
            ],
            "High": [
                "查询复杂度较高，建议优化",
                "考虑拆分为多个简单查询",
                "重点关注索引和执行计划"
            ],
            "Very High": [
                "查询复杂度很高，强烈建议重构",
                "考虑使用视图或存储过程",
                "分析是否可以通过业务逻辑优化减少复杂度"
            ]
        }
        
        return recommendations.get(complexity_level, [])