"""慢查询分析器主模块."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .analyzer_base import BaseSQLAnalyzer
from .database import (
    BaseDatabaseConnector,
    BaseSlowQueryReader,
    DatabaseConfig,
    SlowQueryConfig,
)
from .models import (
    BatchAnalysisResult,
    SlowQueryAnalysisResult,
    SlowQueryEntry,
    SQLAnalysisRequest,
    SQLAnalysisResponse,
)
from .tools import (
    calculate_performance_score,
    detect_performance_issues,
    generate_optimization_suggestions,
)

logger = logging.getLogger(__name__)


class SlowQueryAnalyzer:
    """慢查询分析器.
    
    集成数据库连接、慢查询读取和 AI 分析功能的完整分析器。
    支持 MySQL 和 PostgreSQL。
    """
    
    def __init__(
        self,
        database_config: DatabaseConfig,
        slow_query_config: Optional[SlowQueryConfig] = None,
        ai_analyzer: Optional[BaseSQLAnalyzer] = None
    ) -> None:
        """初始化慢查询分析器.
        
        Args:
            database_config: 数据库连接配置
            slow_query_config: 慢查询配置，如果为 None 则使用默认配置
            ai_analyzer: AI 分析器，如果为 None 则只进行基础分析
        """
        self.database_config = database_config
        self.slow_query_config = slow_query_config or SlowQueryConfig()
        self.ai_analyzer = ai_analyzer
        self.connector: Optional[BaseDatabaseConnector] = None
        self.slow_query_reader: Optional[BaseSlowQueryReader] = None
    
    async def initialize(self) -> None:
        """初始化分析器."""
        try:
            # 创建并连接数据库
            from .database import create_and_connect_database_connector, create_slow_query_reader
            
            self.connector = await create_and_connect_database_connector(self.database_config)
            
            # 测试连接
            if not await self.connector.test_connection():
                raise RuntimeError(f"{self.database_config.get_database_type().value} 连接测试失败")
            
            # 创建慢查询读取器
            self.slow_query_reader = create_slow_query_reader(self.connector, self.slow_query_config)
            
            logger.info(f"慢查询分析器初始化完成 ({self.database_config.get_database_type().value})")
            
        except Exception as e:
            logger.error(f"慢查询分析器初始化失败: {e}")
            raise
    
    async def cleanup(self) -> None:
        """清理资源."""
        if self.connector:
            await self.connector.disconnect()
            logger.info("慢查询分析器资源已清理")
    
    async def analyze_slow_queries(self, limit: Optional[int] = None) -> BatchAnalysisResult:
        """分析慢查询日志.
        
        Args:
            limit: 限制分析的查询数量，如果为 None 则使用配置中的限制
            
        Returns:
            批量分析结果
            
        Raises:
            RuntimeError: 当分析器未初始化时
        """
        if not self.connector or not self.slow_query_reader:
            raise RuntimeError("分析器未初始化，请先调用 initialize() 方法")
        
        start_time = datetime.now()
        
        try:
            # 获取慢查询
            logger.info("正在获取慢查询日志...")
            slow_queries = await self.slow_query_reader.get_slow_queries()
            
            if limit:
                slow_queries = slow_queries[:limit]
            
            total_queries = len(slow_queries)
            logger.info(f"获取到 {total_queries} 条慢查询，开始分析...")
            print(f"获取到 {total_queries} 条慢查询，开始分析...")
            
            # 分析每个慢查询
            results: List[SlowQueryAnalysisResult] = []
            analyzed_count = 0
            failed_count = 0
            
            for i, slow_query in enumerate(slow_queries, 1):
                try:
                    logger.info(f"正在分析第 {i}/{total_queries} 条查询...")
                    print(f"正在分析第 {i}/{total_queries} 条查询...")

                    # 执行 EXPLAIN
                    explain_results = await self.connector.execute_explain(slow_query.sql_statement)
                    
                    # 创建分析请求
                    request = SQLAnalysisRequest(
                        sql_statement=slow_query.sql_statement,
                        explain_results=explain_results,
                        database_schema=slow_query.database,
                        additional_context=f"慢查询信息: 执行时间={slow_query.query_time}秒, 扫描行数={slow_query.rows_examined}"
                    )
                    
                    # 执行分析
                    if self.ai_analyzer:
                        # 使用 AI 分析器
                        analysis_response = await self.ai_analyzer.analyze_sql(request)
                    else:
                        # 使用基础分析
                        analysis_response = await self._basic_analysis(request)
                    
                    # 创建结果
                    result = SlowQueryAnalysisResult(
                        slow_query=slow_query,
                        analysis_response=analysis_response,
                        explain_results=explain_results
                    )
                    results.append(result)
                    analyzed_count += 1
                    
                except Exception as e:
                    logger.error(f"分析第 {i} 条查询失败: {e}")
                    failed_count += 1
                    continue
            
            end_time = datetime.now()
            
            # 生成统计摘要
            summary_stats = self._generate_summary_stats(results)
            
            # 创建批量分析结果
            batch_result = BatchAnalysisResult(
                total_queries=total_queries,
                analyzed_queries=analyzed_count,
                failed_queries=failed_count,
                results=results,
                summary_stats=summary_stats,
                start_time=start_time,
                end_time=end_time
            )
            
            logger.info(f"慢查询分析完成: 总计={total_queries}, 成功={analyzed_count}, 失败={failed_count}")
            return batch_result
            
        except Exception as e:
            logger.error(f"慢查询分析失败: {e}")
            raise
    
    async def analyze_single_query(self, sql: str) -> SlowQueryAnalysisResult:
        """分析单个 SQL 查询.
        
        Args:
            sql: SQL 查询语句
            
        Returns:
            单个查询分析结果
            
        Raises:
            RuntimeError: 当分析器未初始化时
        """
        if not self.connector:
            raise RuntimeError("分析器未初始化，请先调用 initialize() 方法")
        
        try:
            # 执行 EXPLAIN
            explain_results = await self.connector.execute_explain(sql)
            
            # 创建虚拟的慢查询条目
            slow_query = SlowQueryEntry(
                query_time=0.0,
                lock_time=0.0,
                rows_sent=0,
                rows_examined=0,
                sql_statement=sql,
                timestamp=datetime.now()
            )
            
            # 创建分析请求
            request = SQLAnalysisRequest(
                sql_statement=sql,
                explain_results=explain_results,
                database_schema=self.database_config.database
            )
            
            # 执行分析
            if self.ai_analyzer:
                analysis_response = await self.ai_analyzer.analyze_sql(request)
            else:
                analysis_response = await self._basic_analysis(request)
            
            # 创建结果
            result = SlowQueryAnalysisResult(
                slow_query=slow_query,
                analysis_response=analysis_response,
                explain_results=explain_results
            )
            
            return result
            
        except Exception as e:
            logger.error(f"分析 SQL 查询失败: {e}")
            raise
    
    async def _basic_analysis(self, request: SQLAnalysisRequest) -> SQLAnalysisResponse:
        """执行基础分析（不使用 AI）.
        
        Args:
            request: 分析请求
            
        Returns:
            分析响应
        """
        # 检测性能问题
        issues = detect_performance_issues(request)
        
        # 生成优化建议
        suggestions = generate_optimization_suggestions(request, issues)
        
        # 计算性能评分
        score = calculate_performance_score(request, issues)
        
        # 生成分析总结
        summary = self._generate_basic_summary(score, len(issues), len(suggestions))
        
        # 生成执行计划分析
        plan_analysis = self._generate_plan_analysis(request.explain_results)
        
        return SQLAnalysisResponse(
            summary=summary,
            performance_score=score,
            issues=issues,
            suggestions=suggestions,
            detailed_analysis=f"基础分析完成，发现 {len(issues)} 个问题，提供 {len(suggestions)} 个建议。",
            execution_plan_analysis=plan_analysis,
            explain_results=request.explain_results
        )
    
    def _generate_basic_summary(self, score: int, issue_count: int, suggestion_count: int) -> str:
        """生成基础分析总结.
        
        Args:
            score: 性能评分
            issue_count: 问题数量
            suggestion_count: 建议数量
            
        Returns:
            分析总结
        """
        if score >= 80:
            performance_desc = "性能良好"
        elif score >= 60:
            performance_desc = "性能一般"
        elif score >= 40:
            performance_desc = "性能较差"
        else:
            performance_desc = "性能很差"
        
        return f"查询{performance_desc}（评分：{score}/100），发现 {issue_count} 个性能问题，提供 {suggestion_count} 个优化建议。"
    
    def _generate_plan_analysis(self, explain_results: List) -> str:
        """生成执行计划分析.
        
        Args:
            explain_results: EXPLAIN 结果列表
            
        Returns:
            执行计划分析文本
        """
        if not explain_results:
            return "无执行计划信息"
        
        # 获取数据库适配器
        from .database.adapters import DatabaseAdapterFactory
        from .tools import _detect_database_type
        
        database_type = _detect_database_type(explain_results)
        adapter = DatabaseAdapterFactory.create_adapter(database_type)
        
        analysis_parts = []
        for i, result in enumerate(explain_results, 1):
            # 使用适配器获取数据
            table_name = adapter.get_table_name(result)
            connection_type = adapter.get_connection_type(result)
            rows = adapter.get_scan_rows(result)
            index_info = adapter.get_index_info(result)
            cost_info = adapter.get_cost_info(result)
            extra_info = adapter.get_extra_info(result)
            
            part = f"步骤 {i}: 表 {table_name}"
            
            # 添加连接类型
            if connection_type:
                part += f", 连接类型: {connection_type}"
            
            # 添加行数信息
            if rows and rows > 0:
                part += f", 预估扫描行数: {rows:,}"
            
            # 添加实际行数（PostgreSQL特有）
            actual_rows = cost_info.get("actual_rows")
            if actual_rows:
                part += f", 实际扫描行数: {actual_rows:,}"
            
            # 添加索引信息
            key = index_info.get("key")
            possible_keys = index_info.get("possible_keys")
            if key:
                part += f", 使用索引: {key}"
            elif possible_keys:
                part += f", 可用索引: {possible_keys}"
            
            # 添加成本信息（PostgreSQL特有）
            startup_cost = cost_info.get("startup_cost")
            total_cost = cost_info.get("total_cost")
            if startup_cost:
                part += f", 启动成本: {startup_cost}"
            if total_cost:
                part += f", 总成本: {total_cost}"
            
            # 添加额外信息
            if extra_info:
                part += f", 额外信息: {extra_info}"
            
            analysis_parts.append(part)
        
        return "\n".join(analysis_parts)
    
    def _generate_summary_stats(self, results: List[SlowQueryAnalysisResult]) -> Dict[str, Any]:
        """生成统计摘要.
        
        Args:
            results: 分析结果列表
            
        Returns:
            统计摘要字典
        """
        if not results:
            return {}
        
        # 安全地计算平均性能评分
        scores = []
        for r in results:
            if hasattr(r.analysis_response, 'performance_score') and r.analysis_response.performance_score is not None:
                try:
                    score = int(r.analysis_response.performance_score)
                    if 0 <= score <= 100:  # 验证评分范围
                        scores.append(score)
                except (ValueError, TypeError):
                    logger.warning(f"无效的性能评分: {r.analysis_response.performance_score}")
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 统计问题严重程度
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        total_issues = 0
        
        for result in results:
            if hasattr(result.analysis_response, 'issues') and result.analysis_response.issues:
                for issue in result.analysis_response.issues:
                    if hasattr(issue, 'severity') and issue.severity in severity_counts:
                        severity_counts[issue.severity] += 1
                        total_issues += 1
        
        # 安全地统计查询时间
        valid_query_times = []
        for r in results:
            if hasattr(r.slow_query, 'query_time') and r.slow_query.query_time is not None:
                try:
                    query_time = float(r.slow_query.query_time)
                    if query_time >= 0:  # 验证时间为非负数
                        valid_query_times.append(query_time)
                except (ValueError, TypeError):
                    logger.warning(f"无效的查询时间: {r.slow_query.query_time}")
        
        avg_query_time = sum(valid_query_times) / len(valid_query_times) if valid_query_times else 0
        max_query_time = max(valid_query_times) if valid_query_times else 0
        
        # 安全地统计扫描行数
        valid_rows_examined = []
        for r in results:
            if hasattr(r.slow_query, 'rows_examined') and r.slow_query.rows_examined is not None:
                try:
                    rows = int(r.slow_query.rows_examined)
                    if rows >= 0:  # 验证行数为非负数
                        valid_rows_examined.append(rows)
                except (ValueError, TypeError):
                    logger.warning(f"无效的扫描行数: {r.slow_query.rows_examined}")
        
        avg_rows = sum(valid_rows_examined) / len(valid_rows_examined) if valid_rows_examined else 0
        max_rows = max(valid_rows_examined) if valid_rows_examined else 0
        
        # 安全地统计低分查询
        low_score_count = 0
        for score in scores:
            if score < 60:
                low_score_count += 1
        
        # 安全地统计全表扫描查询
        full_scan_count = 0
        for r in results:
            if hasattr(r, 'explain_results') and r.explain_results:
                for explain in r.explain_results:
                    # 兼容MySQL和PostgreSQL的全表扫描检测
                    if (hasattr(explain, 'type') and explain.type == "ALL") or \
                       (hasattr(explain, 'select_type') and "Seq Scan" in str(explain.select_type)):
                        full_scan_count += 1
                        break  # 每个查询只统计一次
        
        return {
            "average_performance_score": round(avg_score, 2),
            "total_issues": total_issues,
            "severity_distribution": severity_counts,
            "average_query_time": round(avg_query_time, 3),
            "max_query_time": round(max_query_time, 3),
            "average_rows_examined": round(avg_rows),
            "max_rows_examined": max_rows,
            "queries_with_score_below_60": low_score_count,
            "queries_with_full_table_scan": full_scan_count
        }


async def create_slow_query_analyzer(
    database_config: DatabaseConfig,
    slow_query_config: Optional[SlowQueryConfig] = None,
    ai_analyzer: Optional[BaseSQLAnalyzer] = None
) -> SlowQueryAnalyzer:
    """创建并初始化慢查询分析器.
    
    Args:
        database_config: 数据库连接配置
        slow_query_config: 慢查询配置
        ai_analyzer: AI 分析器
        
    Returns:
        已初始化的慢查询分析器
    """
    analyzer = SlowQueryAnalyzer(database_config, slow_query_config, ai_analyzer)
    await analyzer.initialize()
    return analyzer 