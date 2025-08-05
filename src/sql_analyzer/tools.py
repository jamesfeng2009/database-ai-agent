"""SQL 分析工具函数."""

from typing import Any, Dict, List

from .database.adapters import DatabaseAdapterFactory
from .models import ExplainResult, OptimizationSuggestion, PerformanceIssue, SQLAnalysisRequest


def _detect_database_type(explain_results: List[ExplainResult]) -> str:
    """检测数据库类型.
    
    Args:
        explain_results: EXPLAIN 结果列表
        
    Returns:
        数据库类型字符串
    """
    if not explain_results:
        return "unknown"
    
    # 检查PostgreSQL特有字段
    if any(result.plan_rows is not None for result in explain_results):
        return "postgresql"
    
    # 检查MySQL特有字段
    if any(result.type is not None for result in explain_results):
        return "mysql"
    
    # 默认返回MySQL（向后兼容）
    return "mysql"


def _get_database_adapter(explain_results: List[ExplainResult]):
    """获取数据库适配器.
    
    Args:
        explain_results: EXPLAIN 结果列表
        
    Returns:
        数据库适配器实例
    """
    database_type = _detect_database_type(explain_results)
    return DatabaseAdapterFactory.create_adapter(database_type)


def analyze_explain_results(explain_results: List[ExplainResult]) -> Dict[str, Any]:
    """分析 EXPLAIN 结果，提取关键性能指标，使用数据库适配器.
    
    Args:
        explain_results: EXPLAIN 查询结果列表
        
    Returns:
        包含性能指标的字典
    """
    if not explain_results:
        return {
            "total_rows": 0,
            "full_table_scans": [],
            "missing_indexes": [],
            "expensive_operations": [],
            "join_types": [],
            "using_temporary": False,
            "using_filesort": False,
            "database_type": "unknown"
        }
    
    adapter = _get_database_adapter(explain_results)
    database_type = _detect_database_type(explain_results)
    
    analysis = {
        "total_rows": 0,
        "full_table_scans": [],
        "missing_indexes": [],
        "expensive_operations": [],
        "join_types": [],
        "using_temporary": False,
        "using_filesort": False,
        "database_type": database_type
    }
    
    for result in explain_results:
        # 使用适配器获取数据
        rows = adapter.get_scan_rows(result)
        table_name = adapter.get_table_name(result)
        connection_type = adapter.get_connection_type(result)
        index_info = adapter.get_index_info(result)
        extra_info = adapter.get_extra_info(result)
        
        # 计算总扫描行数
        analysis["total_rows"] += rows
        
        # 检查全表扫描
        if adapter.is_full_table_scan(result):
            analysis["full_table_scans"].append({
                "table": table_name,
                "rows": rows,
                "scan_type": connection_type
            })
        
        # 检查缺失索引
        possible_keys = index_info.get("possible_keys")
        key = index_info.get("key")
        if possible_keys and not key:
            analysis["missing_indexes"].append({
                "table": table_name,
                "possible_keys": possible_keys
            })
        
        # 检查昂贵操作
        if extra_info:
            if "Using temporary" in extra_info:
                analysis["using_temporary"] = True
            if "Using filesort" in extra_info:
                analysis["using_filesort"] = True
            if "Using where" in extra_info and adapter.is_full_table_scan(result):
                analysis["expensive_operations"].append("全表扫描 + WHERE 过滤")
        
        # 数据库特定的昂贵操作检测
        if database_type == "postgresql":
            if "Sort" in connection_type and not adapter.is_index_scan(result):
                analysis["expensive_operations"].append("排序操作")
            if "Hash" in connection_type:
                analysis["expensive_operations"].append("哈希操作")
            if "Bitmap" in connection_type:
                analysis["expensive_operations"].append("位图操作")
        
        # 记录连接类型
        if connection_type:
            analysis["join_types"].append(connection_type)
    
    return analysis


def detect_performance_issues(request: SQLAnalysisRequest) -> List[PerformanceIssue]:
    """检测 SQL 性能问题，使用数据库适配器.
    
    Args:
        request: SQL 分析请求
        
    Returns:
        发现的性能问题列表
    """
    if not request.explain_results:
        return []
    
    adapter = _get_database_adapter(request.explain_results)
    analysis = analyze_explain_results(request.explain_results)
    database_type = analysis.get("database_type", "unknown")
    
    issues = []
    
    # 检查全表扫描
    if analysis["full_table_scans"]:
        for scan in analysis["full_table_scans"]:
            rows = scan.get("rows", 0)
            table = scan.get("table", "")
            scan_type = scan.get("scan_type", "")
            
            # 根据行数确定严重程度
            severity = "critical" if rows > 10000 else "high"
            
            if database_type == "postgresql":
                description = f"表 {table} 正在进行顺序扫描 ({scan_type})，预估扫描 {rows:,} 行"
            else:
                description = f"表 {table} 正在进行全表扫描，预估扫描 {rows:,} 行"
            
            issues.append(PerformanceIssue(
                severity=severity,
                issue_type="全表扫描",
                description=description,
                impact="会导致大量 I/O 操作，严重影响查询性能",
                affected_tables=[table] if table else []
            ))
    
    # 检查缺失索引
    if analysis["missing_indexes"]:
        for missing in analysis["missing_indexes"]:
            table = missing.get("table", "")
            possible_keys = missing.get("possible_keys", "")
            
            issues.append(PerformanceIssue(
                severity="high",
                issue_type="缺失索引",
                description=f"表 {table} 有可用索引但未使用: {possible_keys}",
                impact="无法利用索引加速查询，导致性能下降",
                affected_tables=[table] if table else []
            ))
    
    # 检查临时表
    if analysis["using_temporary"]:
        issues.append(PerformanceIssue(
            severity="medium",
            issue_type="使用临时表",
            description="查询需要创建临时表来存储中间结果",
            impact="增加内存使用和磁盘 I/O，影响查询性能",
            affected_tables=[]
        ))
    
    # 检查文件排序
    if analysis["using_filesort"]:
        issues.append(PerformanceIssue(
            severity="medium",
            issue_type="文件排序",
            description="查询需要进行文件排序操作",
            impact="无法使用索引排序，需要额外的排序开销",
            affected_tables=[]
        ))
    
    # 数据库特定的问题检测
    if database_type == "postgresql":
        for operation in analysis["expensive_operations"]:
            if "排序操作" in operation:
                issues.append(PerformanceIssue(
                    severity="medium",
                    issue_type="排序操作",
                    description="查询需要进行排序操作",
                    impact="排序操作会增加CPU和内存使用",
                    affected_tables=[]
                ))
            elif "哈希操作" in operation:
                issues.append(PerformanceIssue(
                    severity="low",
                    issue_type="哈希操作",
                    description="查询使用哈希操作",
                    impact="哈希操作会消耗额外内存",
                    affected_tables=[]
                ))
    
    # 检查大量行扫描
    total_rows = analysis.get("total_rows", 0)
    if total_rows > 100000:
        issues.append(PerformanceIssue(
            severity="high",
            issue_type="大量行扫描",
            description=f"查询预估需要扫描 {total_rows:,} 行数据",
            impact="大量行扫描会导致严重的性能问题",
            affected_tables=[]
        ))
    
    return issues


def generate_optimization_suggestions(
    request: SQLAnalysisRequest, 
    issues: List[PerformanceIssue]
) -> List[OptimizationSuggestion]:
    """生成优化建议，使用数据库适配器.
    
    Args:
        request: SQL 分析请求
        issues: 检测到的性能问题
        
    Returns:
        优化建议列表
    """
    if not request.explain_results:
        return []
    
    adapter = _get_database_adapter(request.explain_results)
    analysis = analyze_explain_results(request.explain_results)
    database_type = analysis.get("database_type", "unknown")
    
    suggestions = []
    
    # 获取数据库特定的优化建议
    issue_types = [issue.issue_type for issue in issues]
    db_suggestions = adapter.get_optimization_suggestions(issue_types)
    
    # 转换为 OptimizationSuggestion 对象
    for db_suggestion in db_suggestions:
        suggestions.append(OptimizationSuggestion(
            priority="high",
            category="index",
            suggestion=db_suggestion["suggestion"],
            expected_improvement="可大幅减少扫描行数，提升查询性能 50-90%",
            implementation_difficulty="easy",
            sql_example=db_suggestion["sql_example"]
        ))
    
    # 针对临时表的建议
    if analysis["using_temporary"]:
        suggestions.append(OptimizationSuggestion(
            priority="medium",
            category="query_rewrite",
            suggestion="优化 GROUP BY 或 ORDER BY 子句，避免使用临时表",
            expected_improvement="减少内存使用和 I/O 操作，提升性能 20-40%",
            implementation_difficulty="medium",
            sql_example="使用覆盖索引或重写查询逻辑"
        ))
    
    # 针对文件排序的建议
    if analysis["using_filesort"]:
        suggestions.append(OptimizationSuggestion(
            priority="medium",
            category="index",
            suggestion="为 ORDER BY 字段创建索引，避免文件排序",
            expected_improvement="利用索引排序可提升性能 30-60%",
            implementation_difficulty="easy",
            sql_example="CREATE INDEX idx_sort ON table_name (sort_column);"
        ))
    
    # 通用优化建议
    if analysis["total_rows"] > 1000:
        suggestions.append(OptimizationSuggestion(
            priority="low",
            category="query_rewrite",
            suggestion="考虑添加更具选择性的 WHERE 条件来减少扫描行数",
            expected_improvement="减少数据扫描量，提升查询效率",
            implementation_difficulty="medium",
            sql_example="在 WHERE 子句中添加更多过滤条件"
        ))
    
    return suggestions


def calculate_performance_score(request: SQLAnalysisRequest, issues: List[PerformanceIssue]) -> int:
    """计算性能评分，兼容所有数据库.
    
    Args:
        request: SQL 分析请求
        issues: 检测到的性能问题
        
    Returns:
        性能评分 (0-100)
    """
    base_score = 100
    analysis = analyze_explain_results(request.explain_results)
    
    # 根据问题严重程度扣分
    for issue in issues:
        if issue.severity == "critical":
            base_score -= 30
        elif issue.severity == "high":
            base_score -= 20
        elif issue.severity == "medium":
            base_score -= 10
        elif issue.severity == "low":
            base_score -= 5
    
    # 根据扫描行数扣分
    if analysis["total_rows"] > 100000:
        base_score -= 20
    elif analysis["total_rows"] > 10000:
        base_score -= 10
    elif analysis["total_rows"] > 1000:
        base_score -= 1
    
    return max(0, base_score)


def format_analysis_request(request: SQLAnalysisRequest) -> str:
    """格式化分析请求为文本格式，使用数据库适配器.
    
    Args:
        request: SQL 分析请求
        
    Returns:
        格式化的文本
    """
    if not request.explain_results:
        return f"=== SQL 语句 ===\n{request.sql_statement}\n\n=== 无 EXPLAIN 结果 ==="
    
    adapter = _get_database_adapter(request.explain_results)
    
    lines = [
        "=== SQL 语句 ===",
        request.sql_statement,
        "",
        "=== EXPLAIN 结果 ===",
    ]
    
    for i, result in enumerate(request.explain_results, 1):
        lines.append(f"第 {i} 行 EXPLAIN 结果:")
        
        # 使用适配器格式化结果
        formatted_result = adapter.format_explain_result(result)
        
        for key, value in formatted_result.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
        
        lines.append("")
    
    if request.additional_context:
        lines.extend([
            "=== 额外上下文 ===",
            request.additional_context,
            ""
        ])
    
    return "\n".join(lines) 