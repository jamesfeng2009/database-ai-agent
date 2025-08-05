"""HTML 报告生成模块."""

import os
from datetime import datetime
from typing import List, Optional

from .database.adapters import DatabaseAdapterFactory
from .models import ExplainResult, OptimizationSuggestion, PerformanceIssue, SQLAnalysisResponse


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


def _generate_explain_table_html(explain_results: List[ExplainResult]) -> str:
    """生成 EXPLAIN 结果表格的 HTML，使用数据库适配器"""
    if not explain_results:
        return '<div class="empty-state"><p>无 EXPLAIN 结果数据</p></div>'

    # 获取数据库适配器
    database_type = _detect_database_type(explain_results)
    adapter = DatabaseAdapterFactory.create_adapter(database_type)
    
    # 根据数据库类型生成表头
    if database_type == "postgresql":
        headers = [
            "ID", "Node Type", "Table", "Plan Rows", "Actual Rows",
            "Startup Cost", "Total Cost", "Actual Time", "Actual Loops", "Extra"
        ]
    else:
        # MySQL, TiDB 等
        headers = [
            "ID", "Select Type", "Table", "Partitions", "Type",
            "Possible Keys", "Key", "Key Len", "Ref", "Rows", "Filtered", "Extra"
        ]

    # 生成表头
    header_html = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"

    # 生成数据行
    rows_html = ""
    for result in explain_results:
        # 使用适配器格式化结果
        formatted_result = adapter.format_explain_result(result)
        
        if database_type == "postgresql":
            # PostgreSQL数据行
            row_data = [
                formatted_result.get("id", ""),
                formatted_result.get("node_type", ""),
                formatted_result.get("table", ""),
                formatted_result.get("plan_rows", ""),
                formatted_result.get("actual_rows", ""),
                f"{formatted_result.get('startup_cost', 0):.2f}" if formatted_result.get('startup_cost') is not None else "",
                f"{formatted_result.get('total_cost', 0):.2f}" if formatted_result.get('total_cost') is not None else "",
                f"{formatted_result.get('actual_time', 0):.2f}" if formatted_result.get('actual_time') is not None else "",
                formatted_result.get("actual_loops", ""),
                formatted_result.get("extra", "")
            ]
        else:
            # MySQL/TiDB数据行
            row_data = [
                formatted_result.get("id", ""),
                formatted_result.get("select_type", ""),
                formatted_result.get("table", ""),
                formatted_result.get("partitions", ""),
                formatted_result.get("type", ""),
                formatted_result.get("possible_keys", ""),
                formatted_result.get("key", ""),
                formatted_result.get("key_len", ""),
                formatted_result.get("ref", ""),
                formatted_result.get("rows", ""),
                f"{formatted_result.get('filtered', 0)}%" if formatted_result.get('filtered') is not None else "",
                formatted_result.get("extra", "")
            ]
        
        rows_html += "<tr>" + "".join([f"<td>{d}</td>" for d in row_data]) + "</tr>"

    return f"""
    <div class="table-container">
        <table class="explain-table">
            <thead>{header_html}</thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """

def generate_html_report(
    response: SQLAnalysisResponse,
    sql_statement: str,
    output_path: Optional[str] = None,
    title: str = "SQL 性能分析报告"
) -> str:
    """生成 HTML 格式的分析报告.
    
    Args:
        response: SQL 分析响应对象
        sql_statement: 原始 SQL 语句
        output_path: 输出文件路径，如果为 None 则返回 HTML 字符串
        title: 报告标题
        
    Returns:
        HTML 字符串，如果指定了 output_path 则返回文件路径
    """
    html_content = _generate_html_content(response, sql_statement, title)
    
    if output_path:
        # 获取输出路径的目录和文件名
        output_dir = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        
        # 确保report文件夹存在
        if output_dir:
            report_dir = os.path.join(output_dir, "report")
        else:
            report_dir = "report"
        
        os.makedirs(report_dir, exist_ok=True)
        
        # 将文件放在report文件夹下
        final_path = os.path.join(report_dir, filename)
        
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return final_path
    
    return html_content


def _generate_html_content(
    response: SQLAnalysisResponse,
    sql_statement: str,
    title: str
) -> str:
    """生成 HTML 内容.
    
    Args:
        response: SQL 分析响应对象
        sql_statement: 原始 SQL 语句
        title: 报告标题
        
    Returns:
        完整的 HTML 字符串
    """
    # 生成当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # EXPLAIN 结果表格
    explain_table_html = _generate_explain_table_html(response.explain_results)

    # 生成性能评分的颜色和状态
    score_color, score_status = _get_score_display(response.performance_score)
    
    # 生成问题列表 HTML
    issues_html = _generate_issues_html(response.issues)
    
    # 生成建议列表 HTML
    suggestions_html = _generate_suggestions_html(response.suggestions)
    
    # 格式化 SQL 语句
    formatted_sql = _format_sql_for_html(sql_statement)
    
    # JavaScript代码（需要双大括号转义）
    js_code = """
        // 主选项卡切换
        document.querySelectorAll('.tab-btn').forEach(button => {
            button.addEventListener('click', () => {
                // 移除所有选项卡的激活状态
                document.querySelectorAll('.tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // 激活当前选项卡
                button.classList.add('active');
                document.getElementById(button.dataset.tab).classList.add('active');
            });
        });
        
        // 子选项卡切换
        document.querySelectorAll('.sub-tab-btn').forEach(button => {
            button.addEventListener('click', () => {
                // 移除所有子选项卡的激活状态
                document.querySelectorAll('.sub-tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelectorAll('.sub-tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // 激活当前子选项卡
                button.classList.add('active');
                document.getElementById(button.dataset.tab).classList.add('active');
            });
        });
    """
    
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1 class="title">{title}</h1>
            <div class="meta-info">
                <span class="timestamp">生成时间: {current_time}</span>
            </div>
        </header>

        <div class="summary-section">
            <div class="score-card">
                <div class="score-circle {score_color}">
                    <span class="score-number">{response.performance_score}</span>
                    <span class="score-text">/ 100</span>
                </div>
                <div class="score-info">
                    <h3 class="score-title">性能评分</h3>
                    <p class="score-status {score_color}">{score_status}</p>
                </div>
            </div>
            
            <div class="summary-text">
                <h3>分析总结</h3>
                <p>{response.summary}</p>
            </div>
        </div>

        <div class="sql-section">
            <h2 class="section-title">
                <span class="icon">📝</span>
                SQL 语句
            </h2>
            <div class="sql-container">
                <pre><code class="sql">{formatted_sql}</code></pre>
            </div>
        </div>
        
        <!-- 主选项卡 -->
        <div class="tabs">
            <div class="tab-nav">
                <button class="tab-btn active" data-tab="basic-analysis">基础分析</button>
                <button class="tab-btn" data-tab="ai-analysis">AI高级分析</button>
            </div>
            
            <!-- 基础分析选项卡内容 -->
            <div id="basic-analysis" class="tab-content active">
                <!-- 基础分析子选项卡 -->
                <div class="sub-tabs">
                    <div class="sub-tab-nav">
                        <button class="sub-tab-btn active" data-tab="explain-plan">EXPLAIN执行计划</button>
                        <button class="sub-tab-btn" data-tab="plan-analysis">执行计划分析</button>
                        <button class="sub-tab-btn" data-tab="issues">性能问题</button>
                        <button class="sub-tab-btn" data-tab="suggestions">基础优化建议</button>
                    </div>
                    
                    <!-- EXPLAIN执行计划 -->
                    <div id="explain-plan" class="sub-tab-content active">
                        <h3 class="section-title">
                            <span class="icon">📊</span>
                            EXPLAIN 执行计划
                        </h3>
                        {explain_table_html}
                    </div>
                    
                    <!-- 执行计划分析 -->
                    <div id="plan-analysis" class="sub-tab-content">
                        <h3 class="section-title">
                            <span class="icon">📊</span>
                            执行计划分析
                        </h3>
                        <div class="analysis-content">
                            <pre>{response.execution_plan_analysis}</pre>
                        </div>
                    </div>
                    
                    <!-- 性能问题 -->
                    <div id="issues" class="sub-tab-content">
                        <h3 class="section-title">
                            <span class="icon">🚨</span>
                            性能问题 
                            <span class="badge">{len(response.issues)}</span>
                        </h3>
                        {issues_html}
                    </div>
                    
                    <!-- 基础优化建议 -->
                    <div id="suggestions" class="sub-tab-content">
                        <h3 class="section-title">
                            <span class="icon">💡</span>
                            基础优化建议
                            <span class="badge">{len(response.suggestions)}</span>
                        </h3>
                        {suggestions_html}
                    </div>
                </div>
            </div>
            
            <!-- AI高级分析选项卡内容 -->
            <div id="ai-analysis" class="tab-content">
                <h3 class="section-title">
                    <span class="icon">🤖</span>
                    AI高级分析
                </h3>
                <div class="detailed-content">
                    <div class="markdown-content">
                        {_format_markdown_to_html(response.detailed_analysis)}
                    </div>
                </div>
            </div>
        </div>

        <footer class="footer">
            <p>报告由 SQL 分析器自动生成</p>
        </footer>
    </div>
    
    <script>
        {js_code}
    </script>
</body>
</html>"""
    
    return html_template


def _get_css_styles() -> str:
    """获取 CSS 样式.
    
    Returns:
        CSS 样式字符串
    """
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8fafc;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .meta-info {
            font-size: 1rem;
            opacity: 0.9;
        }

        .summary-section {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 2rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }

        .score-card {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
            font-weight: bold;
        }

        .score-circle.excellent {
            background: linear-gradient(45deg, #10b981, #34d399);
            color: white;
        }

        .score-circle.good {
            background: linear-gradient(45deg, #3b82f6, #60a5fa);
            color: white;
        }

        .score-circle.fair {
            background: linear-gradient(45deg, #f59e0b, #fbbf24);
            color: white;
        }

        .score-circle.poor {
            background: linear-gradient(45deg, #ef4444, #f87171);
            color: white;
        }

        .score-number {
            font-size: 2.5rem;
            font-weight: 900;
        }

        .score-text {
            font-size: 1rem;
            margin-top: -0.5rem;
        }

        .score-info h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .score-status {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .score-status.excellent { color: #10b981; }
        .score-status.good { color: #3b82f6; }
        .score-status.fair { color: #f59e0b; }
        .score-status.poor { color: #ef4444; }

        .summary-text {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .summary-text h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            color: #1f2937;
            font-weight: 700;
        }

        .icon {
            font-size: 1.5rem;
        }

        .badge {
            background: #667eea;
            color: white;
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .sql-section, .issues-section, .suggestions-section, .analysis-section, .detailed-section {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }

        .sql-container {
            background: #1f2937;
            border-radius: 8px;
            overflow-x: auto;
        }

        .sql {
            color: #e5e7eb;
            padding: 1.5rem;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .issue-item, .suggestion-item {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .issue-item:hover, .suggestion-item:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }

        .issue-header, .suggestion-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }

        .issue-title, .suggestion-title {
            font-size: 1.2rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .severity-badge, .priority-badge {
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .severity-critical, .priority-high {
            background: #fee2e2;
            color: #dc2626;
        }

        .severity-high {
            background: #fed7aa;
            color: #ea580c;
        }

        .severity-medium, .priority-medium {
            background: #fef3c7;
            color: #d97706;
        }

        .severity-low, .priority-low {
            background: #dcfce7;
            color: #16a34a;
        }

        .issue-description, .suggestion-description {
            margin-bottom: 1rem;
            color: #4b5563;
            line-height: 1.6;
        }

        .issue-impact {
            background: #f3f4f6;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            border-left: 3px solid #f59e0b;
        }

        .affected-tables {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .table-tag {
            background: #e0e7ff;
            color: #3730a3;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .suggestion-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .detail-item {
            background: #f9fafb;
            padding: 0.8rem;
            border-radius: 6px;
        }

        .detail-label {
            font-weight: 600;
            color: #374151;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }

        .detail-value {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .sql-example {
            background: #1f2937;
            color: #e5e7eb;
            padding: 1rem;
            border-radius: 6px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
        }

        .analysis-content, .detailed-content {
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .analysis-content pre {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            line-height: 1.6;
            color: #374151;
            white-space: pre-wrap;
        }

        .markdown-content {
            line-height: 1.8;
            color: #374151;
        }

        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            color: #1f2937;
            margin: 1.5rem 0 1rem 0;
        }

        .markdown-content ul, .markdown-content ol {
            margin-left: 2rem;
            margin-bottom: 1rem;
        }

        .markdown-content li {
            margin-bottom: 0.5rem;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.9rem;
            border-top: 1px solid #e5e7eb;
            margin-top: 2rem;
        }

        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #6b7280;
        }

        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        .explain-section {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .table-container {
            overflow-x: auto;
        }
        
        .explain-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        .explain-table th, 
        .explain-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .explain-table th {
            background-color: #f8fafc;
            font-weight: 600;
            color: #4a5568;
        }
        
        .explain-table tr:hover {
            background-color: #f7fafc;
        }

        /* 选项卡样式 */
        .tabs {
            margin-bottom: 2rem;
        }
        
        .tab-nav {
            display: flex;
            background: white;
            border-radius: 12px 12px 0 0;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .tab-btn {
            flex: 1;
            padding: 1rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            color: #6b7280;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
            outline: none;
        }
        
        .tab-btn:hover {
            background-color: #f9fafb;
        }
        
        .tab-btn.active {
            color: #667eea;
            border-bottom-color: #667eea;
            background-color: #f0f4ff;
        }
        
        .tab-content {
            display: none;
            background: white;
            border-radius: 0 0 12px 12px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .sub-tabs {
            margin-bottom: 2rem;
        }
        
        .sub-tab-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .sub-tab-btn {
            padding: 0.5rem 1rem;
            background: #f3f4f6;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            color: #4b5563;
            transition: all 0.3s ease;
        }
        
        .sub-tab-btn:hover {
            background-color: #e5e7eb;
        }
        
        .sub-tab-btn.active {
            background-color: #667eea;
            color: white;
        }
        
        .sub-tab-content {
            display: none;
        }
        
        .sub-tab-content.active {
            display: block;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .summary-section {
                grid-template-columns: 1fr;
                text-align: center;
            }
            
            .suggestion-details {
                grid-template-columns: 1fr;
            }
            
            .title {
                font-size: 2rem;
            }
            
            .tab-nav, .sub-tab-nav {
                flex-direction: column;
            }
            
            .tab-btn, .sub-tab-btn {
                width: 100%;
                text-align: center;
            }
        }
    """


def _get_score_display(score: int) -> tuple[str, str]:
    """根据评分获取显示颜色和状态.
    
    Args:
        score: 性能评分
        
    Returns:
        (颜色类名, 状态文本) 的元组
    """
    if score >= 90:
        return "excellent", "优秀"
    elif score >= 70:
        return "good", "良好"
    elif score >= 50:
        return "fair", "一般"
    else:
        return "poor", "较差"


def _generate_issues_html(issues: list[PerformanceIssue]) -> str:
    """生成问题列表的 HTML.
    
    Args:
        issues: 性能问题列表
        
    Returns:
        问题列表的 HTML 字符串
    """
    if not issues:
        return """
        <div class="empty-state">
            <div class="icon">✅</div>
            <p>太棒了！没有发现明显的性能问题。</p>
        </div>
        """
    
    html_parts = []
    for issue in issues:
        severity_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(issue.severity, "⚪")
        
        affected_tables_html = ""
        if issue.affected_tables:
            table_tags = [f'<span class="table-tag">{table}</span>' for table in issue.affected_tables]
            affected_tables_html = f"""
            <div style="margin-top: 1rem;">
                <strong>受影响的表：</strong>
                <div class="affected-tables" style="margin-top: 0.5rem;">
                    {' '.join(table_tags)}
                </div>
            </div>
            """
        
        issue_html = f"""
        <div class="issue-item">
            <div class="issue-header">
                <div class="issue-title">
                    <span>{severity_icon}</span>
                    <span>{issue.issue_type}</span>
                </div>
                <span class="severity-badge severity-{issue.severity}">{issue.severity}</span>
            </div>
            <div class="issue-description">
                {issue.description}
            </div>
            <div class="issue-impact">
                <strong>性能影响：</strong> {issue.impact}
            </div>
            {affected_tables_html}
        </div>
        """
        html_parts.append(issue_html)
    
    return ''.join(html_parts)


def _generate_suggestions_html(suggestions: list[OptimizationSuggestion]) -> str:
    """生成建议列表的 HTML.
    
    Args:
        suggestions: 优化建议列表
        
    Returns:
        建议列表的 HTML 字符串
    """
    if not suggestions:
        return """
        <div class="empty-state">
            <div class="icon">💭</div>
            <p>暂无具体的优化建议。</p>
        </div>
        """
    
    html_parts = []
    for suggestion in suggestions:
        priority_icon = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }.get(suggestion.priority, "⚪")
        
        sql_example_html = ""
        if suggestion.sql_example:
            sql_example_html = f"""
            <div style="margin-top: 1rem;">
                <strong>SQL 示例：</strong>
                <pre class="sql-example">{suggestion.sql_example}</pre>
            </div>
            """
        
        suggestion_html = f"""
        <div class="suggestion-item">
            <div class="suggestion-header">
                <div class="suggestion-title">
                    <span>{priority_icon}</span>
                    <span>{suggestion.suggestion}</span>
                </div>
                <span class="priority-badge priority-{suggestion.priority}">{suggestion.priority}</span>
            </div>
            <div class="suggestion-details">
                <div class="detail-item">
                    <div class="detail-label">预期改善</div>
                    <div class="detail-value">{suggestion.expected_improvement}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">实施难度</div>
                    <div class="detail-value">{suggestion.implementation_difficulty}</div>
                </div>
            </div>
            <div class="detail-item" style="margin-top: 1rem;">
                <div class="detail-label">类别</div>
                <div class="detail-value">{suggestion.category}</div>
            </div>
            {sql_example_html}
        </div>
        """
        html_parts.append(suggestion_html)
    
    return ''.join(html_parts)


def _format_sql_for_html(sql: str) -> str:
    """格式化 SQL 语句用于 HTML 显示.
    
    Args:
        sql: 原始 SQL 语句
        
    Returns:
        格式化后的 SQL 字符串
    """
    # 移除多余的空白字符
    lines = [line.strip() for line in sql.strip().split('\n') if line.strip()]
    return '\n'.join(lines)


def _format_markdown_to_html(text: str) -> str:
    """简单的 Markdown 转 HTML 格式化.
    
    Args:
        text: Markdown 文本
        
    Returns:
        HTML 格式的文本
    """
    # 简单的 Markdown 处理
    lines = text.split('\n')
    html_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            html_lines.append('<br>')
            continue
            
        # 处理标题
        if line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        # 处理列表
        elif line.startswith('- '):
            # 处理列表项中的加粗文本
            content = line[2:]
            if '**' in content:
                # 替换所有的 **text** 为 <strong>text</strong>
                import re
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            html_lines.append(f'<li>{content}</li>')
        # 处理其他内容的加粗
        else:
            if '**' in line:
                import re
                line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{line}</p>')
    
    return '\n'.join(html_lines) 