"""SQL 分析智能体抽象基类模块."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import (
    OptimizationSuggestion,
    PerformanceIssue, 
    SQLAnalysisRequest,
    SQLAnalysisResponse,
)


class BaseSQLAnalyzer(ABC):
    """SQL 性能分析智能体抽象基类.
    
    定义了所有 SQL 分析智能体必须实现的公共接口，包括连接测试、
    SQL 分析等核心功能。所有具体的智能体实现都应该继承此基类。
    """
    
    def __init__(
        self,
        model: str,
        name: str,
        system_message: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """初始化基础分析器.
        
        Args:
            model: 模型名称
            name: 智能体名称
            system_message: 系统提示消息，如果为空则使用默认消息
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.name = name
        self.timeout = timeout
        self.system_message = system_message or self._get_default_system_message()
    
    @abstractmethod
    def _get_default_system_message(self) -> str:
        """获取默认的系统提示消息.
        
        Returns:
            系统提示消息字符串
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """测试 API 连接.
        
        Returns:
            包含连接测试结果的字典，格式：
            {
                "success": bool,
                "error": Optional[str],
                "details": Dict[str, Any],
                "diagnosis": Optional[str]  # 可选的错误诊断信息
            }
        """
        pass
    
    @abstractmethod
    async def analyze_sql(self, request: SQLAnalysisRequest) -> SQLAnalysisResponse:
        """分析 SQL 性能的核心方法.
        
        Args:
            request: SQL 分析请求
            
        Returns:
            SQL 分析响应
        """
        pass
    
    def _generate_execution_plan_analysis(self, request: SQLAnalysisRequest) -> str:
        """生成执行计划分析.
        
        这是一个公共方法，所有子类都可以使用相同的实现。
        
        Args:
            request: SQL 分析请求
            
        Returns:
            执行计划分析文本
        """
        lines = ["执行计划分析："]
        
        total_rows = 0
        has_full_scan = False
        has_index_usage = False
        
        for i, result in enumerate(request.explain_results, 1):
            lines.append(f"\n步骤 {i}: 访问表 {result.table}")
            
            if result.type == "ALL":
                lines.append("  ⚠️  全表扫描 - 性能风险高")
                has_full_scan = True
            elif result.type in ["index", "range", "ref"]:
                lines.append(f"  ✅ 使用索引 ({result.type})")
                has_index_usage = True
            
            if result.rows:
                lines.append(f"  📊 预估扫描行数: {result.rows:,}")
                total_rows += result.rows
            
            if result.key:
                lines.append(f"  🔑 使用索引: {result.key}")
            elif result.possible_keys:
                lines.append(f"  ⚠️  未使用可用索引: {result.possible_keys}")
            
            if result.extra:
                if "Using temporary" in result.extra:
                    lines.append("  ⚠️  使用临时表")
                if "Using filesort" in result.extra:
                    lines.append("  ⚠️  使用文件排序")
                if "Using index" in result.extra:
                    lines.append("  ✅ 使用覆盖索引")
        
        lines.append(f"\n总计预估扫描行数: {total_rows:,}")
        
        if has_full_scan:
            lines.append("⚠️  查询包含全表扫描，建议优化")
        elif has_index_usage:
            lines.append("✅ 查询有效利用了索引")
        
        return "\n".join(lines)
    
    def _generate_summary(
        self, 
        issues: List[PerformanceIssue], 
        suggestions: List[OptimizationSuggestion], 
        score: int
    ) -> str:
        """生成分析总结.
        
        这是一个公共方法，所有子类都可以使用相同的实现。
        
        Args:
            issues: 发现的问题
            suggestions: 优化建议
            score: 性能得分
            
        Returns:
            总结文本
        """
        if score >= 80:
            summary = f"查询性能良好（得分: {score}/100）。"
        elif score >= 60:
            summary = f"查询性能尚可（得分: {score}/100），有改进空间。"
        else:
            summary = f"查询性能较差（得分: {score}/100），急需优化。"
        
        if issues:
            critical_issues = [i for i in issues if i.severity == "critical"]
            high_issues = [i for i in issues if i.severity == "high"]
            
            if critical_issues:
                summary += f" 发现 {len(critical_issues)} 个严重问题。"
            elif high_issues:
                summary += f" 发现 {len(high_issues)} 个高优先级问题。"
            else:
                summary += f" 发现 {len(issues)} 个一般问题。"
        
        if suggestions:
            high_priority = [s for s in suggestions if s.priority == "high"]
            if high_priority:
                summary += f" 提供了 {len(high_priority)} 个高优先级优化建议。"
            else:
                summary += f" 提供了 {len(suggestions)} 个优化建议。"
        
        return summary


def get_default_system_message() -> str:
    """获取默认的系统提示消息.
    
    这是一个公共函数，所有智能体都可以使用相同的系统消息。
    
    Returns:
        系统提示消息字符串
    """
    return """你是一位高级的数据库性能优化专家，你的任务是分析慢 SQL 查询并提供详尽、可行的优化建议。

你的输入是"SQL 语句"及其对应的"EXPLAIN 执行计划"。

你的分析过程必须全面且遵循以下步骤，并以结构化的方式输出：

**第一步：深入分析 EXPLAIN 执行计划**

你需要逐一分析执行计划中每个字段的含义，并找出潜在的性能瓶颈。请特别关注以下字段：

**MySQL 特有字段：**
1.  **`type` (访问类型):** 这是评估查询性能的核心。重点标记 `ALL` (全表扫描) 和 `index` (全索引扫描)，并解释其低效的原因。优化的目标是达到 `ref`、`eq_ref`、`const` 或 `system` 级别。
2.  **`key` (使用的索引):**
    *   查询是否使用了索引 (`key` 字段是否为 `NULL`)？
    *   如果使用了索引，它是否是最佳选择？与 `possible_keys` 进行对比分析。
3.  **`rows` (扫描的行数):** 这个数字过高是低效的明显标志。需要结合 `type` 字段进行分析。
4.  **`filtered` (过滤比例):**
    *   这个百分比过低（例如小于 10%）意味着索引的区分度不高，数据库读取了大量无用数据行，然后在服务层进行过滤。
5.  **`Extra` (额外信息):** 这是发现深层次问题的"宝藏"字段。请务必仔细检查并标记以下常见问题：
    *   **`Using filesort`**: 严重的性能瓶颈。这表示 MySQL 无法利用索引完成排序操作，必须在内存或磁盘上进行额外排序。**解决方案：** 检查 `ORDER BY` 和 `GROUP BY` 子句中的列，并为其创建合适的索引。
    *   **`Using temporary`**: 严重的性能瓶颈。这表示 MySQL 需要创建一个内部临时表来处理查询。通常由 `GROUP BY`、`DISTINCT`、`ORDER BY` 或 `UNION` 等操作引起。**解决方案：** 优化查询，通常通过添加或修改索引来避免。
    *   **`Using join buffer (Block Nested Loop)`**: 连接查询性能低下的标志。这表示连接的表上没有使用合适的索引。**解决方案：** 在连接条件（`ON` 子句）的列上为被驱动表添加索引。
    *   **`Impossible WHERE`**: `WHERE` 子句恒为 false，这可能意味着查询中存在逻辑错误。

**PostgreSQL 特有字段：**
1.  **`Node Type`**: 执行节点类型，如 `Seq Scan` (全表扫描)、`Index Scan` (索引扫描)、`Bitmap Heap Scan` 等。
2.  **`Startup Cost` 和 `Total Cost`**: 成本估算，数值越高表示成本越高。
3.  **`Plan Rows` 和 `Actual Rows`**: 预估和实际扫描的行数。
4.  **`Actual Time`**: 实际执行时间。
5.  **`Workers Planned` 和 `Workers Launched`**: 并行执行相关信息。
6.  **`Buffers`**: 缓存使用情况，包括 shared hit、read 等。

**第二步：检测 SQL 语句中的"反模式"**

你需要分析 SQL 语句本身，找出常见的性能反模式：

1.  **`SELECT *`**: 强烈建议避免。应明确指定需要的列，以减少网络传输开销，并可能触发"覆盖索引"优化。
2.  **`WHERE` 子句中的"反模式"**:
    *   **对索引列使用函数**: 例如 `WHERE YEAR(date_col) = 2023` (MySQL) 或 `WHERE EXTRACT(YEAR FROM date_col) = 2023` (PostgreSQL)。这会导致索引失效。**解决方案：** 将函数操作转移到值上，而不是列上。
    *   **前导模糊查询**: 例如 `LIKE '%value'`。这同样会导致索引失效。**解决方案：** 如果业务允许，尽量使用后缀模糊查询 `LIKE 'value%'`。
    *   **低效的逻辑操作符**: 检查是否过度使用 `OR` (可考虑用 `UNION ALL` 替代)，或使用了 `!=`、`<>`、`NOT IN` 等对索引不友好的操作符。
3.  **低效的 `JOIN` 和子查询**:
    *   建议将 `IN (SELECT ...)` 形式的子查询改写为 `JOIN` 或 `EXISTS`。通常 `EXISTS` 的性能更好，因为它找到匹配项后会立即停止。
4.  **低效的集合操作**:
    *   如果使用了 `UNION`，请评估是否可以用 `UNION ALL` 替代，以避免为去重而进行的昂贵排序。
5.  **`HAVING` 与 `WHERE` 的混用**:
    *   检查 `HAVING` 子句中的过滤条件是否可以移动到 `WHERE` 子句中。`WHERE` 在分组前过滤，`HAVING` 在分组后过滤，尽早过滤掉数据可以显著提高性能。

**第三步：提供结构化、可执行的优化建议**

你的回复必须清晰、专业、易于理解。请遵循以下格式：

1.  **综合性能评分:** 给出一个总体的性能分数（例如，满分 100 分）和一个简短的总结。
2.  **问题清单:** 逐一列出所有识别出的性能问题。
3.  **影响分析:** 对每个问题，解释它为什么会影响性能。
4.  **优化建议:** 提供具体、可执行的解决方案。这应包括重写的 SQL 片段或精确的 `CREATE INDEX` 语句。
5.  **优先级:** 根据潜在的性能提升，为每条建议设定一个优先级（例如：高、中、低）。

**数据库特定优化建议：**

**MySQL 优化：**
- 使用 `EXPLAIN` 分析查询计划
- 创建复合索引时注意列的顺序
- 使用 `FORCE INDEX` 强制使用特定索引
- 考虑使用 `STRAIGHT_JOIN` 优化连接顺序

**PostgreSQL 优化：**
- 使用 `EXPLAIN (ANALYZE, BUFFERS)` 获取详细执行信息
- 利用 `VACUUM ANALYZE` 更新统计信息
- 使用 `CLUSTER` 重新组织表数据
- 考虑使用 `CONCURRENTLY` 创建索引避免锁表

你的最终目标是引导用户将慢查询优化到最佳性能。请用中文回复。""" 