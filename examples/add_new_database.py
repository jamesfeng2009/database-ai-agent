#!/usr/bin/env python3
"""添加新数据库支持的示例.

这个示例展示了如何为 SQL 分析器添加新的数据库支持。
以 Oracle 数据库为例。
"""

from src.sql_analyzer.database.adapters import DatabaseAdapter, DatabaseAdapterFactory
from src.sql_analyzer.models import ExplainResult


class OracleAdapter(DatabaseAdapter):
    """Oracle 数据库适配器示例."""
    
    def get_database_name(self) -> str:
        return "Oracle"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        # Oracle 的全表扫描检测逻辑
        return result.type == "TABLE ACCESS FULL" or "FULL" in str(result.select_type)
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        # Oracle 的索引扫描检测逻辑
        return result.type in ["INDEX RANGE SCAN", "INDEX UNIQUE SCAN", "INDEX FAST FULL SCAN"]
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        # Oracle 的行数获取逻辑
        return result.rows or result.cardinality or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        # Oracle 的连接类型获取逻辑
        return result.type or result.operation or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        # Oracle 的表名获取逻辑
        return result.table or result.object_name or ""
    
    def get_index_info(self, result: ExplainResult) -> dict:
        # Oracle 的索引信息获取逻辑
        return {
            "index_name": getattr(result, 'index_name', None),
            "index_type": getattr(result, 'index_type', None),
            "access_predicates": getattr(result, 'access_predicates', None)
        }
    
    def get_cost_info(self, result: ExplainResult) -> dict:
        # Oracle 的成本信息获取逻辑
        return {
            "cost": getattr(result, 'cost', None),
            "cardinality": getattr(result, 'cardinality', None),
            "bytes": getattr(result, 'bytes', None),
            "cpu_cost": getattr(result, 'cpu_cost', None),
            "io_cost": getattr(result, 'io_cost', None)
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        # Oracle 的额外信息获取逻辑
        return result.extra or result.predicate_info or ""
    
    def format_explain_result(self, result: ExplainResult) -> dict:
        # Oracle 的结果格式化逻辑
        return {
            "id": result.id,
            "operation": result.type,
            "object_name": result.table,
            "cardinality": result.rows,
            "cost": getattr(result, 'cost', None),
            "bytes": getattr(result, 'bytes', None),
            "cpu_cost": getattr(result, 'cpu_cost', None),
            "io_cost": getattr(result, 'io_cost', None),
            "predicate_info": getattr(result, 'predicate_info', None)
        }
    
    def get_optimization_suggestions(self, issues: list) -> list:
        # Oracle 特定的优化建议
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引，避免全表扫描",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE TABLE table_name COMPUTE STATISTICS;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用，Oracle支持多种索引类型",
                    "sql_example": "SELECT * FROM user_indexes WHERE table_name = 'TABLE_NAME';",
                    "analyze_command": "ANALYZE TABLE table_name COMPUTE STATISTICS;"
                })
        return suggestions


def register_oracle_adapter():
    """注册 Oracle 适配器."""
    DatabaseAdapterFactory.register_adapter("oracle", OracleAdapter)
    print("✅ Oracle 适配器已注册")


def test_oracle_adapter():
    """测试 Oracle 适配器."""
    # 注册适配器
    register_oracle_adapter()
    
    # 创建模拟的 Oracle EXPLAIN 结果
    oracle_result = ExplainResult(
        id=1,
        type="TABLE ACCESS FULL",
        table="EMPLOYEES",
        rows=1000,
        extra="Using where"
    )
    
    # 使用适配器
    adapter = DatabaseAdapterFactory.create_adapter("oracle")
    
    print(f"数据库名称: {adapter.get_database_name()}")
    print(f"是否为全表扫描: {adapter.is_full_table_scan(oracle_result)}")
    print(f"扫描行数: {adapter.get_scan_rows(oracle_result)}")
    print(f"连接类型: {adapter.get_connection_type(oracle_result)}")
    print(f"表名: {adapter.get_table_name(oracle_result)}")
    
    # 测试优化建议
    suggestions = adapter.get_optimization_suggestions(["全表扫描"])
    print(f"优化建议数量: {len(suggestions)}")
    
    # 查看支持的数据库
    supported = DatabaseAdapterFactory.get_supported_databases()
    print(f"支持的数据库: {supported}")


if __name__ == "__main__":
    test_oracle_adapter() 