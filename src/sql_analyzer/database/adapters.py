"""数据库适配器抽象层，支持多种数据库的特定逻辑."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import ExplainResult


class DatabaseAdapter(ABC):
    """数据库适配器抽象基类.
    
    为不同数据库提供统一的接口，处理数据库特定的逻辑。
    """
    
    @abstractmethod
    def get_database_name(self) -> str:
        """获取数据库名称."""
        pass
    
    @abstractmethod
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        """判断是否为全表扫描."""
        pass
    
    @abstractmethod
    def is_index_scan(self, result: ExplainResult) -> bool:
        """判断是否为索引扫描."""
        pass
    
    @abstractmethod
    def get_scan_rows(self, result: ExplainResult) -> int:
        """获取扫描行数."""
        pass
    
    @abstractmethod
    def get_connection_type(self, result: ExplainResult) -> str:
        """获取连接类型."""
        pass
    
    @abstractmethod
    def get_table_name(self, result: ExplainResult) -> str:
        """获取表名."""
        pass
    
    @abstractmethod
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        """获取索引信息."""
        pass
    
    @abstractmethod
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        """获取成本信息."""
        pass
    
    @abstractmethod
    def get_extra_info(self, result: ExplainResult) -> str:
        """获取额外信息."""
        pass
    
    @abstractmethod
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        """格式化EXPLAIN结果用于显示."""
        pass
    
    @abstractmethod
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        """获取数据库特定的优化建议."""
        pass


class MySQLAdapter(DatabaseAdapter):
    """MySQL数据库适配器."""
    
    def get_database_name(self) -> str:
        return "MySQL"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.type == "ALL"
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.type in ["index", "range", "ref", "eq_ref", "const"]
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "rows": result.rows,
            "filtered": result.filtered
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "select_type": result.select_type,
            "table": result.table,
            "partitions": result.partitions,
            "type": result.type,
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref,
            "rows": result.rows,
            "filtered": result.filtered,
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用",
                    "sql_example": "SHOW INDEX FROM table_name;",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
        return suggestions


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL数据库适配器."""
    
    def get_database_name(self) -> str:
        return "PostgreSQL"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.select_type and "Seq Scan" in result.select_type
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.select_type and "Index Scan" in result.select_type
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.plan_rows or result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.select_type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "index_name": getattr(result, 'index_name', None),
            "scan_direction": getattr(result, 'scan_direction', None)
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "startup_cost": getattr(result, 'startup_cost', None),
            "total_cost": getattr(result, 'total_cost', None),
            "plan_rows": result.plan_rows,
            "actual_rows": getattr(result, 'actual_rows', None),
            "actual_time": getattr(result, 'actual_time', None),
            "actual_loops": getattr(result, 'actual_loops', None)
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "node_type": result.select_type,
            "table": result.table,
            "plan_rows": result.plan_rows,
            "actual_rows": getattr(result, 'actual_rows', None),
            "startup_cost": getattr(result, 'startup_cost', None),
            "total_cost": getattr(result, 'total_cost', None),
            "actual_time": getattr(result, 'actual_time', None),
            "actual_loops": getattr(result, 'actual_loops', None),
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "顺序扫描" in issue or "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引，避免顺序扫描",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用",
                    "sql_example": "SELECT * FROM pg_indexes WHERE tablename = 'table_name';",
                    "analyze_command": "ANALYZE table_name;"
                })
        return suggestions


class TiDBAdapter(DatabaseAdapter):
    """TiDB数据库适配器（兼容MySQL语法）."""
    
    def get_database_name(self) -> str:
        return "TiDB"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.type == "TableFullScan" or result.type == "ALL"
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.type in ["IndexScan", "IndexLookUp", "index", "range", "ref", "eq_ref", "const"]
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "rows": result.rows,
            "filtered": result.filtered
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "select_type": result.select_type,
            "table": result.table,
            "type": result.type,
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref,
            "rows": result.rows,
            "filtered": result.filtered,
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引（TiDB支持多种索引类型）",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用，TiDB支持覆盖索引优化",
                    "sql_example": "SHOW INDEX FROM table_name;",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
        return suggestions


class MariaDBAdapter(DatabaseAdapter):
    """MariaDB数据库适配器（兼容MySQL语法）."""
    
    def get_database_name(self) -> str:
        return "MariaDB"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.type == "ALL"
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.type in ["index", "range", "ref", "eq_ref", "const"]
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "rows": result.rows,
            "filtered": result.filtered
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "select_type": result.select_type,
            "table": result.table,
            "type": result.type,
            "possible_keys": result.possible_keys,
            "key": result.key,
            "key_len": result.key_len,
            "ref": result.ref,
            "rows": result.rows,
            "filtered": result.filtered,
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引（MariaDB支持多种索引类型）",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用，MariaDB支持列存储引擎",
                    "sql_example": "SHOW INDEX FROM table_name;",
                    "analyze_command": "ANALYZE TABLE table_name;"
                })
        return suggestions


class OracleAdapter(DatabaseAdapter):
    """Oracle数据库适配器."""
    
    def get_database_name(self) -> str:
        return "Oracle"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.select_type and "TABLE ACCESS FULL" in result.select_type
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.select_type and ("INDEX" in result.select_type)
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.select_type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "access_predicates": getattr(result, 'access_predicates', None),
            "filter_predicates": getattr(result, 'filter_predicates', None)
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "cost": getattr(result, 'cost', None),
            "cardinality": result.rows,
            "bytes": getattr(result, 'bytes', None)
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "operation": result.select_type,
            "object_name": result.table,
            "cost": getattr(result, 'cost', None),
            "cardinality": result.rows,
            "bytes": getattr(result, 'bytes', None),
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引，考虑使用函数索引或分区",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE TABLE table_name COMPUTE STATISTICS;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用，考虑使用位图索引或复合索引",
                    "sql_example": "SELECT * FROM USER_INDEXES WHERE TABLE_NAME = 'TABLE_NAME';",
                    "analyze_command": "ANALYZE TABLE table_name COMPUTE STATISTICS;"
                })
        return suggestions


class SQLServerAdapter(DatabaseAdapter):
    """SQL Server数据库适配器."""
    
    def get_database_name(self) -> str:
        return "SQL Server"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.select_type and "Table Scan" in result.select_type
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.select_type and ("Index" in result.select_type)
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.select_type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "seek_predicates": getattr(result, 'seek_predicates', None),
            "predicate": getattr(result, 'predicate', None)
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "estimated_cpu_cost": getattr(result, 'estimated_cpu_cost', None),
            "estimated_io_cost": getattr(result, 'estimated_io_cost', None),
            "estimated_rows": result.rows
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "node_id": result.id,
            "physical_operation": result.select_type,
            "object_name": result.table,
            "estimated_rows": result.rows,
            "estimated_cpu_cost": getattr(result, 'estimated_cpu_cost', None),
            "estimated_io_cost": getattr(result, 'estimated_io_cost', None),
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引，考虑使用覆盖索引",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "UPDATE STATISTICS table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用，使用索引建议向导",
                    "sql_example": "SELECT * FROM sys.indexes WHERE object_id = OBJECT_ID('table_name');",
                    "analyze_command": "UPDATE STATISTICS table_name;"
                })
        return suggestions


class SQLiteAdapter(DatabaseAdapter):
    """SQLite数据库适配器."""
    
    def get_database_name(self) -> str:
        return "SQLite"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        return result.select_type and "SCAN TABLE" in result.select_type
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        return result.select_type and ("SEARCH TABLE" in result.select_type or "INDEX" in result.select_type)
    
    def get_scan_rows(self, result: ExplainResult) -> int:
        return result.rows or 0
    
    def get_connection_type(self, result: ExplainResult) -> str:
        return result.select_type or "unknown"
    
    def get_table_name(self, result: ExplainResult) -> str:
        return result.table or ""
    
    def get_index_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "detail": result.extra
        }
    
    def get_cost_info(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "rows": result.rows
        }
    
    def get_extra_info(self, result: ExplainResult) -> str:
        return result.extra or ""
    
    def format_explain_result(self, result: ExplainResult) -> Dict[str, Any]:
        return {
            "id": result.id,
            "detail": result.select_type,
            "table": result.table,
            "extra": result.extra
        }
    
    def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
        suggestions = []
        for issue in issues:
            if "全表扫描" in issue:
                suggestions.append({
                    "suggestion": "为查询条件创建合适的索引",
                    "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                    "analyze_command": "ANALYZE table_name;"
                })
            elif "缺失索引" in issue:
                suggestions.append({
                    "suggestion": "检查并优化索引使用",
                    "sql_example": "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='table_name';",
                    "analyze_command": "ANALYZE table_name;"
                })
        return suggestions


class DatabaseAdapterFactory:
    """数据库适配器工厂类."""
    
    _adapters = {
        "mysql": MySQLAdapter,
        "postgresql": PostgreSQLAdapter,
        "tidb": TiDBAdapter,
        "mariadb": MariaDBAdapter,
        "oracle": OracleAdapter,
        "sqlserver": SQLServerAdapter,
        "sqlite": SQLiteAdapter
    }
    
    @classmethod
    def create_adapter(cls, database_type: str) -> DatabaseAdapter:
        """创建数据库适配器.
        
        Args:
            database_type: 数据库类型
            
        Returns:
            数据库适配器实例
            
        Raises:
            ValueError: 如果数据库类型不支持
        """
        adapter_class = cls._adapters.get(database_type.lower())
        if not adapter_class:
            raise ValueError(f"不支持的数据库类型: {database_type}")
        
        return adapter_class()
    
    @classmethod
    def register_adapter(cls, database_type: str, adapter_class: type) -> None:
        """注册新的数据库适配器.
        
        Args:
            database_type: 数据库类型
            adapter_class: 适配器类
        """
        cls._adapters[database_type.lower()] = adapter_class
    
    @classmethod
    def get_supported_databases(cls) -> List[str]:
        """获取支持的数据库列表.
        
        Returns:
            支持的数据库类型列表
        """
        return list(cls._adapters.keys()) 