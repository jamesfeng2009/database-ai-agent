# 数据库抽象层架构文档

## 概述

SQL 分析器采用了多层数据库抽象架构，支持多种数据库类型，包括 MySQL、PostgreSQL、TiDB 等，并提供了易于扩展的适配器模式。

## 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (app.py)                          │
├─────────────────────────────────────────────────────────────┤
│                   AI 分析层 (agent.py)                      │
├─────────────────────────────────────────────────────────────┤
│                分析工具层 (tools.py)                        │
├─────────────────────────────────────────────────────────────┤
│                数据库适配器层 (adapters.py)                 │
├─────────────────────────────────────────────────────────────┤
│                数据库抽象层 (connector_base.py)             │
├─────────────────────────────────────────────────────────────┤
│                具体实现层 (mysql.py, postgresql.py)         │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 数据库适配器层 (`adapters.py`)

**作用**: 处理不同数据库的特定逻辑，提供统一的接口

**核心类**:
- `DatabaseAdapter`: 抽象基类，定义统一接口
- `MySQLAdapter`: MySQL 数据库适配器
- `PostgreSQLAdapter`: PostgreSQL 数据库适配器  
- `TiDBAdapter`: TiDB 数据库适配器
- `DatabaseAdapterFactory`: 适配器工厂类

**主要功能**:
- 数据库类型检测
- 全表扫描识别
- 索引扫描识别
- 执行计划格式化
- 数据库特定的优化建议

### 2. 数据库抽象层 (`connector_base.py`)

**作用**: 提供数据库连接和慢查询读取的抽象接口

**核心类**:
- `BaseDatabaseConnector`: 数据库连接器抽象基类
- `BaseSlowQueryReader`: 慢查询读取器抽象基类

### 3. 具体实现层

**MySQL 实现** (`mysql.py`):
- `MySQLConnector`: MySQL 连接器实现
- `MySQLSlowQueryReader`: MySQL 慢查询读取器实现

**PostgreSQL 实现** (`postgresql.py`):
- `PostgreSQLConnector`: PostgreSQL 连接器实现
- `PostgreSQLSlowQueryReader`: PostgreSQL 慢查询读取器实现

## 适配器模式的优势

### 1. **统一接口**
所有数据库使用相同的分析接口，上层代码无需关心具体数据库类型：

```python
# 自动检测数据库类型并选择合适的适配器
adapter = DatabaseAdapterFactory.create_adapter(database_type)

# 统一的分析接口
is_full_scan = adapter.is_full_table_scan(result)
scan_rows = adapter.get_scan_rows(result)
optimization_suggestions = adapter.get_optimization_suggestions(issues)
```

### 2. **易于扩展**
添加新数据库支持只需实现适配器接口：

```python
class NewDatabaseAdapter(DatabaseAdapter):
    def get_database_name(self) -> str:
        return "NewDatabase"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        # 实现新数据库的全表扫描检测逻辑
        pass
    
    # ... 实现其他抽象方法

# 注册新适配器
DatabaseAdapterFactory.register_adapter("newdb", NewDatabaseAdapter)
```

### 3. **数据库特定优化**
每个适配器可以提供针对特定数据库的优化建议：

```python
def get_optimization_suggestions(self, issues: List[str]) -> List[Dict[str, str]]:
    suggestions = []
    for issue in issues:
        if "全表扫描" in issue:
            suggestions.append({
                "suggestion": "数据库特定的优化建议",
                "sql_example": "CREATE INDEX idx_column ON table_name (column_name);",
                "analyze_command": "数据库特定的分析命令"
            })
    return suggestions
```

## 支持的数据库

### 1. **MySQL**
- **连接器**: `MySQLConnector`
- **适配器**: `MySQLAdapter`
- **特性**: 支持 performance_schema 和慢查询日志
- **优化建议**: MySQL 特定的索引和查询优化

### 2. **PostgreSQL**
- **连接器**: `PostgreSQLConnector`
- **适配器**: `PostgreSQLAdapter`
- **特性**: 支持 pg_stat_statements 和 pg_stat_activity
- **优化建议**: PostgreSQL 特定的顺序扫描和索引优化

### 3. **TiDB**
- **连接器**: 兼容 MySQL 连接器
- **适配器**: `TiDBAdapter`
- **特性**: 兼容 MySQL 语法，支持分布式查询
- **优化建议**: TiDB 特定的分布式优化建议

## 添加新数据库支持

### 步骤 1: 创建适配器

```python
from src.sql_analyzer.database.adapters import DatabaseAdapter
from src.sql_analyzer.models import ExplainResult

class NewDatabaseAdapter(DatabaseAdapter):
    def get_database_name(self) -> str:
        return "NewDatabase"
    
    def is_full_table_scan(self, result: ExplainResult) -> bool:
        # 实现全表扫描检测逻辑
        return result.type == "FULL_SCAN"
    
    def is_index_scan(self, result: ExplainResult) -> bool:
        # 实现索引扫描检测逻辑
        return result.type in ["INDEX_SCAN", "INDEX_LOOKUP"]
    
    # ... 实现其他抽象方法
```

### 步骤 2: 创建连接器（可选）

如果新数据库的连接方式与现有数据库不同，需要创建新的连接器：

```python
from src.sql_analyzer.database.connector_base import BaseDatabaseConnector

class NewDatabaseConnector(BaseDatabaseConnector):
    async def execute_explain(self, sql: str) -> List[ExplainResult]:
        # 实现新数据库的 EXPLAIN 执行逻辑
        pass
    
    async def test_connection(self) -> bool:
        # 实现连接测试逻辑
        pass
```

### 步骤 3: 注册适配器

```python
from src.sql_analyzer.database.adapters import DatabaseAdapterFactory

# 注册新适配器
DatabaseAdapterFactory.register_adapter("newdb", NewDatabaseAdapter)
```

### 步骤 4: 更新配置

在 `app.py` 中添加新数据库的配置支持：

```python
def detect_database_type() -> DatabaseType:
    # 添加新数据库的检测逻辑
    if all(os.getenv(var) for var in ["NEWDB_HOST", "NEWDB_USER", "NEWDB_PASSWORD"]):
        return DatabaseType.NEWDB
    # ... 其他检测逻辑
```

## 使用示例

### 基本使用

```python
from src.sql_analyzer.database import DatabaseAdapterFactory
from src.sql_analyzer.models import ExplainResult

# 创建适配器（自动检测数据库类型）
explain_results = [...]  # 从数据库获取的 EXPLAIN 结果
adapter = DatabaseAdapterFactory.create_adapter_from_results(explain_results)

# 分析执行计划
for result in explain_results:
    if adapter.is_full_table_scan(result):
        print(f"发现全表扫描: {adapter.get_table_name(result)}")
    
    scan_rows = adapter.get_scan_rows(result)
    print(f"扫描行数: {scan_rows}")
```

### 获取优化建议

```python
# 检测性能问题
issues = ["全表扫描", "缺失索引"]

# 获取数据库特定的优化建议
suggestions = adapter.get_optimization_suggestions(issues)

for suggestion in suggestions:
    print(f"建议: {suggestion['suggestion']}")
    print(f"SQL示例: {suggestion['sql_example']}")
    print(f"分析命令: {suggestion['analyze_command']}")
```

### 扩展支持

```python
# 查看当前支持的数据库
supported_databases = DatabaseAdapterFactory.get_supported_databases()
print(f"支持的数据库: {supported_databases}")

# 注册新的数据库适配器
DatabaseAdapterFactory.register_adapter("oracle", OracleAdapter)
```

## 最佳实践

### 1. **适配器设计原则**
- 保持接口一致性
- 提供数据库特定的优化建议
- 正确处理空值和异常情况
- 添加详细的文档说明

### 2. **性能考虑**
- 适配器应该是轻量级的
- 避免在适配器中执行复杂的计算
- 缓存常用的检测结果

### 3. **错误处理**
- 提供有意义的错误信息
- 支持降级到默认行为
- 记录详细的调试信息

### 4. **测试**
- 为每个适配器编写单元测试
- 测试边界情况和异常情况
- 验证优化建议的准确性

## 总结

数据库抽象层架构提供了：

✅ **统一接口**: 所有数据库使用相同的分析接口
✅ **易于扩展**: 添加新数据库支持简单快速
✅ **数据库特定优化**: 针对不同数据库提供专门的优化建议
✅ **向后兼容**: 现有功能不受影响
✅ **可维护性**: 清晰的层次结构，易于维护和调试

这种架构使得 SQL 分析器能够轻松支持更多数据库类型，同时保持代码的整洁和可维护性。 