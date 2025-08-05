# 基于 AI 的智能数据库慢查询分析工具

这是一个 AI 驱动的智能 SQL 性能分析工具，支持 MySQL、PostgreSQL、TiDB 等多种数据库，专门用于自动化抓取并分析慢查询，然后提供优化建议。

## 功能特点

- 🗄️ **多数据库支持**: 支持 MySQL、PostgreSQL、TiDB 等多种数据库
- 🔀 **三种分析模式**: 支持基础分析、本地AI分析和云端AI分析 三种分析模式
- 📋 **执行计划解读**: 用自然语言解读数据库 EXPLAIN 执行计划
- 🚨 **自动问题检测**: 从执行计划中自动识别全表扫描、缺失索引、临时表等多种常见性能问题
- 📊 **性能评分**: 自动计算每条查询的性能评分（0-100 分）
- 💡 **优化建议**: 提供具体的 SQL 优化建议和示例代码
- 🔍 **AI 深度分析**: 基于大语言模型提供专业的 SQL 性能分析和优化建议
- 📦 **批量处理**: 支持批量分析多个慢查询并生成统计结果
- 📄 **HTML 报告**: 生成美观的可视化 HTML 分析报告
- 🔧 **数据库抽象层**: 易于扩展支持新的数据库类型

## 🛡️ 安全特性

工具会对抓取到的慢 SQL 自动执行 EXPLAIN 执行计划分析，为确保数据操作安全性，因此特别添加如下安全增强手段：

### 1. 输入验证增强

- **空值检查**: 防止空或无效 SQL 语句
- **长度限制**: 防止过大 SQL 语句攻击
- **格式验证**: 使用 sqlparse 进行 SQL 语法验证

### 2. SQL 危险操作检测

- **关键字黑名单**: 检测 DROP, DELETE, UPDATE, INSERT 等危险操作
- **多语句检测**: 防止通过分号注入多条 SQL 语句
- **操作限制**: EXPLAIN 功能只允许 SELECT 语句

### 3. 数据隐私保护

- **本地处理**: Ollama 模式下，所有数据处理在本地完成，不上传到云端

## 🚀 三种分析模式

本工具提供了三种不同的分析模式，使用者可根据环境和需求灵活选择：

### 🔧 基础分析模式 (无需 AI)

当未使用任何 AI 配置时，工具使用规则引擎进行快速分析：

- ⚡ **执行计划分析**: 全面解读数据库 EXPLAIN 执行计划
- 🚨 **性能问题检测**: 自动识别全表扫描、索引缺失、临时表等常见性能问题
- 📊 **性能评分**: 基于规则算法计算查询性能评分（0-100 分）
- 💡 **基础优化建议**: 基于检测问题提供结构化的优化建议
- 📋 **详细报告**: 生成包含执行计划和问题分析的基础报告
- ⚡ **极速响应**: 基于规则引擎快速返回分析结果，极速响应无延迟

### 🤖 云端 AI 增强分析模式

当配置了云端模型API密钥时，使用云端模型进行深度分析：

- 🧠 **智能深度分析**: 基于大语言模型的高级 SQL 性能分析
- 🔍 **上下文理解**: AI 理解查询意图和业务逻辑，提供更精准的分析
- 💬 **自然语言解释**: 用易懂的中文解释复杂的性能问题
- 💡 **高级优化建议**: 基于 AI 模型，提供更专业、更具体的优化策略
- 🎯 **SQL重构示例**: 提供优化后的 SQL 重构示例代码
- 📈 **影响评估**: 预测优化后的性能改进效果

### 🦙 Ollama 本地分析模式 

当配置了 Ollama 本地模型时，使用本地大模型进行分析：

- 🔒 **数据隐私**: 所有数据在本地处理，无需上传到云端
- ⚡ **快速响应**: 本地推理，无网络延迟
- 💰 **零成本**: 无 API 调用费用，完全免费使用
- 📡 **离线运行**: 支持完全离线环境下的 SQL 性能分析

### 分析模式对比

| 特性         | 基础分析模式           | 云端 AI 增强分析模式         | Ollama 本地分析模式        |
| ------------ | ---------------------- | ----------------------- | ------------------------ |
| **环境要求** | 无需使用AI          | 需要 API 密钥    | 需要 Ollama 本地模型     |
| **分析速度** | 非常快                   | 有推理延迟                | 取决于本地硬件条件           |
| **分析深度** | 规则驱动，覆盖常见问题 | 先进模型深度理解上下文 | 取决于本地模型的推理能力     |
| **建议质量** | 结构化，仅基于执行计划结果 | 个性化，考虑具体场景和行业最佳实践    | 个性化，考虑具体场景和行业最佳实践 |
| **成本**     | 免费                   | 需要 API 调用费用       | 免费                     |
| **适用场景** | 快速诊断，批量分析     | 深度优化，复杂查询分析  | 日常 SQL 分析            |

## 📦 安装说明

### 系统要求

- Python 3.9 或更高版本
- 支持的操作系统：Windows、macOS、Linux

### 快速安装

#### 方法一：使用安装脚本（推荐）

```bash
# 克隆项目
git clone https://github.com/your-username/sql-analyzer.git
cd sql-analyzer

# 运行安装脚本
python install_dependencies.py
```

#### 方法二：使用 pip 安装

```bash
# 克隆项目
git clone https://github.com/your-username/sql-analyzer.git
cd sql-analyzer

# 创建虚拟环境（推荐）
python -m venv sql_analyzer_env
source sql_analyzer_env/bin/activate  # Linux/Mac
# 或
sql_analyzer_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 方法三：使用 pyproject.toml

```bash
# 从项目根目录安装
pip install -e .
```

### 验证安装

```bash
# 运行安装验证
python install_dependencies.py

# 或手动验证关键模块
python -c "import pydantic, dotenv, aiomysql, asyncpg, httpx, openai; print('✅ 安装成功')"
```

## 🗄️ 支持的数据库

### MySQL
- **连接方式**: 异步连接 (aiomysql)
- **慢查询来源**: performance_schema.events_statements_history_long
- **执行计划**: EXPLAIN FORMAT=JSON
- **优化建议**: MySQL 特定的索引和查询优化

### PostgreSQL
- **连接方式**: 异步连接 (asyncpg)
- **慢查询来源**: pg_stat_statements
- **执行计划**: EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
- **优化建议**: PostgreSQL 特定的顺序扫描和索引优化

### TiDB
- **连接方式**: 兼容 MySQL 连接器
- **慢查询来源**: 兼容 MySQL performance_schema
- **执行计划**: 兼容 MySQL EXPLAIN
- **优化建议**: TiDB 特定的分布式优化建议

## 项目结构

```
sqlAnalyzer/
├── app.py                    # 主程序入口
├── install_dependencies.py   # 依赖安装脚本
├── requirements.txt          # 依赖包列表
├── pyproject.toml           # 项目配置文件
├── config/
│   └── env.example          # 环境变量配置示例
├── src/
│   └── sql_analyzer/
│       ├── __init__.py      # 包初始化
│       ├── agent.py         # AI 智能体实现
│       ├── analyzer_base.py # SQL 分析智能体抽象基类
│       ├── config.py        # 配置管理
│       ├── models.py        # 数据模型定义
│       ├── report.py        # HTML 报告生成
│       ├── slow_query_analyzer.py  # 慢查询分析器
│       ├── tools.py         # 分析工具函数
│       └── database/        # 数据库抽象层
│           ├── __init__.py
│           ├── connector_base.py  # 数据库连接器抽象基类
│           ├── adapters.py   # 数据库适配器
│           ├── factory.py   # 数据库工厂函数
│           ├── models.py    # 数据库配置模型
│           ├── mysql.py     # MySQL 实现
│           └── postgresql.py # PostgreSQL 实现
├── examples/                # 使用示例
│   └── add_new_database.py # 添加新数据库支持示例
└── README.md
```

### 核心架构

```
src/sql_analyzer/
├── analyzer_base.py         # 抽象基类和接口定义
├── config.py               # 配置管理模块
├── agent.py                # 智能体实现
├── models.py               # 数据模型定义
├── tools.py                # 分析工具函数
├── report.py               # HTML 报告生成
├── slow_query_analyzer.py  # 慢查询分析器主模块
└── database/               # 数据库抽象层
    ├── connector_base.py   # 数据库连接器抽象基类
    ├── adapters.py         # 数据库适配器
    ├── factory.py          # 数据库工厂函数
    ├── mysql.py            # MySQL 实现
    └── postgresql.py       # PostgreSQL 实现

app.py   # 主程序入口
report/  # HTML报告目录
tests/   # 单元测试目录
```

## 🔄 工具运作流程

下图展示了工具的完整运作原理和核心流程：

```mermaid
flowchart TD
    A[开始] --> B{检测数据库类型}
    B --> C[MySQL]
    B --> D[PostgreSQL]
    B --> E[TiDB]
    
    C --> F[创建数据库连接器]
    D --> F
    E --> F
    
    F --> G{测试AI模型连接}
    G --> |连接失败| G1[回退到基础分析模式]
    G --> |连接成功| H[使用AI增强分析模式]
    G1 --> I{数据源选择}
    H --> I{数据源选择}
    
    I -->|MySQL| J[Performance Schema<br/>events_statements_history_long]
    I -->|PostgreSQL| K[pg_stat_statements]
    I -->|TiDB| L[兼容MySQL数据源]

    J --> M[获取慢查询数据]
    K --> M
    L --> M

    M --> N{是否有慢查询}
    N -->|否| N1[提示无慢查询数据]
    N1 --> Z

    N -->|是| O[SQL 安全验证]
    O --> P{安全检查通过}
    P -->|否| P1[跳过危险 SQL]
    P -->|是| Q[执行 EXPLAIN 分析]
    P1 --> Q

    Q --> R[解析执行计划]
    R --> S[数据库适配器分析]
    S --> T[基础问题检测]

    T --> U[全表扫描检测]
    T --> V[缺失索引检测]
    T --> W[临时表使用检测]
    T --> X[文件排序检测]
    T --> Y[大量行扫描检测]

    U --> Z1[基础优化建议]
    V --> Z1
    W --> Z1
    X --> Z1
    Y --> Z1

    Z1 --> Z2[性能评分计算]

    Z2 -->|基础分析模式| Z3[生成基础总结]
    Z2 -->|AI 增强分析模式| Z4[AI 智能体分析]

    Z4 --> Z5[深入分析 EXPLAIN 执行计划]
    Z4 --> Z6[检测 SQL 语句中的反模式]
    Z4 --> Z7[提供个性化的优化建议]
    Z3 --> Z8[生成分析报告]

    Z5 --> Z8
    Z6 --> Z8
    Z7 --> Z8

    Z8 --> Z9[输出统计结果]
    Z9 --> Z10[生成 HTML 报告]

    Z10 --> Z[结束]

    style A fill:#e1f5fe
    style G1 fill:#f3e5f5
    style H fill:#f3e5f5
    style M fill:#e8f5e8
    style O fill:#fff8e1
    style T fill:#fce4ec
    style Z1 fill:#e3f2fd
    style Z3 fill:#f1f8e9
    style Z4 fill:#e8f5e8
```


#### 基础分析模式的规则引擎原理

规则引擎会从MySQL EXPLAIN 执行计划中基于规则自动匹配常见性能缺陷并评估问题严重等级，可识别的常见问题如下：

1. **全表扫描 (Critical/High)**

   - 检测 `type=ALL` 的查询
   - 根据扫描行数评估严重程度

2. **缺失索引 (High)**

   - 检测有可用索引但未使用的情况
   - 分析 `possible_keys` 和 `key` 字段

3. **临时表使用 (Medium)**

   - 检测 `Using temporary` 操作
   - 增加内存使用和磁盘 I/O，影响查询性能

4. **文件排序 (Medium)**

   - 检测 `Using filesort` 操作
   - 无法使用索引排序，需要额外的排序开销

5. **大量行扫描 (High)**
   - 检测扫描行数过多的查询
   - 可配置阈值

### 性能评分算法

评分基于以下因素：

- 基础分数：100 分
- 问题严重程度扣分：
  - Critical: -30 分
  - High: -20 分
  - Medium: -10 分
  - Low: -5 分
- 扫描行数扣分：
  - 大于 10 万行: -20 分
  - 大于 1 万行: -10 分
  - 大于 1 千行: -1 分

最终评分范围：0-100 分


## 🛠️ 安装和配置

### 1. 基础安装
推荐使用 uv 管理项目依赖

```bash
# 如果没有uv，需要安装
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
uv pip install .
```

### 2. 环境配置

工具支持多种配置方式，在.env中配置，可根据需要选择合适的 AI 分析模式：

#### 选项一：Ollama 本地模式


```bash
# Ollama 模型名称（必填）
OLLAMA_MODEL=deepseek-r1:1.5b

# Ollama API 基础 URL（可选，默认: http://localhost:11434）
OLLAMA_BASE_URL=http://localhost:11434

# 请求超时时间，单位秒（可选，默认: 60.0）
OLLAMA_TIMEOUT=120.0

# 是否启用流式响应（可选，默认: false）
# 使用深度思考模型时强烈建议开启此选项，否则可能超时报错
OLLAMA_STREAM=false
```

#### 选项二：云端 AI 模式

```bash
# OpenAI API 密钥（必填）
OPENAI_API_KEY=your_openai_api_key_here

# 模型名称（必填）
# 目前暂不建议使用deepseek-reasoner，后续会支持
OPENAI_MODEL=deepseek-chat

# API 基础 URL（可选，用于自定义端点）
OPENAI_BASE_URL=https://api.deepseek.com

# 请求超时时间，单位秒（可选，默认: 60.0）
OPENAI_TIMEOUT=60.0

# 最大重试次数（可选，默认: 3）
OPENAI_MAX_RETRIES=3
```

#### 选项三：仅基础分析模式

无需配置AI即可使用基础分析功能。

#### 数据库配置

```bash
MYSQL_HOST="127.0.0.1"
MYSQL_PORT="3306"
MYSQL_USER="username"
MYSQL_PASSWORD="password"
MYSQL_DATABASE="test"

#可选配置
MYSQL_SLOW_THRESHOLD="1"    # 慢查询时间阈值（秒）
MYSQL_ROWS_THRESHOLD="10000"   # 扫描行数阈值
MYSQL_SLOW_LIMIT="3"         # 分析的慢查询数量限制
```

### 3. 启用Performance Schema

```sql
-- 检查状态
SHOW VARIABLES LIKE 'performance_schema';

-- 如果显示 OFF，需要在 my.cnf 中启用并重启 MySQL
[mysqld]
performance_schema=ON
```

```sql
-- 检查 performance_schema的消费者是否开启
SELECT * FROM performance_schema.setup_consumers WHERE NAME = 'events_statements_history_long';

-- 如果 ENABLED 显示 NO，需要启用
UPDATE performance_schema.setup_consumers SET ENABLED = 'YES' WHERE NAME = 'events_statements_history_long';
```

### 4. 运行主程序

```bash
python app.py
```

### MySQL 慢查询数据源

工具支持多种慢查询数据源：

#### 1. Performance Schema（推荐）

从 MySQL 的 `performance_schema.events_statements_history_long` 表读取慢查询数据：

```sql
SELECT
            TIMER_WAIT / 1000000000000 as query_time,
            LOCK_TIME / 1000000000000 as lock_time,
            ROWS_SENT as rows_sent,
            ROWS_EXAMINED as rows_examined,
            SQL_TEXT as sql_statement,
            TIMER_START / 1000000000000 as timestamp_micro,
            SUBSTRING_INDEX(USER(), '@', 1) as user,
            SUBSTRING_INDEX(USER(), '@', -1) as host,
            CURRENT_SCHEMA as database_name
        FROM performance_schema.events_statements_history_long
        WHERE TIMER_WAIT / 1000000000000 >= 1
        AND ROWS_EXAMINED >= 1000
        AND SQL_TEXT IS NOT NULL
        AND SQL_TEXT NOT LIKE 'SHOW%'
        AND SQL_TEXT NOT LIKE 'EXPLAIN%'
        ORDER BY TIMER_WAIT DESC
        LIMIT 3
```

#### 2. 进程列表回退

如果 Performance Schema 不可用，工具会自动回退到使用 `INFORMATION_SCHEMA.PROCESSLIST`：

```sql
SELECT
    ID,
    USER,
    HOST,
    DB,
    COMMAND,
    TIME,
    STATE,
    INFO
FROM INFORMATION_SCHEMA.PROCESSLIST
WHERE COMMAND = 'Query'
AND TIME >= 1.0
AND INFO IS NOT NULL
ORDER BY TIME DESC
```

备注：information_schema.processlist 是 MySQL 中一个系统视图，用于展示**当前**服务器中所有正在运行的线程信息

#### 3. 日志文件读取（计划中）

未来版本将支持直接从慢查询日志文件读取数据。

### 常见问题

**Q: 为什么没有检测到慢查询？**
A: 请检查：

- Performance Schema 是否启用
- 慢查询阈值设置是否合理（包括时间、扫描行数）

**Q: 云端 AI 分析功能无法使用？**
A: 请检查：

- OPENAI_API_KEY 是否正确设置
- 网络连接是否正常
- API 密钥是否有足够的额度

**Q: 如果没有 API 密钥，工具还能正常使用吗？**
A: 完全可以！基础分析模式提供了快速 SQL 性能分析功能：

- ✅ EXPLAIN 执行计划分析
- ✅ 常见性能问题检测（全表扫描、缺失索引等）
- ✅ 性能评分计算
- ✅ 结构化的基础优化建议
- ✅ HTML 可视化报告


