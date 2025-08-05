"""SQL 分析器主包."""

from .agent import (
    SQLAnalyzerAgent,
    OllamaAgent,
    create_agent_from_config,
    create_ollama_agent,
    create_sql_analyzer_agent,
)
from .analyzer_base import BaseSQLAnalyzer, get_default_system_message
from .config import (
    OpenAIConfig,
    OllamaConfig,
    load_config_from_env,
)
from .database import (
    BaseDatabaseConnector,
    BaseSlowQueryReader,
    DatabaseConfig,
    DatabaseType,
    create_and_connect_database_connector,
    create_database_connector,
    create_slow_query_reader,
)
from .database.models import (
    MySQLConfig,
    PostgreSQLConfig,
    SlowQueryConfig,
)
from .models import (
    BatchAnalysisResult,
    ExplainResult,
    OptimizationSuggestion,
    PerformanceIssue,
    SlowQueryAnalysisResult,
    SlowQueryEntry,
    SQLAnalysisRequest,
    SQLAnalysisResponse,
)
from .report import generate_html_report
from .slow_query_analyzer import SlowQueryAnalyzer, create_slow_query_analyzer
from .tools import (
    calculate_performance_score,
    detect_performance_issues,
    format_analysis_request,
    generate_optimization_suggestions,
)

__all__ = [
    # 智能体相关
    "BaseSQLAnalyzer",
    "SQLAnalyzerAgent",
    "OllamaAgent",
    "create_agent_from_config",
    "create_ollama_agent",
    "create_sql_analyzer_agent",
    "get_default_system_message",
    
    # 配置相关
    "OpenAIConfig",
    "OllamaConfig",
    "load_config_from_env",
    
    # 数据库抽象层
    "BaseDatabaseConnector",
    "BaseSlowQueryReader",
    "DatabaseConfig",
    "DatabaseType",
    "MySQLConfig",
    "PostgreSQLConfig",
    "SlowQueryConfig",
    "create_and_connect_database_connector",
    "create_database_connector",
    "create_slow_query_reader",
    
    # 数据模型
    "BatchAnalysisResult",
    "ExplainResult",
    "OptimizationSuggestion",
    "PerformanceIssue",
    "SlowQueryAnalysisResult",
    "SlowQueryEntry",
    "SQLAnalysisRequest",
    "SQLAnalysisResponse",
    
    # 分析器
    "SlowQueryAnalyzer",
    "create_slow_query_analyzer",
    
    # 工具函数
    "calculate_performance_score",
    "detect_performance_issues",
    "format_analysis_request",
    "generate_optimization_suggestions",
    
    # 报告生成
    "generate_html_report",
] 