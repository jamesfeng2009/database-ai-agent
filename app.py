#!/usr/bin/env python3
"""SQL 分析器主程序入口
"""
import asyncio
import os
import warnings
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# 在导入aiomysql相关模块之前设置警告过滤
warnings.filterwarnings("ignore", module="aiomysql")
warnings.filterwarnings("ignore", module="pymysql")
warnings.filterwarnings("ignore", message=".*select#.*")

from src.sql_analyzer import (
    BatchAnalysisResult,
    generate_html_report,
)
from src.sql_analyzer.agent import create_agent_from_config
from src.sql_analyzer.analyzer_base import BaseSQLAnalyzer
from src.sql_analyzer.config import load_config_from_env
from src.sql_analyzer.database import (
    DatabaseType,
    MySQLConfig,
    PostgreSQLConfig,
    SlowQueryConfig,
    create_and_connect_database_connector,
    create_slow_query_reader,
)
from src.sql_analyzer.models import SQLAnalysisResponse

load_dotenv()


def check_database_environment_variables(database_type: DatabaseType) -> Optional[List[str]]:
    """检查数据库环境变量是否设置完整。
    
    Args:
        database_type: 数据库类型
        
    Returns:
        如果有缺失的环境变量，返回缺失变量的列表；如果全部设置，返回 None
    """
    if database_type == DatabaseType.MYSQL:
        required_vars = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
        prefix = "MYSQL"
    elif database_type == DatabaseType.POSTGRESQL:
        required_vars = ["POSTGRESQL_HOST", "POSTGRESQL_USER", "POSTGRESQL_PASSWORD", "POSTGRESQL_DATABASE"]
        prefix = "POSTGRESQL"
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return missing_vars if missing_vars else None


def create_database_config(database_type: DatabaseType) -> MySQLConfig | PostgreSQLConfig:
    """创建数据库配置对象。
    
    Args:
        database_type: 数据库类型
        
    Returns:
        配置好的数据库配置对象
    """
    if database_type == DatabaseType.MYSQL:
        return MySQLConfig(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            charset="utf8mb4",
            connect_timeout=10,
            max_connections=5
        )
    elif database_type == DatabaseType.POSTGRESQL:
        return PostgreSQLConfig(
            host=os.getenv("POSTGRESQL_HOST", "localhost"),
            port=int(os.getenv("POSTGRESQL_PORT", "5432")),
            user=os.getenv("POSTGRESQL_USER"),
            password=os.getenv("POSTGRESQL_PASSWORD"),
            database=os.getenv("POSTGRESQL_DATABASE"),
            ssl_mode=os.getenv("POSTGRESQL_SSL_MODE", "prefer"),
            application_name="sql_analyzer",
            connect_timeout=10,
            max_connections=5
        )
    else:
        raise ValueError(f"不支持的数据库类型: {database_type}")


def create_slow_query_config() -> SlowQueryConfig:
    """创建慢查询配置对象。
    
    Returns:
        配置好的 SlowQueryConfig 对象
    """
    return SlowQueryConfig(
        use_performance_schema=True,
        query_time_threshold=float(os.getenv("SQL_SLOW_THRESHOLD", "1.0")),
        rows_examined_threshold=int(os.getenv("SQL_ROWS_THRESHOLD", "1000")),
        limit=int(os.getenv("SQL_SLOW_LIMIT", "5")),
        time_range_hours=int(os.getenv("SQL_TIME_RANGE", "24"))
    )


def detect_database_type() -> DatabaseType:
    """检测要使用的数据库类型。
    
    Returns:
        检测到的数据库类型
    """
    # 检查环境变量来确定数据库类型
    mysql_vars = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    postgresql_vars = ["POSTGRESQL_HOST", "POSTGRESQL_USER", "POSTGRESQL_PASSWORD", "POSTGRESQL_DATABASE"]
    
    mysql_configured = all(os.getenv(var) for var in mysql_vars)
    postgresql_configured = all(os.getenv(var) for var in postgresql_vars)
    
    if mysql_configured and postgresql_configured:
        # 如果都配置了，优先使用MySQL（向后兼容）
        print("⚠️  检测到MySQL和PostgreSQL都配置了，默认使用MySQL")
        return DatabaseType.MYSQL
    elif mysql_configured:
        return DatabaseType.MYSQL
    elif postgresql_configured:
        return DatabaseType.POSTGRESQL
    else:
        # 默认使用MySQL
        return DatabaseType.MYSQL


def detect_and_create_ai_analyzer() -> Tuple[Optional[BaseSQLAnalyzer], str]:
    """检测可用的AI配置并创建对应的智能体。
    
    Returns:
        tuple: (智能体实例或None, 模式描述)
    """
    # 加载配置
    config = load_config_from_env()
    
    # 优先检查 Ollama 配置（推荐模式）
    if config["ollama"]:
        try:
            print("🦙 检测到 Ollama 配置，将使用本地大模型分析")
            print(f"   模型: {config['ollama'].model}")
            print(f"   端点: {config['ollama'].base_url}")
            agent = create_agent_from_config("ollama")
            return agent, "Ollama 本地模式"
        except Exception as e:
            print(f"⚠️  Ollama 智能体创建失败: {e}")
            print("   回退到其他模式...")
    
    # 检查 OpenAI 配置
    if config["openai"]:
        try:
            print("🤖 检测到 OpenAI 配置，将使用云端大模型分析")
            print(f"   模型: {config['openai'].model}")
            print(f"   端点: {config['openai'].base_url}")
            agent = create_agent_from_config("openai")
            return agent, "云端 AI 模式"
        except Exception as e:
            print(f"⚠️  OpenAI 智能体创建失败: {e}")
            print("   回退到基础分析模式...")
    
    # 回退到基础分析模式
    print("⚠️  未配置 AI 模型，将使用基础分析模式")
    return None, "基础分析模式"


async def test_ai_analyzer_connection(analyzer: BaseSQLAnalyzer, mode: str) -> bool:
    """测试AI智能体连接。
    
    Args:
        analyzer: 智能体实例
        mode: 模式描述
        
    Returns:
        连接测试是否成功
    """
    if analyzer is None:
        return True  # 基础分析模式无需测试连接
    
    print(f"🔍 正在测试 {mode} 连接...")
    try:
        result = await analyzer.test_connection()
        if result["success"]:
            print(f"✅ {mode} 连接测试成功")
            return True
        else:
            print(f"❌ {mode} 连接测试失败: {result['error']}")
            if result.get("diagnosis"):
                print(f"   诊断: {result['diagnosis']}")
            print("   将回退到基础分析模式")
            return False
    except Exception as e:
        print(f"❌ {mode} 连接测试异常: {e}")
        print("   将回退到基础分析模式")
        return False


def show_ai_configuration_help() -> None:
    """显示AI配置帮助信息。"""
    print("\n💡 AI 模式配置提示:")
    print("   🦙 Ollama 本地模式（推荐）:")
    print("      export OLLAMA_MODEL=llama3.2:3b")
    print("      export OLLAMA_BASE_URL=http://localhost:11434")
    print("")
    print("   🤖 AI 云端模式:")
    print("      export OPENAI_API_KEY=your_api_key")
    print("      export OPENAI_MODEL=deepseek-chat")
    print("      export OPENAI_BASE_URL=https://api.deepseek.com")
    print("")
    print("   📋 更多配置选项请查看: config/env.example")


def show_database_configuration_help(database_type: DatabaseType) -> None:
    """显示数据库配置帮助信息。
    
    Args:
        database_type: 数据库类型
    """
    if database_type == DatabaseType.MYSQL:
        print("\n💡 MySQL 配置提示:")
        print("   export MYSQL_HOST=localhost")
        print("   export MYSQL_PORT=3306")
        print("   export MYSQL_USER=your_username")
        print("   export MYSQL_PASSWORD=your_password")
        print("   export MYSQL_DATABASE=your_database")
    elif database_type == DatabaseType.POSTGRESQL:
        print("\n💡 PostgreSQL 配置提示:")
        print("   export POSTGRESQL_HOST=localhost")
        print("   export POSTGRESQL_PORT=5432")
        print("   export POSTGRESQL_USER=your_username")
        print("   export POSTGRESQL_PASSWORD=your_password")
        print("   export POSTGRESQL_DATABASE=your_database")
        print("   export POSTGRESQL_SSL_MODE=prefer")


def display_analysis_summary(batch_result: BatchAnalysisResult, mode: str) -> None:
    """显示分析结果摘要。
    
    Args:
        batch_result: 批量分析结果
        mode: 分析模式描述
    """
    print(f"\n📋 分析结果摘要 ({mode}):")
    print(f"   总查询数: {batch_result.total_queries}")
    print(f"   成功分析: {batch_result.analyzed_queries}")
    print(f"   分析失败: {batch_result.failed_queries}")
    print(f"   分析耗时: {(batch_result.end_time - batch_result.start_time).total_seconds():.2f} 秒")
    
    # 显示统计信息
    stats = batch_result.summary_stats
    if stats:
        print("\n📈 统计信息:")
        print(f"   平均性能评分: {stats.get('average_performance_score', 'N/A')}")
        print(f"   总问题数: {stats.get('total_issues', 0)}")
        print(f"   平均查询时间: {stats.get('average_query_time', 'N/A')} 秒")
        print(f"   全表扫描查询数: {stats.get('queries_with_full_table_scan', 0)}")
        print(f"   低评分查询数 (<60): {stats.get('queries_with_score_below_60', 0)}")


def display_detailed_results(batch_result: BatchAnalysisResult, limit: int) -> None:
    """显示详细分析结果。
    
    Args:
        batch_result: 批量分析结果
        limit: 显示的查询数量限制
    """
    if not batch_result.results:
        print("\n📭 未发现符合条件的慢查询")
        print("   可能原因:")
        print("   • 查询时间阈值设置过高")
        print("   • 时间范围内没有符合条件的查询")
        print("   • performance_schema 未开启或数据不完整")
        return
    
    print(f"\n🔍 详细分析结果 (显示前{limit}个):")
    for i, result in enumerate(batch_result.results[:limit], 1):
        print(f"\n--- 查询 {i} ---")
        slow_query = result.slow_query
        analysis = result.analysis_response
        
        print(f"SQL: {slow_query.sql_statement[:80]}...")
        print(f"执行时间: {slow_query.query_time:.3f} 秒")
        print(f"扫描行数: {slow_query.rows_examined:,}")
        print(f"性能评分: {analysis.performance_score}/100")
        print(f"问题数量: {len(analysis.issues)}")
        print(f"建议数量: {len(analysis.suggestions)}")
        
        if analysis.issues:
            print("主要问题:")
            for issue in analysis.issues[:2]:
                severity_icon = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(issue.severity, "⚪")
                print(f"  {severity_icon} {issue.issue_type}")


def generate_reports(batch_result: BatchAnalysisResult, limit: int, mode: str, database_type: DatabaseType) -> None:
    """生成 HTML 报告。
    
    Args:
        batch_result: 批量分析结果
        limit: 生成报告的查询数量限制
        mode: 分析模式描述
        database_type: 数据库类型
    """
    if not batch_result.results:
        return
    
    db_name = "MySQL" if database_type == DatabaseType.MYSQL else "PostgreSQL"
    
    for i, result in enumerate(batch_result.results[:limit], 1):
        print(f"\n📄 生成第{i}个查询的 HTML 报告...")
        generate_html_report(
            response=result.analysis_response,
            sql_statement=result.slow_query.sql_statement,
            output_path=f"{database_type.value}_slow_query_analysis_{i}.html",
            title=f"{db_name} 慢查询分析报告 - {mode} (评分: {result.analysis_response.performance_score})"
        )
        print(f"✅ HTML 报告已生成: {database_type.value}_slow_query_analysis_{i}.html")


async def slow_query_analysis() -> None:
    """慢查询分析主函数。"""
    # 检测数据库类型
    database_type = detect_database_type()
    db_name = "MySQL" if database_type == DatabaseType.MYSQL else "PostgreSQL"
    
    # 检查数据库环境变量
    missing_vars = check_database_environment_variables(database_type)
    if missing_vars:
        print(f"\n⚠️  未设置 {db_name} 连接环境变量！！！")
        print(f"   要体验 {db_name} 慢查询分析功能，请设置以下环境变量:")
        for var in missing_vars:
            print(f"   export {var}='your_{var.lower()}'")
        show_database_configuration_help(database_type)
        return
    
    print(f"\n🗄️  {db_name} 慢查询分析开始 🚀")
    print("=" * 50)
    
    try:
        # 配置数据库连接和慢查询
        database_config = create_database_config(database_type)
        slow_query_config = create_slow_query_config()
        
        # 检测并创建AI分析器
        ai_analyzer, mode = detect_and_create_ai_analyzer()
        
        # 测试AI分析器连接（如果有的话）
        connection_ok = await test_ai_analyzer_connection(ai_analyzer, mode)
        if not connection_ok:
            ai_analyzer = None
            mode = "基础分析模式"
        
        print(f"🔌 正在连接 {db_name} 数据库 {database_config.host}:{database_config.port}/{database_config.database}...")
        
        # 创建数据库连接器
        connector = await create_and_connect_database_connector(database_config)
        
        # 测试数据库连接
        if not await connector.test_connection():
            raise RuntimeError(f"{db_name} 连接测试失败")
        
        print(f"✅ {db_name} 连接成功")
        
        # 创建慢查询读取器
        slow_query_reader = create_slow_query_reader(connector, slow_query_config)
        
        # 执行慢查询分析
        print(f"\n📊 正在使用 {mode} 分析慢查询日志...")
        if ai_analyzer and "Ollama" in mode:
            print("   💡 使用本地大模型，数据安全，无API费用")
        elif ai_analyzer and "OpenAI" in mode:
            print("   🧠 正在进行 云端 AI 深度分析...")
        else:
            print("   ⚡ 使用规则引擎快速分析")
        
        # 获取慢查询
        slow_queries = await slow_query_reader.get_slow_queries()
        
        if not slow_queries:
            print("\n📭 未发现符合条件的慢查询")
            print("   可能原因:")
            print("   • 查询时间阈值设置过高")
            print("   • 时间范围内没有符合条件的查询")
            print("   • 性能统计未开启或数据不完整")
            return
        
        print(f"获取到 {len(slow_queries)} 条慢查询，开始分析...")
        
        # 分析每个慢查询
        results = []
        analyzed_count = 0
        failed_count = 0
        
        for i, slow_query in enumerate(slow_queries, 1):
            try:
                print(f"正在分析第 {i}/{len(slow_queries)} 条查询...")
                
                # 执行 EXPLAIN
                explain_results = await connector.execute_explain(slow_query.sql_statement)
                
                # 创建分析请求
                from src.sql_analyzer.models import SQLAnalysisRequest, SlowQueryAnalysisResult
                request = SQLAnalysisRequest(
                    sql_statement=slow_query.sql_statement,
                    explain_results=explain_results,
                    database_schema=slow_query.database,
                    additional_context=f"慢查询信息: 执行时间={slow_query.query_time}秒, 扫描行数={slow_query.rows_examined}"
                )
                
                # 执行分析
                if ai_analyzer:
                    analysis_response = await ai_analyzer.analyze_sql(request)
                else:
                    # 使用基础分析
                    from src.sql_analyzer.tools import (
                        calculate_performance_score,
                        detect_performance_issues,
                        generate_optimization_suggestions,
                    )
                    
                    issues = detect_performance_issues(request)
                    suggestions = generate_optimization_suggestions(request, issues)
                    score = calculate_performance_score(request, issues)
                    
                    analysis_response = SQLAnalysisResponse(
                        summary=f"查询性能{'良好' if score >= 80 else '一般' if score >= 60 else '较差'}（评分：{score}/100）",
                        performance_score=score,
                        issues=issues,
                        suggestions=suggestions,
                        detailed_analysis=f"基础分析完成，发现 {len(issues)} 个问题，提供 {len(suggestions)} 个建议。",
                        execution_plan_analysis="执行计划分析完成",
                        explain_results=explain_results
                    )
                
                # 创建结果
                result = SlowQueryAnalysisResult(
                    slow_query=slow_query,
                    analysis_response=analysis_response,
                    explain_results=explain_results
                )
                results.append(result)
                analyzed_count += 1
                
            except Exception as e:
                print(f"分析第 {i} 条查询失败: {e}")
                failed_count += 1
                continue
        
        # 创建批量分析结果
        from datetime import datetime
        from src.sql_analyzer.models import BatchAnalysisResult
        
        batch_result = BatchAnalysisResult(
            total_queries=len(slow_queries),
            analyzed_queries=analyzed_count,
            failed_queries=failed_count,
            results=results,
            summary_stats={},  # 可以添加统计信息
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # 显示分析结果
        display_analysis_summary(batch_result, mode)
        display_detailed_results(batch_result, slow_query_config.limit)
        
        # 生成 HTML 报告
        generate_reports(batch_result, slow_query_config.limit, mode, database_type)
        
        # 清理资源
        await connector.disconnect()
        
        # 如果使用基础分析模式，显示AI配置提示
        if ai_analyzer is None:
            show_ai_configuration_help()
        
    except Exception as e:
        print(f"❌ {db_name} 慢查询分析失败: {e}")
        print("   请检查:")
        print("   • 数据库连接配置是否正确")
        print("   • 数据库用户是否有足够权限")
        print("   • 性能统计是否开启")
        if "ollama" in str(e).lower():
            print("   • Ollama 服务是否正在运行")
            print("   • 模型是否已下载")


def main() -> None:
    """主函数."""
    print(" 基于 AI 的智能数据库慢查询分析工具 ")
    print("=" * 60)
    print("🚀 支持三种分析模式:")
    print("   ⚡ 基础分析模式 - 快速诊断")
    print("   🦙 Ollama 本地模式 - 隐私保护，零费用")
    print("   🤖 云端AI模式 - 高质量分析")
    print("\n🗄️  支持数据库类型:")
    print("   🐬 MySQL - 完整支持")
    print("   🐘 PostgreSQL - 完整支持")
    
    
    # 运行异步
    async def run_async() -> None:
        """运行异步分析."""
        
        # 慢查询分析 
        await slow_query_analysis()
    
    asyncio.run(run_async())
    
    print("\n✅ 运行结束！！！")
    print("\n📖 更多信息:")
    print("   • 查看 README.md 了解完整功能")
    print("   • 查看生成的 HTML 报告获取详细分析结果")
    

if __name__ == "__main__":
    main() 