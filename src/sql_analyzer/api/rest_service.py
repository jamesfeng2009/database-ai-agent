from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sql_analyzer.agent import SQLAnalyzerAgent
from sql_analyzer.agent_core.services.auto_optimizer import AutoOptimizer
from sql_analyzer.agent_core.services.safety_validator import (
    SafetyValidator,
    User,
    UserRole,
)
from sql_analyzer.config import OpenAIConfig
from sql_analyzer.database import DatabaseConfig, DatabaseType, MySQLConfig, PostgreSQLConfig
from sql_analyzer.database import SlowQueryConfig
from sql_analyzer.models import SQLAnalysisResponse
from sql_analyzer.slow_query_analyzer import (
    BatchAnalysisResult,
    SlowQueryAnalyzer,
    SlowQueryAnalysisResult,
    create_slow_query_analyzer,
)


logger = logging.getLogger(__name__)


class AnalyzeSQLRequest(BaseModel):
    """Request body for /analyze_sql."""

    sql: str


class AnalyzeSlowQueriesRequest(BaseModel):
    """Request body for /analyze_slow_queries."""

    limit: Optional[int] = None


class OptimizeSQLRequest(BaseModel):
    """Request body for /optimize_sql."""

    sql: str
    auto_apply: bool = False


class OptimizeSQLResponse(BaseModel):
    """Response body for /optimize_sql.

    结构与 AutoOptimizer.optimize_query 的返回字段对应。
    """

    original_sql: str
    best_sql: str
    best_cost: float
    original_cost: float
    improvement: float
    best_reason: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]]
    rewrite_suggestions: List[Dict[str, Any]]
    auto_applied: bool


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    ai_enabled: bool
    db_type: Optional[str] = None


app = FastAPI(title="SQL Analyzer REST Service", version="0.1.0")


# Global analyzer / optimizer instances (initialised on startup)
slow_query_analyzer: Optional[SlowQueryAnalyzer] = None
ai_analyzer: Optional[SQLAnalyzerAgent] = None
safety_validator: Optional[SafetyValidator] = None
auto_optimizer: Optional[AutoOptimizer] = None


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _load_database_config_from_env() -> DatabaseConfig:
    """Load a DatabaseConfig from environment variables.

    Currently supports MySQL and PostgreSQL.
    """

    db_type = os.getenv("DB_TYPE", "mysql").lower()
    host = _get_required_env("DB_HOST")
    port = int(os.getenv("DB_PORT", "0") or 0)
    user = _get_required_env("DB_USER")
    password = _get_required_env("DB_PASSWORD")
    database = _get_required_env("DB_NAME")

    if db_type == DatabaseType.MYSQL.value:
        if not port:
            port = 3306
        return MySQLConfig(host=host, port=port, user=user, password=password, database=database)
    if db_type == DatabaseType.POSTGRESQL.value:
        if not port:
            port = 5432
        return PostgreSQLConfig(host=host, port=port, user=user, password=password, database=database)

    raise RuntimeError(f"Unsupported DB_TYPE: {db_type}. Only mysql and postgresql are supported by this REST service.")


@lru_cache(maxsize=1)
def _load_openai_config() -> Optional[OpenAIConfig]:
    """Try to load OpenAIConfig from env; return None if not configured."""

    try:
        return OpenAIConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAIConfig not loaded: %s", exc)
        return None


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise AI analyzer and slow query analyzer."""

    global slow_query_analyzer, ai_analyzer, safety_validator, auto_optimizer

    logging.basicConfig(level=logging.INFO)

    # Database config
    db_config = _load_database_config_from_env()
    slow_query_config = SlowQueryConfig()  # use defaults

    # AI analyzer (optional)
    openai_cfg = _load_openai_config()
    if openai_cfg is not None:
        try:
            ai_analyzer = SQLAnalyzerAgent.from_config(openai_cfg)
            logger.info("AI analyzer initialised with model=%s", openai_cfg.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialise SQLAnalyzerAgent, fallback to basic analysis: %s", exc)
            ai_analyzer = None
    else:
        logger.info("OPENAI_* environment variables not configured, using basic analysis only")

    # Slow query analyzer (always available)
    slow_query_analyzer = await create_slow_query_analyzer(
        database_config=db_config,
        slow_query_config=slow_query_config,
        ai_analyzer=ai_analyzer,
    )
    logger.info("SlowQueryAnalyzer initialised for DB %s", db_config.get_database_type().value)

    # Auto optimizer and safety validator
    safety_validator = SafetyValidator()
    auto_optimizer = AutoOptimizer(safety_validator)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Cleanup resources."""

    global slow_query_analyzer

    if slow_query_analyzer is not None:
        try:
            await slow_query_analyzer.cleanup()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during SlowQueryAnalyzer cleanup: %s", exc)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple health check endpoint."""

    if slow_query_analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialised")

    db_connected = False
    try:
        if slow_query_analyzer.connector is not None:
            db_connected = await slow_query_analyzer.connector.test_connection()
    except Exception:  # noqa: BLE001
        db_connected = False

    ai_enabled = ai_analyzer is not None
    db_type = slow_query_analyzer.database_config.get_database_type().value

    status = "ok" if db_connected else "degraded"

    return HealthResponse(status=status, db_connected=db_connected, ai_enabled=ai_enabled, db_type=db_type)


@app.post("/analyze_sql", response_model=SlowQueryAnalysisResult)
async def analyze_sql(request: AnalyzeSQLRequest) -> SlowQueryAnalysisResult:
    """Analyze a single SQL statement.

    This will execute EXPLAIN against the configured database and then use
    SQLAnalyzerAgent (if available) or the built-in basic analyzer.
    """

    if slow_query_analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialised")

    if not request.sql or not request.sql.strip():
        raise HTTPException(status_code=400, detail="SQL must not be empty")

    try:
        result = await slow_query_analyzer.analyze_single_query(request.sql)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("analyze_sql failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze_slow_queries", response_model=BatchAnalysisResult)
async def analyze_slow_queries(request: AnalyzeSlowQueriesRequest) -> BatchAnalysisResult:
    """Analyze slow queries from the configured source (log/performance_schema).

    The behaviour and source are controlled by SlowQueryConfig and the
    SlowQueryReader created inside SlowQueryAnalyzer.
    """

    if slow_query_analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialised")

    try:
        result = await slow_query_analyzer.analyze_slow_queries(limit=request.limit)

        # 为每条慢查询附加自动优化结果 (如果可用)
        if auto_optimizer is not None and slow_query_analyzer.connector is not None:
            db_name = slow_query_analyzer.database_config.database
            user = User(
                user_id="rest_api",
                username="rest_api",
                role=UserRole.DEVELOPER,
                database_access={db_name},
            )

            for item in result.results:
                try:
                    opt = await auto_optimizer.optimize_query(
                        sql=item.slow_query.sql_statement,
                        connector=slow_query_analyzer.connector,
                        user=user,
                        auto_apply=False,
                    )
                    item.optimize_result = {
                        "best_sql": opt.get("best_sql"),
                        "best_reason": opt.get("best_reason"),
                        "improvement": opt.get("improvement"),
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("optimize_query for slow query failed: %s", exc)
                    continue

        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("analyze_slow_queries failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/optimize_sql", response_model=OptimizeSQLResponse)
async def optimize_sql(request: OptimizeSQLRequest) -> OptimizeSQLResponse:
    """Generate rewrite recommendations and cost evaluation for a single SQL.

    当前实现只调用 AutoOptimizer.optimize_query 进行候选生成与 EXPLAIN 评估,
    不会在数据库中自动执行任何变更。
    """

    if slow_query_analyzer is None or slow_query_analyzer.connector is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialised")

    if auto_optimizer is None:
        raise HTTPException(status_code=503, detail="Auto optimizer not initialised")

    if not request.sql or not request.sql.strip():
        raise HTTPException(status_code=400, detail="SQL must not be empty")

    # 构造一个用于安全验证/审计的虚拟用户
    db_name = slow_query_analyzer.database_config.database
    user = User(
        user_id="rest_api",
        username="rest_api",
        role=UserRole.DEVELOPER,
        database_access={db_name},
    )

    try:
        result_dict = await auto_optimizer.optimize_query(
            sql=request.sql,
            connector=slow_query_analyzer.connector,
            user=user,
            auto_apply=request.auto_apply,
        )
        return OptimizeSQLResponse(**result_dict)
    except Exception as exc:  # noqa: BLE001
        logger.error("optimize_sql failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
