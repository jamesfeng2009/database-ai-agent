from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sql_analyzer.agent import SQLAnalyzerAgent
from sql_analyzer.agent_core.services.auto_optimizer import (
    AutoOptimizer,
    IndexSpec,
    OptimizationStatus,
)
from sql_analyzer.agent_core.services.safety_validator import (
    SafetyValidator,
    User,
    UserRole,
)
from sql_analyzer.agent_core.services.rollback_manager import (
    RollbackManager,
    SnapshotType,
    RollbackStrategy,
)
from sql_analyzer.agent_core.services.knowledge_service import KnowledgeService
from sql_analyzer.agent_core.management.agent_orchestrator import (
    AgentOrchestrator,
    WorkflowStatus,
    TaskPriority,
)
from sql_analyzer.nlp import NLPProcessor
from sql_analyzer.config import OpenAIConfig
from sql_analyzer.database import DatabaseConfig, DatabaseType, MySQLConfig, PostgreSQLConfig
from sql_analyzer.database import SlowQueryConfig
from sql_analyzer.models import (
    SQLAnalysisResponse,
    SlowQueryOptimizationTask,
    SlowQueryOptimizationTaskStatus,
    SlowQueryEntry,
)
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
    use_orchestrator: bool = False


class AnalyzeSlowQueriesRequest(BaseModel):
    """Request body for /analyze_slow_queries."""

    limit: Optional[int] = None
    use_orchestrator: bool = False


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


class OptimizationTaskReviewRequest(BaseModel):
    """Request body for approving / rejecting / executing optimization tasks."""

    reviewer: Optional[str] = None
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    ai_enabled: bool
    db_type: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for /chat natural language endpoint."""

    text: str
    session_id: Optional[str] = None
    language: Optional[str] = "zh"


class ChatResponse(BaseModel):
    """High-level response for /chat.

    包含意图解析结果、简要回复, 以及可能触发的子操作结果(如 SQL 分析或知识查询结果)。
    """

    intent: Dict[str, Any]
    reply: str
    sql_analysis: Optional[SlowQueryAnalysisResult] = None
    slow_queries: Optional[BatchAnalysisResult] = None
    knowledge_entries: Optional[List[Dict[str, Any]]] = None


app = FastAPI(title="SQL Analyzer REST Service", version="0.1.0")


# Global analyzer / optimizer instances (initialised on startup)
slow_query_analyzer: Optional[SlowQueryAnalyzer] = None
ai_analyzer: Optional[SQLAnalyzerAgent] = None
safety_validator: Optional[SafetyValidator] = None
auto_optimizer: Optional[AutoOptimizer] = None
rollback_manager: Optional[RollbackManager] = None
knowledge_service: Optional[KnowledgeService] = None
nlp_processor: Optional[NLPProcessor] = None
agent_orchestrator: Optional[AgentOrchestrator] = None


TASK_TABLE_NAME = "slow_query_optimization_tasks"


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


def _get_db_connector():
    """Helper to get current DB connector or raise 503-style error context.

    NOTE: This helper itself does not raise HTTPException; endpoints should
    handle error cases and return appropriate HTTP errors.
    """

    if slow_query_analyzer is None or slow_query_analyzer.connector is None:
        return None
    return slow_query_analyzer.connector


async def _ensure_task_table() -> None:
    """Create the task table in the target database if it does not exist.

    The DDL is written to be compatible with both MySQL and PostgreSQL.
    """

    connector = _get_db_connector()
    if connector is None:
        return

    # 使用通用 DDL，TEXT/VARCHAR 在 MySQL / PostgreSQL 均可用
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TASK_TABLE_NAME} (
      task_id         VARCHAR(64) PRIMARY KEY,
      status          VARCHAR(32) NOT NULL,
      slow_query      TEXT        NOT NULL,
      optimize_result TEXT        NOT NULL,
      reviewer        TEXT,
      review_comment  TEXT,
      created_at      TEXT        NOT NULL,
      updated_at      TEXT        NOT NULL,
      executed_at     TEXT
    )
    """
    try:
        await connector.execute_query(create_sql)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to ensure task table exists: %s", exc)


def _serialize_json(obj: Any) -> str:
    """Serialize Python object to JSON string safe for inline SQL.

    为简单起见，仅做 JSON dumps + 单引号转义，避免打断字符串。
    更严格的实现可以改为使用预编译/参数化查询。
    """

    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("'", "''")


def _deserialize_json(s: str) -> Any:
    return json.loads(s)


async def _insert_optimization_task(task: SlowQueryOptimizationTask) -> None:
    connector = _get_db_connector()
    if connector is None:
        return

    slow_query_json = _serialize_json(task.slow_query.dict())
    optimize_json = _serialize_json(task.optimize_result)
    reviewer = f"'{task.reviewer}'" if task.reviewer else "NULL"
    comment = f"'{task.review_comment}'" if task.review_comment else "NULL"
    created = task.created_at.isoformat()
    updated = task.updated_at.isoformat()

    sql = f"""
    INSERT INTO {TASK_TABLE_NAME}
      (task_id, status, slow_query, optimize_result, reviewer, review_comment, created_at, updated_at, executed_at)
    VALUES
      ('{task.task_id}', '{task.status.value}', '{slow_query_json}', '{optimize_json}',
       {reviewer}, {comment}, '{created}', '{updated}', NULL)
    """
    try:
        await connector.execute_query(sql)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to insert optimization task %s: %s", task.task_id, exc)


async def _row_to_task(row: Dict[str, Any]) -> SlowQueryOptimizationTask:
    slow_query_data = _deserialize_json(row["slow_query"])
    optimize_result = _deserialize_json(row["optimize_result"])
    slow_query = SlowQueryEntry.parse_obj(slow_query_data)

    created_at = datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.utcnow()
    updated_at = datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else created_at
    executed_at = (
        datetime.fromisoformat(row["executed_at"]) if row.get("executed_at") else None
    )

    status = SlowQueryOptimizationTaskStatus(row["status"])

    return SlowQueryOptimizationTask(
        task_id=row["task_id"],
        slow_query=slow_query,
        optimize_result=optimize_result,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        reviewer=row.get("reviewer"),
        review_comment=row.get("review_comment"),
        executed_at=executed_at,
    )


async def _get_optimization_task(task_id: str) -> Optional[SlowQueryOptimizationTask]:
    connector = _get_db_connector()
    if connector is None:
        return None

    sql = f"SELECT * FROM {TASK_TABLE_NAME} WHERE task_id = '{task_id}'"
    rows = await connector.execute_query(sql)
    if not rows:
        return None
    return await _row_to_task(rows[0])


async def _list_optimization_tasks() -> List[SlowQueryOptimizationTask]:
    connector = _get_db_connector()
    if connector is None:
        return []

    sql = f"SELECT * FROM {TASK_TABLE_NAME} ORDER BY created_at DESC"
    rows = await connector.execute_query(sql)
    tasks: List[SlowQueryOptimizationTask] = []
    for row in rows:
        try:
            task = await _row_to_task(row)
            tasks.append(task)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse optimization task row: %s", exc)
            continue
    return tasks


async def _update_optimization_task_status(
    task_id: str,
    status: SlowQueryOptimizationTaskStatus,
    reviewer: Optional[str] = None,
    comment: Optional[str] = None,
    executed_at: Optional[datetime] = None,
) -> None:
    connector = _get_db_connector()
    if connector is None:
        return

    reviewer_sql = f"'{reviewer}'" if reviewer else "NULL"
    comment_sql = f"'{comment}'" if comment else "NULL"
    executed_sql = f"'{executed_at.isoformat()}'" if executed_at else "NULL"
    updated_sql = datetime.utcnow().isoformat()

    sql = f"""
    UPDATE {TASK_TABLE_NAME}
    SET status = '{status.value}',
        reviewer = {reviewer_sql},
        review_comment = {comment_sql},
        updated_at = '{updated_sql}',
        executed_at = {executed_sql}
    WHERE task_id = '{task_id}'
    """
    try:
        await connector.execute_query(sql)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update optimization task %s: %s", task_id, exc)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise AI analyzer and slow query analyzer."""

    global slow_query_analyzer, ai_analyzer, safety_validator, auto_optimizer, rollback_manager, knowledge_service, nlp_processor, agent_orchestrator

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

    # Auto optimizer, safety validator, rollback manager and knowledge service
    safety_validator = SafetyValidator()
    rollback_manager = RollbackManager(safety_validator)

    # Knowledge service 基于与慢查询相同的业务数据库构建
    if slow_query_analyzer.connector is not None:
        knowledge_service = KnowledgeService(slow_query_analyzer.connector)
    else:
        knowledge_service = None

    auto_optimizer = AutoOptimizer(safety_validator, knowledge_service)

    # NLP 处理器
    nlp_processor = NLPProcessor()

    # AgentOrchestrator 及简单的 SQL 分析工作流模板
    agent_orchestrator = AgentOrchestrator()
    try:
        await agent_orchestrator.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to start AgentOrchestrator, orchestrator features disabled: %s", exc)
        agent_orchestrator = None

    # 注册基于 SlowQueryAnalyzer 的工作流模板
    if agent_orchestrator is not None:
        async def _sql_analysis_workflow_template(workflow, parameters: Dict[str, Any]) -> None:
            sql = parameters.get("sql")
            if not sql or slow_query_analyzer is None:
                raise RuntimeError("sql_analysis workflow requires sql and initialized analyzer")
            result = await slow_query_analyzer.analyze_single_query(sql)
            # 将结果序列化存入 workflow.result
            try:
                workflow.result = result.dict()
            except Exception:
                workflow.result = None

        async def _slow_query_batch_workflow_template(workflow, parameters: Dict[str, Any]) -> None:
            if slow_query_analyzer is None:
                raise RuntimeError("slow_query_batch_analysis workflow requires initialized analyzer")
            limit = parameters.get("limit")
            batch = await slow_query_analyzer.analyze_slow_queries(limit=limit)
            try:
                workflow.result = batch.dict()
            except Exception:
                workflow.result = None

        # 直接访问模板字典进行注册（保守实现）
        agent_orchestrator._workflow_templates["sql_analysis"] = _sql_analysis_workflow_template
        agent_orchestrator._workflow_templates["slow_query_batch_analysis"] = _slow_query_batch_workflow_template


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

    # 如果配置使用 orchestrator, 尝试通过工作流执行
    if request.use_orchestrator and agent_orchestrator is not None:
        try:
            wf_id = await agent_orchestrator.create_workflow(
                workflow_type="sql_analysis",
                name="sql_analysis_via_orchestrator",
                parameters={"sql": request.sql},
                priority=TaskPriority.MEDIUM,
            )
            await agent_orchestrator.execute_workflow(wf_id)

            # 简单轮询等待完成
            for _ in range(60):  # 最长约 60 秒
                status = await agent_orchestrator.get_workflow_status(wf_id)
                if not status:
                    break
                if status["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                    if status["result"]:
                        try:
                            return SlowQueryAnalysisResult.parse_obj(status["result"])
                        except Exception:
                            break
                    break
                await asyncio.sleep(1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_sql via orchestrator failed, fallback to direct path: %s", exc)

    # 回退: 直接调用 SlowQueryAnalyzer
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

    # 如需要, 通过 orchestrator 执行批量慢查询分析工作流
    if request.use_orchestrator and agent_orchestrator is not None:
        try:
            wf_id = await agent_orchestrator.create_workflow(
                workflow_type="slow_query_batch_analysis",
                name="slow_query_batch_via_orchestrator",
                parameters={"limit": request.limit},
                priority=TaskPriority.MEDIUM,
            )
            await agent_orchestrator.execute_workflow(wf_id)

            for _ in range(60):
                status = await agent_orchestrator.get_workflow_status(wf_id)
                if not status:
                    break
                if status["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                    if status["result"]:
                        try:
                            return BatchAnalysisResult.parse_obj(status["result"])
                        except Exception:
                            break
                    break
                await asyncio.sleep(1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_slow_queries via orchestrator failed, fallback to direct path: %s", exc)

    try:
        result = await slow_query_analyzer.analyze_slow_queries(limit=request.limit)

        # 为每条慢查询附加自动优化结果 (如果可用)，并创建 DB 持久化的优化任务
        if auto_optimizer is not None and slow_query_analyzer.connector is not None:
            db_name = slow_query_analyzer.database_config.database
            user = User(
                user_id="rest_api",
                username="rest_api",
                role=UserRole.DEVELOPER,
                database_access={db_name},
            )

            # 确保任务表已存在
            await _ensure_task_table()

            for item in result.results:
                try:
                    opt = await auto_optimizer.optimize_query(
                        sql=item.slow_query.sql_statement,
                        connector=slow_query_analyzer.connector,
                        user=user,
                        auto_apply=False,
                    )
                    # 附加精简版优化结果到分析结果中，方便前端直接展示
                    item.optimize_result = {
                        "best_sql": opt.get("best_sql"),
                        "best_reason": opt.get("best_reason"),
                        "improvement": opt.get("improvement"),
                    }

                    # 为该慢查询创建一个优化任务，进入待审核队列（写入 DB）
                    task_id = f"task-{uuid4()}"
                    task = SlowQueryOptimizationTask(
                        task_id=task_id,
                        slow_query=item.slow_query,
                        optimize_result=opt,
                    )
                    await _insert_optimization_task(task)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("optimize_query for slow query failed: %s", exc)
                    continue

        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("analyze_slow_queries failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """自然语言对话入口.

    使用 NLPProcessor 提取意图与实体, 并根据意图路由到 SQL 分析、慢查询/优化或知识查询。
    """

    if nlp_processor is None:
        raise HTTPException(status_code=503, detail="NLP processor not initialised")

    # 1. 提取意图
    try:
        user_intent = await nlp_processor.extract_intent(request.text)
    except Exception as exc:  # noqa: BLE001
        logger.error("extract_intent failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process natural language input") from exc

    reply = ""
    sql_analysis: Optional[SlowQueryAnalysisResult] = None
    slow_batch: Optional[BatchAnalysisResult] = None
    knowledge_entries: Optional[List[Dict[str, Any]]] = None

    intent_type = user_intent.intent_type.value
    params = user_intent.parameters or {}

    # 2. 根据意图类型路由
    try:
        if intent_type == "query_analysis" and slow_query_analyzer is not None:
            sql = params.get("sql_statement") or request.text
            result = await slow_query_analyzer.analyze_single_query(sql)
            sql_analysis = result
            reply = "已为你分析该 SQL 的执行计划和性能特征。"

        elif intent_type == "optimization_request" and slow_query_analyzer is not None:
            # 对应慢查询/优化请求, 先跑一次批量慢查询分析, 再基于已有逻辑生成优化任务
            batch = await slow_query_analyzer.analyze_slow_queries(limit=params.get("limit"))
            slow_batch = batch
            reply = "已执行慢查询分析, 并为相关语句生成优化建议(如可用)。"

        elif intent_type == "knowledge_query" and knowledge_service is not None:
            topic = params.get("query_topic") or request.text
            entries = await knowledge_service.search(topic, limit=5)
            knowledge_entries = [
                {"entry_id": e.entry_id, "title": e.title, "source": e.source}
                for e in entries
            ]
            if knowledge_entries:
                reply = "为你找到了一些相关的知识条目。"
            else:
                reply = "暂时没有找到与问题高度匹配的知识条目。"

        else:
            reply = "当前主要支持 SQL 分析、优化请求和知识问答相关的自然语言指令。"

    except Exception as exc:  # noqa: BLE001
        logger.error("chat routing failed: %s", exc)
        reply = "处理你的请求时发生错误, 请稍后重试。"

    intent_dict: Dict[str, Any] = user_intent.dict()

    return ChatResponse(
        intent=intent_dict,
        reply=reply,
        sql_analysis=sql_analysis,
        slow_queries=slow_batch,
        knowledge_entries=knowledge_entries,
    )


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


@app.get("/optimization_tasks", response_model=List[SlowQueryOptimizationTask])
async def list_optimization_tasks() -> List[SlowQueryOptimizationTask]:
    """List all slow query optimization tasks from the database."""

    await _ensure_task_table()
    return await _list_optimization_tasks()


@app.get("/optimization_tasks/{task_id}", response_model=SlowQueryOptimizationTask)
async def get_optimization_task(task_id: str) -> SlowQueryOptimizationTask:
    """Get details of a specific optimization task."""

    await _ensure_task_table()
    task = await _get_optimization_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/optimization_tasks/{task_id}/approve", response_model=SlowQueryOptimizationTask)
async def approve_optimization_task(task_id: str, request: OptimizationTaskReviewRequest) -> SlowQueryOptimizationTask:
    """Mark an optimization task as approved (no execution yet)."""

    await _ensure_task_table()
    task = await _get_optimization_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == SlowQueryOptimizationTaskStatus.EXECUTED:
        raise HTTPException(status_code=400, detail="Task already executed")

    await _update_optimization_task_status(
        task_id=task_id,
        status=SlowQueryOptimizationTaskStatus.APPROVED,
        reviewer=request.reviewer,
        comment=request.comment,
        executed_at=None,
    )
    updated = await _get_optimization_task(task_id)
    assert updated is not None
    return updated


@app.post("/optimization_tasks/{task_id}/reject", response_model=SlowQueryOptimizationTask)
async def reject_optimization_task(task_id: str, request: OptimizationTaskReviewRequest) -> SlowQueryOptimizationTask:
    """Reject an optimization task."""

    await _ensure_task_table()
    task = await _get_optimization_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == SlowQueryOptimizationTaskStatus.EXECUTED:
        raise HTTPException(status_code=400, detail="Task already executed")

    await _update_optimization_task_status(
        task_id=task_id,
        status=SlowQueryOptimizationTaskStatus.REJECTED,
        reviewer=request.reviewer,
        comment=request.comment,
        executed_at=None,
    )
    updated = await _get_optimization_task(task_id)
    assert updated is not None
    return updated


@app.post("/optimization_tasks/{task_id}/execute", response_model=SlowQueryOptimizationTask)
async def execute_optimization_task(task_id: str, request: OptimizationTaskReviewRequest) -> SlowQueryOptimizationTask:
    """Execute an approved optimization task.

    当前实现仅更新任务状态/元数据，不在数据库中执行真实变更，
    后续可在此处接入 AutoOptimizer 的具体执行能力。
    """

    await _ensure_task_table()
    task = await _get_optimization_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == SlowQueryOptimizationTaskStatus.REJECTED:
        raise HTTPException(status_code=400, detail="Rejected task cannot be executed")

    # 这里接入真实的优化执行与回滚/审计闭环
    if (
        slow_query_analyzer is None
        or slow_query_analyzer.connector is None
        or safety_validator is None
        or auto_optimizer is None
    ):
        raise HTTPException(status_code=503, detail="Analyzer, optimizer or safety validator not initialised")

    connector = slow_query_analyzer.connector
    db_name = slow_query_analyzer.database_config.database

    opt = task.optimize_result
    planned_actions = []
    if isinstance(opt, dict):
        planned_actions = opt.get("planned_actions") or []
    best_sql = opt.get("best_sql") if isinstance(opt, dict) else None

    # 如果未提供结构化 planned_actions，则退化为对 best_sql 的直接执行计划
    if not planned_actions and best_sql:
        planned_actions = [
            {
                "id": "exec-best-sql",
                "type": "execute_sql",
                "sqls": [best_sql],
            }
        ]

    if not planned_actions:
        raise HTTPException(status_code=400, detail="Task has no planned_actions or best_sql to execute")

    reviewer_name = request.reviewer or "reviewer"
    user = User(
        user_id=reviewer_name,
        username=reviewer_name,
        role=UserRole.DEVELOPER,
        database_access={db_name},
    )

    operation_id = f"optimization_task:{task_id}"
    executed_sqls: list[str] = []
    rollback_sqls: list[str] = []
    errors: list[str] = []
    rollback_plan_id: str | None = None
    post_snapshot_id: str | None = None

    # 创建回滚快照和审计日志（如可用）
    pre_snapshot_id: str | None = None
    if rollback_manager is not None:
        try:
            pre_snapshot = await rollback_manager.create_snapshot(
                connector=connector,
                operation_id=operation_id,
                user=user,
                snapshot_type=SnapshotType.FULL_SNAPSHOT,
                objects=None,
            )
            pre_snapshot_id = pre_snapshot.snapshot_id
            await rollback_manager.log_operation(
                operation_id=operation_id,
                operation_type="slow_query_optimization",
                database=db_name,
                user=user,
                sql_statements=[],
                affected_objects=[],
                pre_snapshot_id=pre_snapshot_id,
                session_id=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create snapshot or log operation for task %s: %s", task_id, exc)

    # 执行 planned_actions
    for action in planned_actions:
        a_type = action.get("type")
        try:
            if a_type == "execute_sql":
                sqls = action.get("sqls") or []
                for sql in sqls:
                    validation = await safety_validator.validate_sql_operation(sql, user, db_name)
                    if not validation.is_valid:
                        errors.append(
                            "execute_sql safety validation failed: "
                            + ", ".join(validation.violations)
                        )
                        raise RuntimeError("execute_sql safety validation failed")
                    await connector.execute_query(sql)
                    executed_sqls.append(sql)

            elif a_type == "index_creation":
                spec_dict = action.get("spec") or {}
                index_spec = IndexSpec(**spec_dict)
                res = await auto_optimizer.create_index(connector, index_spec, user)
                executed_sqls.extend(res.executed_sql)
                rollback_sqls.extend(res.rollback_sql)
                if res.status != OptimizationStatus.COMPLETED:
                    errors.append(res.error_message or "index_creation failed")
                    raise RuntimeError("index_creation failed")

            elif a_type == "statistics_update":
                tables = action.get("tables") or []
                res = await auto_optimizer.update_statistics(connector, tables, user)
                executed_sqls.extend(res.executed_sql)
                if res.status != OptimizationStatus.COMPLETED:
                    errors.append(res.error_message or "statistics_update failed")
                    raise RuntimeError("statistics_update failed")

            elif a_type == "config_tuning":
                res = await auto_optimizer.optimize_configuration(connector, user)
                executed_sqls.extend(res.executed_sql)
                rollback_sqls.extend(res.rollback_sql)
                if res.status != OptimizationStatus.COMPLETED:
                    errors.append(res.error_message or "config_tuning failed")
                    raise RuntimeError("config_tuning failed")

            else:
                errors.append(f"unsupported action type: {a_type}")
                raise RuntimeError(f"unsupported action type: {a_type}")

        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            break

    # 执行结束后，如有实际变更且可用 RollbackManager，则生成回滚计划并完成审计
    if rollback_manager is not None and executed_sqls:
        try:
            post_snapshot = await rollback_manager.create_snapshot(
                connector=connector,
                operation_id=operation_id,
                user=user,
                snapshot_type=SnapshotType.FULL_SNAPSHOT,
                objects=None,
            )
            post_snapshot_id = post_snapshot.snapshot_id

            plan = await rollback_manager.create_rollback_plan(
                operation_id=operation_id,
                database=db_name,
                original_sql=executed_sqls,
                strategy=RollbackStrategy.MANUAL,
            )
            rollback_plan_id = plan.plan_id

            await rollback_manager.complete_operation_log(
                operation_id=operation_id,
                success=not errors,
                error_message="; ".join(errors) if errors else None,
                post_snapshot_id=post_snapshot_id,
                rollback_executed=False,
                rollback_plan_id=rollback_plan_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create rollback plan or complete audit log for task %s: %s",
                task_id,
                exc,
            )

    comment_base = request.comment or ""
    extra_info = ""
    if rollback_plan_id:
        extra_info += f" | rollback_plan_id={rollback_plan_id}"
    extra_info += f" | operation_id={operation_id}"

    if errors:
        comment = comment_base + f" | execute failed: {'; '.join(errors)}" + extra_info
        await _update_optimization_task_status(
            task_id=task_id,
            status=SlowQueryOptimizationTaskStatus.REJECTED,
            reviewer=reviewer_name,
            comment=comment,
            executed_at=None,
        )
    else:
        comment = comment_base + " | optimization executed successfully" + extra_info
        await _update_optimization_task_status(
            task_id=task_id,
            status=SlowQueryOptimizationTaskStatus.EXECUTED,
            reviewer=reviewer_name,
            comment=comment,
            executed_at=datetime.utcnow(),
        )

    updated = await _get_optimization_task(task_id)
    assert updated is not None
    return updated
