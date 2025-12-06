"""回滚和恢复管理器 - 提供数据库操作的回滚和恢复功能."""

import json
import logging
import asyncio
import os
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ...database.connector_base import BaseDatabaseConnector
from .safety_validator import SafetyValidator, User, ValidationResult, RiskLevel
from ..models.models import Task, TaskStatus

logger = logging.getLogger(__name__)


class SnapshotType(str, Enum):
    """快照类型枚举."""
    SCHEMA_SNAPSHOT = "schema_snapshot"
    DATA_SNAPSHOT = "data_snapshot"
    INDEX_SNAPSHOT = "index_snapshot"
    CONFIG_SNAPSHOT = "config_snapshot"
    FULL_SNAPSHOT = "full_snapshot"


class RecoveryStatus(str, Enum):
    """恢复状态枚举."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RollbackStrategy(str, Enum):
    """回滚策略枚举."""
    IMMEDIATE = "immediate"  # 立即回滚
    DELAYED = "delayed"      # 延迟回滚
    MANUAL = "manual"        # 手动回滚
    CONDITIONAL = "conditional"  # 条件回滚


class DatabaseSnapshot(BaseModel):
    """数据库快照模型."""
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()), description="快照ID")
    database: str = Field(..., description="数据库名")
    snapshot_type: SnapshotType = Field(..., description="快照类型")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    created_by: str = Field(..., description="创建用户")
    operation_id: str = Field(..., description="关联的操作ID")
    
    # 快照数据
    schema_info: Dict[str, Any] = Field(default_factory=dict, description="模式信息")
    index_info: Dict[str, Any] = Field(default_factory=dict, description="索引信息")
    config_info: Dict[str, Any] = Field(default_factory=dict, description="配置信息")
    data_checksums: Dict[str, str] = Field(default_factory=dict, description="数据校验和")
    
    # 元数据
    size_bytes: int = Field(default=0, description="快照大小(字节)")
    compression_ratio: float = Field(default=1.0, description="压缩比")
    retention_days: int = Field(default=30, description="保留天数")
    tags: List[str] = Field(default_factory=list, description="标签")


class RollbackPlan(BaseModel):
    """回滚计划模型."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()), description="计划ID")
    operation_id: str = Field(..., description="原始操作ID")
    database: str = Field(..., description="数据库名")
    strategy: RollbackStrategy = Field(..., description="回滚策略")
    
    # 回滚步骤
    rollback_steps: List[Dict[str, Any]] = Field(..., description="回滚步骤")
    validation_queries: List[str] = Field(default_factory=list, description="验证查询")
    dependency_checks: List[str] = Field(default_factory=list, description="依赖检查")
    
    # 执行配置
    max_execution_time: int = Field(default=3600, description="最大执行时间(秒)")
    retry_count: int = Field(default=3, description="重试次数")
    rollback_timeout: int = Field(default=1800, description="回滚超时时间(秒)")
    
    # 状态信息
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    executed_at: Optional[datetime] = Field(None, description="执行时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    status: RecoveryStatus = Field(default=RecoveryStatus.PENDING, description="状态")


class RecoveryOperation(BaseModel):
    """恢复操作模型."""
    recovery_id: str = Field(default_factory=lambda: str(uuid4()), description="恢复ID")
    operation_id: str = Field(..., description="原始操作ID")
    failure_reason: str = Field(..., description="失败原因")
    recovery_type: str = Field(..., description="恢复类型")
    
    # 恢复步骤
    recovery_steps: List[Dict[str, Any]] = Field(..., description="恢复步骤")
    rollback_plan_id: Optional[str] = Field(None, description="关联的回滚计划ID")
    
    # 执行信息
    started_at: datetime = Field(default_factory=datetime.now, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    status: RecoveryStatus = Field(default=RecoveryStatus.PENDING, description="状态")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 结果信息
    recovered_objects: List[str] = Field(default_factory=list, description="已恢复对象")
    failed_objects: List[str] = Field(default_factory=list, description="失败对象")
    recovery_log: List[str] = Field(default_factory=list, description="恢复日志")


class OperationAuditLog(BaseModel):
    """操作审计日志模型."""
    log_id: str = Field(default_factory=lambda: str(uuid4()), description="日志ID")
    operation_id: str = Field(..., description="操作ID")
    operation_type: str = Field(..., description="操作类型")
    database: str = Field(..., description="数据库名")
    
    # 操作信息
    user_id: str = Field(..., description="用户ID")
    sql_statements: List[str] = Field(default_factory=list, description="SQL语句")
    affected_objects: List[str] = Field(default_factory=list, description="影响的对象")
    
    # 快照信息
    pre_snapshot_id: Optional[str] = Field(None, description="操作前快照ID")
    post_snapshot_id: Optional[str] = Field(None, description="操作后快照ID")
    
    # 执行信息
    started_at: datetime = Field(default_factory=datetime.now, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    execution_time: float = Field(default=0.0, description="执行时间(秒)")
    
    # 结果信息
    success: bool = Field(default=False, description="是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    rollback_executed: bool = Field(default=False, description="是否执行了回滚")
    rollback_plan_id: Optional[str] = Field(None, description="回滚计划ID")
    
    # 元数据
    session_id: Optional[str] = Field(None, description="会话ID")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")


class DatabaseRollbackStorageBackend:
    """基于业务数据库的回滚状态持久化后端.

    使用单一逻辑表分别存储快照、回滚计划和审计日志的 JSON payload,
    以便在多实例和重启场景下共享状态。当前实现为保守版本, 未在
    RollbackManager 中强制启用, 由外部按需注入并管理生命周期。
    """

    def __init__(self, connector: BaseDatabaseConnector, table_prefix: str = "rollback_") -> None:
        self.connector = connector
        self.table_prefix = table_prefix

    async def _ensure_tables(self) -> None:
        """确保持久化表已创建."""

        db_type = self.connector.database_type.lower()
        snapshots_table = f"{self.table_prefix}snapshots"
        plans_table = f"{self.table_prefix}plans"
        audits_table = f"{self.table_prefix}audits"

        if db_type == "mysql":
            snapshots_sql = f"""
            CREATE TABLE IF NOT EXISTS {snapshots_table} (
              snapshot_id   VARCHAR(64) PRIMARY KEY,
              database_name VARCHAR(255) NOT NULL,
              created_at    DATETIME      NOT NULL,
              payload       LONGTEXT      NOT NULL
            )
            """
            plans_sql = f"""
            CREATE TABLE IF NOT EXISTS {plans_table} (
              plan_id       VARCHAR(64) PRIMARY KEY,
              operation_id  VARCHAR(128) NOT NULL,
              database_name VARCHAR(255) NOT NULL,
              status        VARCHAR(32)  NOT NULL,
              created_at    DATETIME      NOT NULL,
              payload       LONGTEXT      NOT NULL
            )
            """
            audits_sql = f"""
            CREATE TABLE IF NOT EXISTS {audits_table} (
              log_id        VARCHAR(64) PRIMARY KEY,
              operation_id  VARCHAR(128) NOT NULL,
              database_name VARCHAR(255) NOT NULL,
              started_at    DATETIME      NOT NULL,
              completed_at  DATETIME      NULL,
              payload       LONGTEXT      NOT NULL
            )
            """
        elif db_type == "postgresql":
            snapshots_sql = f"""
            CREATE TABLE IF NOT EXISTS {snapshots_table} (
              snapshot_id   VARCHAR(64) PRIMARY KEY,
              database_name VARCHAR(255) NOT NULL,
              created_at    TIMESTAMP    NOT NULL,
              payload       TEXT         NOT NULL
            )
            """
            plans_sql = f"""
            CREATE TABLE IF NOT EXISTS {plans_table} (
              plan_id       VARCHAR(64) PRIMARY KEY,
              operation_id  VARCHAR(128) NOT NULL,
              database_name VARCHAR(255) NOT NULL,
              status        VARCHAR(32)  NOT NULL,
              created_at    TIMESTAMP    NOT NULL,
              payload       TEXT         NOT NULL
            )
            """
            audits_sql = f"""
            CREATE TABLE IF NOT EXISTS {audits_table} (
              log_id        VARCHAR(64) PRIMARY KEY,
              operation_id  VARCHAR(128) NOT NULL,
              database_name VARCHAR(255) NOT NULL,
              started_at    TIMESTAMP    NOT NULL,
              completed_at  TIMESTAMP    NULL,
              payload       TEXT         NOT NULL
            )
            """
        else:
            logger.warning("DatabaseRollbackStorageBackend: unsupported db type %s", db_type)
            return

        try:
            await self.connector.execute_query(snapshots_sql)
            await self.connector.execute_query(plans_sql)
            await self.connector.execute_query(audits_sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to ensure rollback tables: %s", exc)

    async def save_snapshot(self, snapshot: DatabaseSnapshot) -> None:
        await self._ensure_tables()
        table = f"{self.table_prefix}snapshots"
        payload = json.dumps(snapshot.dict(), ensure_ascii=False).replace("'", "''")
        sql = f"""
        INSERT INTO {table} (snapshot_id, database_name, created_at, payload)
        VALUES (
          '{snapshot.snapshot_id}',
          '{snapshot.database}',
          '{snapshot.created_at.isoformat()}',
          '{payload}'
        )
        ON CONFLICT (snapshot_id) DO UPDATE SET
          database_name = EXCLUDED.database_name,
          created_at    = EXCLUDED.created_at,
          payload       = EXCLUDED.payload
        """
        if self.connector.database_type.lower() == "mysql":
            sql = f"""
            REPLACE INTO {table} (snapshot_id, database_name, created_at, payload)
            VALUES (
              '{snapshot.snapshot_id}',
              '{snapshot.database}',
              '{snapshot.created_at.isoformat()}',
              '{payload}'
            )
            """
        try:
            await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist snapshot %s: %s", snapshot.snapshot_id, exc)

    async def save_rollback_plan(self, plan: RollbackPlan) -> None:
        await self._ensure_tables()
        table = f"{self.table_prefix}plans"
        payload = json.dumps(plan.dict(), ensure_ascii=False).replace("'", "''")
        sql = f"""
        INSERT INTO {table} (plan_id, operation_id, database_name, status, created_at, payload)
        VALUES (
          '{plan.plan_id}',
          '{plan.operation_id}',
          '{plan.database}',
          '{plan.status.value}',
          '{plan.created_at.isoformat()}',
          '{payload}'
        )
        ON CONFLICT (plan_id) DO UPDATE SET
          operation_id  = EXCLUDED.operation_id,
          database_name = EXCLUDED.database_name,
          status        = EXCLUDED.status,
          created_at    = EXCLUDED.created_at,
          payload       = EXCLUDED.payload
        """
        if self.connector.database_type.lower() == "mysql":
            sql = f"""
            REPLACE INTO {table} (plan_id, operation_id, database_name, status, created_at, payload)
            VALUES (
              '{plan.plan_id}',
              '{plan.operation_id}',
              '{plan.database}',
              '{plan.status.value}',
              '{plan.created_at.isoformat()}',
              '{payload}'
            )
            """
        try:
            await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist rollback plan %s: %s", plan.plan_id, exc)

    async def save_audit_log(self, log: OperationAuditLog) -> None:
        await self._ensure_tables()
        table = f"{self.table_prefix}audits"
        payload = json.dumps(log.dict(), ensure_ascii=False).replace("'", "''")
        sql = f"""
        INSERT INTO {table} (log_id, operation_id, database_name, started_at, completed_at, payload)
        VALUES (
          '{log.log_id}',
          '{log.operation_id}',
          '{log.database}',
          '{log.started_at.isoformat()}',
          {f"'{log.completed_at.isoformat()}'" if log.completed_at else 'NULL'},
          '{payload}'
        )
        ON CONFLICT (log_id) DO UPDATE SET
          operation_id  = EXCLUDED.operation_id,
          database_name = EXCLUDED.database_name,
          started_at    = EXCLUDED.started_at,
          completed_at  = EXCLUDED.completed_at,
          payload       = EXCLUDED.payload
        """
        if self.connector.database_type.lower() == "mysql":
            sql = f"""
            REPLACE INTO {table} (log_id, operation_id, database_name, started_at, completed_at, payload)
            VALUES (
              '{log.log_id}',
              '{log.operation_id}',
              '{log.database}',
              '{log.started_at.isoformat()}',
              {f"'{log.completed_at.isoformat()}'" if log.completed_at else 'NULL'},
              '{payload}'
            )
            """
        try:
            await self.connector.execute_query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist audit log %s: %s", log.log_id, exc)


class RollbackManager:
    """回滚和恢复管理器."""
    
    def __init__(self, safety_validator: SafetyValidator, storage_backend: DatabaseRollbackStorageBackend | None = None):
        """初始化回滚管理器."""
        self.safety_validator = safety_validator
        self.storage_backend: DatabaseRollbackStorageBackend | None = storage_backend
        
        # 存储
        self.snapshots: Dict[str, DatabaseSnapshot] = {}
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.recovery_operations: Dict[str, RecoveryOperation] = {}
        self.audit_logs: Dict[str, OperationAuditLog] = {}
        
        # 配置
        self.max_snapshots_per_database = 100
        self.snapshot_retention_days = 30
        self.auto_cleanup_enabled = True
        # 持久化配置（默认关闭，避免对外部环境产生副作用，如需启用可在外部设置为 True）
        self.persistence_enabled = False
        self.persistence_dir = ".rollback_state"
        # 尝试加载已有持久化状态
        self._load_persisted_state()
        
        # 初始化清理任务
        self._init_cleanup_tasks()
    
    def _init_cleanup_tasks(self):
        """初始化清理任务."""
        if self.auto_cleanup_enabled:
            # 这里可以添加定期清理过期快照的任务
            pass

    # ---- 持久化相关辅助方法（可选启用） ----

    def _load_persisted_state(self) -> None:
        """从本地JSON文件加载持久化状态.

        默认仅在 self.persistence_enabled 为 True 时生效，避免对外部环境产生副作用。
        """
        if not self.persistence_enabled:
            return

        try:
            if not os.path.isdir(self.persistence_dir):
                return

            def _load_json(filename: str) -> Any:
                path = os.path.join(self.persistence_dir, filename)
                if not os.path.exists(path):
                    return None
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)

            snapshots_data = _load_json("snapshots.json") or {}
            plans_data = _load_json("rollback_plans.json") or {}
            recoveries_data = _load_json("recovery_operations.json") or {}
            audits_data = _load_json("audit_logs.json") or {}

            self.snapshots = {
                sid: DatabaseSnapshot(**data) for sid, data in snapshots_data.items()
            }
            self.rollback_plans = {
                pid: RollbackPlan(**data) for pid, data in plans_data.items()
            }
            self.recovery_operations = {
                rid: RecoveryOperation(**data) for rid, data in recoveries_data.items()
            }
            self.audit_logs = {
                lid: OperationAuditLog(**data) for lid, data in audits_data.items()
            }
        except Exception as e:  # noqa: BLE001
            # 持久化加载失败不影响主流程
            logger.warning(f"加载回滚管理持久化状态失败: {e}")

    def _persist_state(self) -> None:
        """将当前状态持久化到本地JSON文件.

        仅在 self.persistence_enabled 为 True 时执行。
        """
        if not self.persistence_enabled:
            return

        try:
            os.makedirs(self.persistence_dir, exist_ok=True)

            def _dump_json(filename: str, data: Any) -> None:
                path = os.path.join(self.persistence_dir, filename)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, default=str, ensure_ascii=False, indent=2)

            _dump_json("snapshots.json", {k: v.dict() for k, v in self.snapshots.items()})
            _dump_json("rollback_plans.json", {k: v.dict() for k, v in self.rollback_plans.items()})
            _dump_json("recovery_operations.json", {k: v.dict() for k, v in self.recovery_operations.items()})
            _dump_json("audit_logs.json", {k: v.dict() for k, v in self.audit_logs.items()})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"持久化回滚管理状态失败: {e}")
    
    async def create_snapshot(self,
                            connector: BaseDatabaseConnector,
                            operation_id: str,
                            user: User,
                            snapshot_type: SnapshotType = SnapshotType.FULL_SNAPSHOT,
                            objects: List[str] = None) -> DatabaseSnapshot:
        """
        创建数据库快照.
        
        Args:
            connector: 数据库连接器
            operation_id: 操作ID
            user: 用户信息
            snapshot_type: 快照类型
            objects: 要快照的对象列表
            
        Returns:
            DatabaseSnapshot: 数据库快照
        """
        logger.info(f"开始创建数据库快照，操作ID: {operation_id}, 类型: {snapshot_type}")
        
        snapshot = DatabaseSnapshot(
            database=connector.config.database,
            snapshot_type=snapshot_type,
            created_by=user.user_id,
            operation_id=operation_id
        )
        
        try:
            # 1. 创建模式快照
            if snapshot_type in [SnapshotType.SCHEMA_SNAPSHOT, SnapshotType.FULL_SNAPSHOT]:
                snapshot.schema_info = await self._create_schema_snapshot(connector, objects)
            
            # 2. 创建索引快照
            if snapshot_type in [SnapshotType.INDEX_SNAPSHOT, SnapshotType.FULL_SNAPSHOT]:
                snapshot.index_info = await self._create_index_snapshot(connector, objects)
            
            # 3. 创建配置快照
            if snapshot_type in [SnapshotType.CONFIG_SNAPSHOT, SnapshotType.FULL_SNAPSHOT]:
                snapshot.config_info = await self._create_config_snapshot(connector)
            
            # 4. 创建数据快照（校验和）
            if snapshot_type in [SnapshotType.DATA_SNAPSHOT, SnapshotType.FULL_SNAPSHOT]:
                snapshot.data_checksums = await self._create_data_checksums(connector, objects)
            
            # 5. 计算快照大小
            snapshot.size_bytes = self._calculate_snapshot_size(snapshot)
            
            # 6. 存储快照
            self.snapshots[snapshot.snapshot_id] = snapshot
            self._persist_state()

            # 6.1 可选: 持久化到业务数据库
            if self.storage_backend is not None:
                try:
                    await self.storage_backend.save_snapshot(snapshot)
                except Exception as e:  # noqa: BLE001
                    logger.warning("持久化快照到数据库失败: %s", e)
            
            # 7. 清理过期快照
            await self._cleanup_old_snapshots(connector.config.database)
            
            logger.info(f"数据库快照创建成功，快照ID: {snapshot.snapshot_id}")
            
        except Exception as e:
            logger.error(f"创建数据库快照失败，操作ID: {operation_id}, 错误: {e}")
            raise
        
        return snapshot
    
    async def create_rollback_plan(self,
                                 operation_id: str,
                                 database: str,
                                 original_sql: List[str],
                                 strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE) -> RollbackPlan:
        """
        创建回滚计划.
        
        Args:
            operation_id: 操作ID
            database: 数据库名
            original_sql: 原始SQL语句列表
            strategy: 回滚策略
            
        Returns:
            RollbackPlan: 回滚计划
        """
        logger.info(f"开始创建回滚计划，操作ID: {operation_id}")
        
        try:
            # 1. 分析原始SQL生成回滚步骤
            rollback_steps = await self._generate_rollback_steps(original_sql, database)
            
            # 2. 生成验证查询
            validation_queries = self._generate_validation_queries(original_sql, database)
            
            # 3. 生成依赖检查
            dependency_checks = self._generate_dependency_checks(original_sql, database)
            
            # 4. 创建回滚计划
            plan = RollbackPlan(
                operation_id=operation_id,
                database=database,
                strategy=strategy,
                rollback_steps=rollback_steps,
                validation_queries=validation_queries,
                dependency_checks=dependency_checks
            )
            
            # 5. 存储回滚计划
            self.rollback_plans[plan.plan_id] = plan
            self._persist_state()

            # 5.1 可选: 持久化到业务数据库
            if self.storage_backend is not None:
                try:
                    await self.storage_backend.save_rollback_plan(plan)
                except Exception as e:  # noqa: BLE001
                    logger.warning("持久化回滚计划到数据库失败: %s", e)
            
            logger.info(f"回滚计划创建成功，计划ID: {plan.plan_id}")
            
        except Exception as e:
            logger.error(f"创建回滚计划失败，操作ID: {operation_id}, 错误: {e}")
            raise
        
        return plan
    
    async def execute_rollback(self,
                             plan_id: str,
                             connector: BaseDatabaseConnector,
                             user: User,
                             force: bool = False) -> bool:
        """
        执行回滚操作.
        
        Args:
            plan_id: 回滚计划ID
            connector: 数据库连接器
            user: 用户信息
            force: 是否强制执行
            
        Returns:
            bool: 回滚是否成功
        """
        logger.info(f"开始执行回滚操作，计划ID: {plan_id}")
        
        plan = self.rollback_plans.get(plan_id)
        if not plan:
            logger.error(f"未找到回滚计划: {plan_id}")
            return False
        
        plan.status = RecoveryStatus.IN_PROGRESS
        plan.executed_at = datetime.now()
        
        try:
            # 1. 安全验证
            if not force:
                validation_result = await self._validate_rollback_execution(plan, user)
                if not validation_result.is_valid:
                    logger.error(f"回滚操作安全验证失败: {validation_result.violations}")
                    plan.status = RecoveryStatus.FAILED
                    return False
            
            # 2. 执行依赖检查
            dependency_check_passed = await self._execute_dependency_checks(plan, connector)
            if not dependency_check_passed:
                logger.error(f"回滚操作依赖检查失败")
                plan.status = RecoveryStatus.FAILED
                return False
            
            # 3. 执行回滚步骤
            success = await self._execute_rollback_steps(plan, connector, user)
            
            if success:
                # 4. 执行验证查询
                validation_passed = await self._execute_validation_queries(plan, connector)
                if validation_passed:
                    plan.status = RecoveryStatus.COMPLETED
                    plan.completed_at = datetime.now()
                    logger.info(f"回滚操作执行成功，计划ID: {plan_id}")
                else:
                    plan.status = RecoveryStatus.PARTIAL
                    logger.warning(f"回滚操作部分成功，计划ID: {plan_id}")
            else:
                plan.status = RecoveryStatus.FAILED
                logger.error(f"回滚操作执行失败，计划ID: {plan_id}")
            
            return success
            
        except Exception as e:
            plan.status = RecoveryStatus.FAILED
            plan.completed_at = datetime.now()
            logger.error(f"回滚操作执行异常，计划ID: {plan_id}, 错误: {e}")
            return False
    
    async def create_recovery_operation(self,
                                      operation_id: str,
                                      failure_reason: str,
                                      recovery_type: str = "automatic") -> RecoveryOperation:
        """
        创建恢复操作.
        
        Args:
            operation_id: 原始操作ID
            failure_reason: 失败原因
            recovery_type: 恢复类型
            
        Returns:
            RecoveryOperation: 恢复操作
        """
        logger.info(f"开始创建恢复操作，操作ID: {operation_id}")
        
        try:
            # 1. 分析失败原因生成恢复步骤
            recovery_steps = await self._generate_recovery_steps(
                operation_id, failure_reason
            )
            
            # 2. 创建恢复操作
            recovery = RecoveryOperation(
                operation_id=operation_id,
                failure_reason=failure_reason,
                recovery_type=recovery_type,
                recovery_steps=recovery_steps
            )
            
            # 3. 查找相关的回滚计划
            rollback_plan = self._find_rollback_plan_by_operation(operation_id)
            if rollback_plan:
                recovery.rollback_plan_id = rollback_plan.plan_id
            
            # 4. 存储恢复操作
            self.recovery_operations[recovery.recovery_id] = recovery
            self._persist_state()
            
            logger.info(f"恢复操作创建成功，恢复ID: {recovery.recovery_id}")
            
        except Exception as e:
            logger.error(f"创建恢复操作失败，操作ID: {operation_id}, 错误: {e}")
            raise
        
        return recovery
    
    async def execute_recovery(self,
                             recovery_id: str,
                             connector: BaseDatabaseConnector,
                             user: User) -> bool:
        """
        执行恢复操作.
        
        Args:
            recovery_id: 恢复操作ID
            connector: 数据库连接器
            user: 用户信息
            
        Returns:
            bool: 恢复是否成功
        """
        logger.info(f"开始执行恢复操作，恢复ID: {recovery_id}")
        
        recovery = self.recovery_operations.get(recovery_id)
        if not recovery:
            logger.error(f"未找到恢复操作: {recovery_id}")
            return False
        
        recovery.status = RecoveryStatus.IN_PROGRESS
        
        try:
            # 1. 执行恢复步骤
            for i, step in enumerate(recovery.recovery_steps):
                step_success = await self._execute_recovery_step(step, connector, user)
                
                if step_success:
                    recovery.recovered_objects.append(step.get("object", f"step_{i}"))
                    recovery.recovery_log.append(f"步骤 {i+1} 执行成功: {step.get('description', '')}")
                else:
                    recovery.failed_objects.append(step.get("object", f"step_{i}"))
                    recovery.recovery_log.append(f"步骤 {i+1} 执行失败: {step.get('description', '')}")
                    
                    # 根据策略决定是否继续
                    if step.get("critical", False):
                        recovery.status = RecoveryStatus.FAILED
                        recovery.completed_at = datetime.now()
                        recovery.error_message = f"关键步骤 {i+1} 执行失败"
                        return False
            
            # 2. 如果有关联的回滚计划，执行回滚
            if recovery.rollback_plan_id:
                rollback_success = await self.execute_rollback(
                    recovery.rollback_plan_id, connector, user
                )
                if not rollback_success:
                    recovery.recovery_log.append("回滚操作执行失败")
            
            # 3. 确定最终状态
            if recovery.failed_objects:
                recovery.status = RecoveryStatus.PARTIAL
            else:
                recovery.status = RecoveryStatus.COMPLETED
            
            recovery.completed_at = datetime.now()
            logger.info(f"恢复操作执行完成，恢复ID: {recovery_id}, 状态: {recovery.status}")
            
            return recovery.status in [RecoveryStatus.COMPLETED, RecoveryStatus.PARTIAL]
            
        except Exception as e:
            recovery.status = RecoveryStatus.FAILED
            recovery.completed_at = datetime.now()
            recovery.error_message = str(e)
            logger.error(f"恢复操作执行异常，恢复ID: {recovery_id}, 错误: {e}")
            return False
    
    async def log_operation(self,
                          operation_id: str,
                          operation_type: str,
                          database: str,
                          user: User,
                          sql_statements: List[str],
                          affected_objects: List[str] = None,
                          pre_snapshot_id: str = None,
                          session_id: str = None) -> OperationAuditLog:
        """
        记录操作审计日志.
        
        Args:
            operation_id: 操作ID
            operation_type: 操作类型
            database: 数据库名
            user: 用户信息
            sql_statements: SQL语句列表
            affected_objects: 影响的对象列表
            pre_snapshot_id: 操作前快照ID
            session_id: 会话ID
            
        Returns:
            OperationAuditLog: 操作审计日志
        """
        audit_log = OperationAuditLog(
            operation_id=operation_id,
            operation_type=operation_type,
            database=database,
            user_id=user.user_id,
            sql_statements=sql_statements,
            affected_objects=affected_objects or [],
            pre_snapshot_id=pre_snapshot_id,
            session_id=session_id
        )
        
        self.audit_logs[audit_log.log_id] = audit_log
        self._persist_state()

        # 可选: 持久化到业务数据库
        if self.storage_backend is not None:
            try:
                await self.storage_backend.save_audit_log(audit_log)
            except Exception as e:  # noqa: BLE001
                logger.warning("持久化审计日志到数据库失败: %s", e)
        
        logger.info(f"记录操作审计日志，日志ID: {audit_log.log_id}")
        
        return audit_log
    
    async def complete_operation_log(self,
                                   operation_id: str,
                                   success: bool,
                                   error_message: str = None,
                                   post_snapshot_id: str = None,
                                   rollback_executed: bool = False,
                                   rollback_plan_id: str = None):
        """
        完成操作审计日志.
        
        Args:
            operation_id: 操作ID
            success: 是否成功
            error_message: 错误信息
            post_snapshot_id: 操作后快照ID
            rollback_executed: 是否执行了回滚
            rollback_plan_id: 回滚计划ID
        """
        # 查找对应的审计日志
        audit_log = None
        for log in self.audit_logs.values():
            if log.operation_id == operation_id:
                audit_log = log
                break
        
        if audit_log:
            audit_log.completed_at = datetime.now()
            audit_log.execution_time = (
                audit_log.completed_at - audit_log.started_at
            ).total_seconds()
            audit_log.success = success
            audit_log.error_message = error_message
            audit_log.post_snapshot_id = post_snapshot_id
            audit_log.rollback_executed = rollback_executed
            audit_log.rollback_plan_id = rollback_plan_id
            
            logger.info(f"完成操作审计日志，操作ID: {operation_id}, 成功: {success}")
            self._persist_state()

            # 可选: 更新持久化记录
            if self.storage_backend is not None:
                try:
                    await self.storage_backend.save_audit_log(audit_log)
                except Exception as e:  # noqa: BLE001
                    logger.warning("更新持久化审计日志失败: %s", e)
    
    # 私有辅助方法
    
    async def _create_schema_snapshot(self, connector: BaseDatabaseConnector, objects: List[str] = None) -> Dict[str, Any]:
        """创建模式快照."""
        schema_info = {}
        
        try:
            if connector.database_type.lower() == "mysql":
                # 获取表结构
                tables_sql = """
                SELECT table_name, table_type, engine, table_comment
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                """
                if objects:
                    obj_list = "','".join(objects)
                    tables_sql += f" AND table_name IN ('{obj_list}')"
                
                tables = await connector.execute_query(tables_sql)
                schema_info["tables"] = tables
                
                # 获取列信息
                for table in tables:
                    columns_sql = f"""
                    SELECT column_name, data_type, is_nullable, column_default, column_comment
                    FROM information_schema.columns
                    WHERE table_schema = DATABASE() AND table_name = '{table['table_name']}'
                    ORDER BY ordinal_position
                    """
                    columns = await connector.execute_query(columns_sql)
                    schema_info[f"columns_{table['table_name']}"] = columns
            
            elif connector.database_type.lower() == "postgresql":
                # PostgreSQL的模式快照逻辑: 记录表和列基础信息
                tables_sql = """
                SELECT tablename as table_name, schemaname as schema_name
                FROM pg_tables
                WHERE schemaname = 'public'
                """
                if objects:
                    obj_list = "','".join(objects)
                    tables_sql += f" AND tablename IN ('{obj_list}')"
                
                tables = await connector.execute_query(tables_sql)
                schema_info["tables"] = tables

                # 获取列信息
                for table in tables:
                    tname = table["table_name"]
                    columns_sql = f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = '{tname}'
                    ORDER BY ordinal_position
                    """
                    columns = await connector.execute_query(columns_sql)
                    schema_info[f"columns_{tname}"] = columns
        
        except Exception as e:
            logger.error(f"创建模式快照失败: {e}")
            schema_info["error"] = str(e)
        
        return schema_info
    
    async def _create_index_snapshot(self, connector: BaseDatabaseConnector, objects: List[str] = None) -> Dict[str, Any]:
        """创建索引快照."""
        index_info = {}
        
        try:
            if connector.database_type.lower() == "mysql":
                indexes_sql = """
                SELECT table_name, index_name, column_name, index_type, non_unique
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                """
                if objects:
                    obj_list = "','".join(objects)
                    indexes_sql += f" AND table_name IN ('{obj_list}')"
                indexes_sql += " ORDER BY table_name, index_name, seq_in_index"
                
                indexes = await connector.execute_query(indexes_sql)
                index_info["indexes"] = indexes
            
            elif connector.database_type.lower() == "postgresql":
                indexes_sql = """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                """
                if objects:
                    obj_list = "','".join(objects)
                    indexes_sql += f" AND tablename IN ('{obj_list}')"
                
                indexes = await connector.execute_query(indexes_sql)
                index_info["indexes"] = indexes
        
        except Exception as e:
            logger.error(f"创建索引快照失败: {e}")
            index_info["error"] = str(e)
        
        return index_info
    
    async def _create_config_snapshot(self, connector: BaseDatabaseConnector) -> Dict[str, Any]:
        """创建配置快照."""
        config_info = {}
        
        try:
            if connector.database_type.lower() == "mysql":
                config_sql = "SHOW VARIABLES"
                variables = await connector.execute_query(config_sql)
                config_info["variables"] = {var["Variable_name"]: var["Value"] for var in variables}
            
            elif connector.database_type.lower() == "postgresql":
                config_sql = "SELECT name, setting, unit FROM pg_settings"
                settings = await connector.execute_query(config_sql)
                config_info["settings"] = {
                    setting["name"]: {
                        "value": setting["setting"],
                        "unit": setting["unit"]
                    } for setting in settings
                }
        
        except Exception as e:
            logger.error(f"创建配置快照失败: {e}")
            config_info["error"] = str(e)
        
        return config_info
    
    async def _create_data_checksums(self, connector: BaseDatabaseConnector, objects: List[str] = None) -> Dict[str, str]:
        """创建数据校验和."""
        checksums = {}
        
        try:
            # 获取表列表
            if connector.database_type.lower() == "mysql":
                tables_sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
            elif connector.database_type.lower() == "postgresql":
                tables_sql = "SELECT tablename as table_name FROM pg_tables WHERE schemaname = 'public'"
            else:
                return checksums
            
            if objects:
                obj_list = "','".join(objects)
                tables_sql += f" AND table_name IN ('{obj_list}')"
            
            tables = await connector.execute_query(tables_sql)
            
            # 为每个表计算校验和
            for table in tables:
                table_name = table["table_name"]
                try:
                    if connector.database_type.lower() == "mysql":
                        checksum_sql = f"CHECKSUM TABLE {table_name}"
                        result = await connector.execute_query(checksum_sql)
                        if result:
                            checksums[table_name] = str(result[0].get("Checksum", ""))
                    elif connector.database_type.lower() == "postgresql":
                        # PostgreSQL没有内置的CHECKSUM，使用行数作为简单校验
                        count_sql = f"SELECT COUNT(*) as row_count FROM {table_name}"
                        result = await connector.execute_query(count_sql)
                        if result:
                            checksums[table_name] = str(result[0]["row_count"])
                except Exception as e:
                    logger.warning(f"计算表 {table_name} 校验和失败: {e}")
                    checksums[table_name] = "error"
        
        except Exception as e:
            logger.error(f"创建数据校验和失败: {e}")
        
        return checksums
    
    def _calculate_snapshot_size(self, snapshot: DatabaseSnapshot) -> int:
        """计算快照大小."""
        try:
            # 简单估算：将所有数据序列化为JSON并计算大小
            data = {
                "schema_info": snapshot.schema_info,
                "index_info": snapshot.index_info,
                "config_info": snapshot.config_info,
                "data_checksums": snapshot.data_checksums
            }
            return len(json.dumps(data, default=str))
        except Exception:
            return 0
    
    async def _cleanup_old_snapshots(self, database: str):
        """清理过期快照."""
        if not self.auto_cleanup_enabled:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.snapshot_retention_days)
        
        # 找到过期的快照
        expired_snapshots = []
        for snapshot_id, snapshot in self.snapshots.items():
            if (snapshot.database == database and 
                snapshot.created_at < cutoff_date):
                expired_snapshots.append(snapshot_id)
        
        # 删除过期快照
        for snapshot_id in expired_snapshots:
            del self.snapshots[snapshot_id]
            logger.info(f"删除过期快照: {snapshot_id}")
        
        # 如果快照数量仍然超过限制，删除最旧的快照
        db_snapshots = [
            (sid, s) for sid, s in self.snapshots.items() 
            if s.database == database
        ]
        
        if len(db_snapshots) > self.max_snapshots_per_database:
            # 按创建时间排序，删除最旧的
            db_snapshots.sort(key=lambda x: x[1].created_at)
            excess_count = len(db_snapshots) - self.max_snapshots_per_database
            
            for i in range(excess_count):
                snapshot_id = db_snapshots[i][0]
                del self.snapshots[snapshot_id]
                logger.info(f"删除超量快照: {snapshot_id}")
        # 快照可能被删除, 持久化当前状态
        self._persist_state()
    
    async def _generate_rollback_steps(self, original_sql: List[str], database: str) -> List[Dict[str, Any]]:
        """生成回滚步骤."""
        rollback_steps = []
        
        for sql in original_sql:
            sql_lower = sql.lower().strip()
            
            if sql_lower.startswith("create index"):
                # 创建索引的回滚是删除索引
                index_match = re.search(r"create\s+(?:unique\s+)?index\s+(\w+)", sql_lower)
                if index_match:
                    index_name = index_match.group(1)
                    rollback_steps.append({
                        "type": "drop_index",
                        "sql": f"DROP INDEX {index_name}",
                        "description": f"删除索引 {index_name}",
                        "order": len(rollback_steps) + 1
                    })
            
            elif sql_lower.startswith("drop index"):
                # 删除索引的回滚需要重新创建索引（需要从快照中获取）
                rollback_steps.append({
                    "type": "recreate_index",
                    "sql": "-- 需要从快照中恢复索引定义",
                    "description": "重新创建被删除的索引",
                    "order": len(rollback_steps) + 1,
                    "requires_snapshot": True
                })
            
            elif sql_lower.startswith("alter table"):
                # ALTER TABLE的回滚比较复杂，需要具体分析
                rollback_steps.append({
                    "type": "reverse_alter",
                    "sql": "-- 需要分析ALTER语句生成反向操作",
                    "description": "反向ALTER TABLE操作",
                    "order": len(rollback_steps) + 1,
                    "requires_analysis": True,
                    "original_sql": sql
                })
            
            elif sql_lower.startswith("update"):
                # UPDATE的回滚需要从快照中恢复数据
                rollback_steps.append({
                    "type": "restore_data",
                    "sql": "-- 需要从快照中恢复更新前的数据",
                    "description": "恢复UPDATE操作前的数据",
                    "order": len(rollback_steps) + 1,
                    "requires_snapshot": True
                })
            
            elif sql_lower.startswith("delete"):
                # DELETE的回滚需要重新插入数据
                rollback_steps.append({
                    "type": "restore_deleted_data",
                    "sql": "-- 需要从快照中恢复被删除的数据",
                    "description": "恢复DELETE操作删除的数据",
                    "order": len(rollback_steps) + 1,
                    "requires_snapshot": True
                })
        
        # 反转步骤顺序（后执行的操作先回滚）
        rollback_steps.reverse()
        for i, step in enumerate(rollback_steps):
            step["order"] = i + 1
        
        return rollback_steps
    
    def _generate_validation_queries(self, original_sql: List[str], database: str) -> List[str]:
        """生成验证查询."""
        validation_queries = []
        
        for sql in original_sql:
            sql_lower = sql.lower().strip()
            
            if sql_lower.startswith("create index"):
                # 验证索引是否被删除
                index_match = re.search(r"create\\s+(?:unique\\s+)?index\\s+(\\w+)", sql_lower)
                if index_match:
                    index_name = index_match.group(1)
                    validation_queries.append(
                        f"SELECT COUNT(*) as count FROM information_schema.statistics "
                        f"WHERE index_name = '{index_name}' AND table_schema = '{database}' /* expect_zero */"
                    )
            
            elif sql_lower.startswith("drop index"):
                # 验证索引是否被重新创建
                index_match = re.search(r"drop\\s+index\\s+(\\w+)", sql_lower)
                if index_match:
                    index_name = index_match.group(1)
                    validation_queries.append(
                        f"SELECT COUNT(*) as count FROM information_schema.statistics "
                        f"WHERE index_name = '{index_name}' AND table_schema = '{database}' /* expect_positive */"
                    )
        
        return validation_queries
    
    def _generate_dependency_checks(self, original_sql: List[str], database: str) -> List[str]:
        """生成依赖检查."""
        dependency_checks = []
        
        # 检查数据库连接
        dependency_checks.append("SELECT 1 as connection_test")
        
        # 检查数据库是否存在
        dependency_checks.append(f"SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '{database}'")
        
        return dependency_checks
    
    async def _validate_rollback_execution(self, plan: RollbackPlan, user: User) -> ValidationResult:
        """验证回滚执行的安全性."""
        violations = []
        warnings = []
        
        # 检查用户权限
        if not self.safety_validator._can_execute_optimization(user):
            violations.append(f"用户 {user.username} 没有执行回滚操作的权限")
        
        # 检查回滚计划的完整性
        if not plan.rollback_steps:
            violations.append("回滚计划缺少具体步骤")
        
        if not plan.validation_queries:
            warnings.append("回滚计划缺少验证查询")
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            risk_level=RiskLevel.MEDIUM if violations else RiskLevel.LOW,
            violations=violations,
            warnings=warnings,
            recommendations=["建议在非高峰时段执行回滚操作"]
        )
    
    async def _execute_dependency_checks(self, plan: RollbackPlan, connector: BaseDatabaseConnector) -> bool:
        """执行依赖检查."""
        try:
            for check_sql in plan.dependency_checks:
                result = await connector.execute_query(check_sql)
                if not result:
                    logger.error(f"依赖检查失败: {check_sql}")
                    return False
            return True
        except Exception as e:
            logger.error(f"执行依赖检查异常: {e}")
            return False
    
    async def _execute_rollback_steps(self, plan: RollbackPlan, connector: BaseDatabaseConnector, user: User) -> bool:
        """执行回滚步骤."""
        try:
            for step in plan.rollback_steps:
                if step.get("requires_snapshot") or step.get("requires_analysis"):
                    # 需要特殊处理的步骤
                    success = await self._execute_special_rollback_step(step, plan, connector)
                else:
                    # 直接执行SQL的步骤
                    sql = step.get("sql", "")
                    if sql and not sql.startswith("--"):
                        await connector.execute_query(sql)
                        success = True
                    else:
                        logger.warning(f"跳过无效的回滚步骤: {step}")
                        success = True
                
                if not success:
                    logger.error(f"回滚步骤执行失败: {step}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"执行回滚步骤异常: {e}")
            return False
    
    async def _execute_special_rollback_step(self, step: Dict[str, Any], plan: RollbackPlan, connector: BaseDatabaseConnector) -> bool:
        """执行特殊的回滚步骤."""
        try:
            if step["type"] == "recreate_index":
                # 从快照中恢复索引
                return await self._restore_index_from_snapshot(plan.operation_id, connector)
            elif step["type"] == "restore_data":
                # 从快照中恢复数据
                return await self._restore_data_from_snapshot(plan.operation_id, connector)
            elif step["type"] == "reverse_alter":
                # 分析并反向ALTER操作
                return await self._reverse_alter_operation(step, connector)
            else:
                logger.warning(f"未知的特殊回滚步骤类型: {step['type']}")
                return True
        except Exception as e:
            logger.error(f"执行特殊回滚步骤失败: {e}")
            return False
    
    async def _restore_index_from_snapshot(self, operation_id: str, connector: BaseDatabaseConnector) -> bool:
        """从快照中恢复索引."""
        # 查找相关快照
        snapshot = None
        for s in self.snapshots.values():
            if s.operation_id == operation_id:
                snapshot = s
                break
        
        if not snapshot or not snapshot.index_info:
            logger.error(f"未找到操作 {operation_id} 的索引快照")
            return False
        
        try:
            # 从快照中重建索引
            indexes = snapshot.index_info.get("indexes", [])

            db_type = connector.database_type.lower()

            if db_type == "mysql":
                # information_schema.statistics 字段: table_name, index_name, column_name, index_type, non_unique
                # 这里按 (table_name, index_name) 分组, 利用查询时的 ORDER BY seq_in_index 保证列顺序
                grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for idx in indexes:
                    table_name = idx.get("table_name") or idx.get("TABLE_NAME")
                    index_name = idx.get("index_name") or idx.get("INDEX_NAME")
                    column_name = idx.get("column_name") or idx.get("COLUMN_NAME")
                    index_type = idx.get("index_type") or idx.get("INDEX_TYPE") or "BTREE"
                    non_unique = idx.get("non_unique") if "non_unique" in idx else idx.get("NON_UNIQUE")

                    if not table_name or not index_name or not column_name:
                        continue

                    # 跳过主键索引 (PRIMARY 由表定义维护, 不在这里重建)
                    if index_name.upper() == "PRIMARY":
                        continue

                    key = (table_name, index_name)
                    if key not in grouped:
                        grouped[key] = {
                            "table_name": table_name,
                            "index_name": index_name,
                            "index_type": index_type,
                            "non_unique": non_unique,
                            "columns": [],
                        }
                    grouped[key]["columns"].append(column_name)

                # 生成并执行 CREATE INDEX 语句
                for (table_name, index_name), info in grouped.items():
                    columns = info["columns"]
                    if not columns:
                        continue

                    # non_unique 为 0 表示唯一索引
                    is_unique = str(info.get("non_unique", "1")) == "0"
                    index_type = (info.get("index_type") or "BTREE").upper()

                    col_list = ", ".join(f"`{col}`" for col in columns)
                    sql = "CREATE"
                    if is_unique:
                        sql += " UNIQUE"
                    sql += f" INDEX `{index_name}` ON `{table_name}` ({col_list})"
                    if index_type and index_type != "BTREE":
                        sql += f" USING {index_type}"

                    logger.info(f"从快照重建 MySQL 索引: {sql}")
                    await connector.execute_query(sql)

            elif db_type == "postgresql":
                # PostgreSQL 直接使用快照中的 indexdef 重建索引
                for index in indexes:
                    indexdef = index.get("indexdef") or index.get("INDEXDEF")
                    if indexdef:
                        logger.info(f"从快照重建 PostgreSQL 索引: {indexdef}")
                        await connector.execute_query(indexdef)
            
            return True
        except Exception as e:
            logger.error(f"从快照恢复索引失败: {e}")
            return False
    
    async def _restore_data_from_snapshot(self, operation_id: str, connector: BaseDatabaseConnector) -> bool:
        """从快照中恢复数据."""
        # 注意: 当前快照仅存储表级校验和/行数信息(data_checksums), 并不包含原始数据内容,
        # 因此这里实现的是基于快照的"一致性校验"而非真正的数据回滚。

        # 查找相关快照
        snapshot = None
        for s in self.snapshots.values():
            if s.operation_id == operation_id:
                snapshot = s
                break

        if not snapshot or not snapshot.data_checksums:
            logger.error(f"未找到操作 {operation_id} 的数据快照或快照中无校验和信息")
            return False

        db_type = connector.database_type.lower()
        mismatch_found = False

        for table_name, expected_checksum in snapshot.data_checksums.items():
            try:
                if db_type == "mysql":
                    # 使用 CHECKSUM TABLE 重新计算校验和
                    checksum_sql = f"CHECKSUM TABLE {table_name}"
                    result = await connector.execute_query(checksum_sql)
                    if not result:
                        logger.warning(f"无法获取表 {table_name} 的当前校验和结果")
                        mismatch_found = True
                        continue
                    current_checksum = str(result[0].get("Checksum", ""))
                elif db_type == "postgresql":
                    # PostgreSQL 下, 快照中使用行数作为简单校验
                    count_sql = f"SELECT COUNT(*) as row_count FROM {table_name}"
                    result = await connector.execute_query(count_sql)
                    if not result:
                        logger.warning(f"无法获取表 {table_name} 的当前行数")
                        mismatch_found = True
                        continue
                    current_checksum = str(result[0].get("row_count", ""))
                else:
                    logger.warning(f"暂不支持的数据恢复数据库类型: {db_type}")
                    return False

                if current_checksum != expected_checksum:
                    logger.warning(
                        "表 %s 当前校验值(%s) 与快照值(%s) 不一致, 需要人工介入恢复",
                        table_name,
                        current_checksum,
                        expected_checksum,
                    )
                    mismatch_found = True
                else:
                    logger.info(
                        "表 %s 当前数据与快照校验和一致(值=%s)",
                        table_name,
                        current_checksum,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(f"检查表 {table_name} 数据校验和失败: {e}")
                mismatch_found = True

        # 如果存在不一致, 返回 False 以便上层流程决定是否继续或触发更高级别的恢复机制
        return not mismatch_found
    
    async def _reverse_alter_operation(self, step: Dict[str, Any], connector: BaseDatabaseConnector) -> bool:
        """反向ALTER操作."""
        # 当前实现聚焦于最常见、可安全反向的部分: "ALTER TABLE ... ADD COLUMN" -> "ALTER TABLE ... DROP COLUMN"。
        # 其它复杂 ALTER（如 DROP COLUMN、修改数据类型等）仍需人工介入，这里只做日志提示并返回 True，避免中断整个回滚流程。

        original_sql = (step.get("original_sql") or step.get("sql") or "").strip()
        sql_lower = original_sql.lower()

        if not sql_lower.startswith("alter table"):
            logger.warning(f"无法反向的ALTER语句(未识别为ALTER TABLE): {original_sql}")
            return True

        # 提取表名和后续子句
        m = re.match(r"alter\s+table\s+`?(\w+)`?\s+(.*)", sql_lower, re.IGNORECASE)
        if not m:
            logger.warning(f"无法解析ALTER TABLE语句: {original_sql}")
            return True

        table_name = m.group(1)
        alter_body = m.group(2).strip()

        # 仅处理单条 ADD COLUMN 情况: ALTER TABLE t ADD COLUMN col ...
        add_col_match = re.match(r"add\s+column\s+`?(\w+)`?", alter_body, re.IGNORECASE)
        if add_col_match:
            col_name = add_col_match.group(1)
            reverse_sql = f"ALTER TABLE {table_name} DROP COLUMN {col_name}"
            logger.info(f"生成反向ALTER语句: {reverse_sql} (原始: {original_sql})")
            try:
                await connector.execute_query(reverse_sql)
                return True
            except Exception as e:  # noqa: BLE001
                logger.error(f"执行反向ALTER失败: {e}")
                return False

        # 其它类型暂不自动反向
        logger.warning(f"暂不支持自动反向的ALTER语句, 请人工处理: {original_sql}")
        return True
    
    async def _execute_validation_queries(self, plan: RollbackPlan, connector: BaseDatabaseConnector) -> bool:
        """执行验证查询."""
        try:
            for query in plan.validation_queries:
                raw_query = query
                expectation = None

                # 解析期望标记: /* expect_zero */ 或 /* expect_positive */
                m = re.search(r"/\*\s*expect_(zero|positive)\s*\*/", query)
                if m:
                    expectation = m.group(1)
                    # 去掉注释再执行
                    query = re.sub(r"/\*.*?\*/", "", query).strip()

                result = await connector.execute_query(query)
                if not result:
                    logger.warning(f"验证查询无结果: {raw_query}")
                    return False

                if expectation:
                    count_val = result[0].get("count") or result[0].get("COUNT")
                    try:
                        count_int = int(count_val)
                    except Exception:  # noqa: BLE001
                        logger.warning(f"验证查询返回的count无法解析为整数: {raw_query}, 值: {count_val}")
                        return False

                    if expectation == "zero" and count_int != 0:
                        logger.warning(f"验证失败: 期望索引不存在(count=0), 实际为 {count_int}, 查询: {raw_query}")
                        return False
                    if expectation == "positive" and count_int <= 0:
                        logger.warning(f"验证失败: 期望索引存在(count>0), 实际为 {count_int}, 查询: {raw_query}")
                        return False

            return True
        except Exception as e:
            logger.error(f"执行验证查询异常: {e}")
            return False
    
    async def _generate_recovery_steps(self, operation_id: str, failure_reason: str) -> List[Dict[str, Any]]:
        """生成恢复步骤."""
        recovery_steps = []
        
        # 基于失败原因生成恢复步骤
        if "timeout" in failure_reason.lower():
            recovery_steps.append({
                "type": "retry_with_timeout",
                "description": "增加超时时间后重试",
                "timeout": 7200,  # 2小时
                "critical": False,
                "operation_id": operation_id,
            })
        
        if "lock" in failure_reason.lower():
            recovery_steps.append({
                "type": "wait_for_locks",
                "description": "等待锁释放后重试",
                "wait_time": 300,  # 5分钟
                "critical": False,
                "operation_id": operation_id,
            })
        
        if "permission" in failure_reason.lower():
            recovery_steps.append({
                "type": "check_permissions",
                "description": "检查并修复权限问题",
                "critical": True,
                "operation_id": operation_id,
            })
        
        # 默认恢复步骤
        if not recovery_steps:
            recovery_steps.append({
                "type": "rollback_and_retry",
                "description": "执行回滚后重试操作",
                "critical": False,
                "operation_id": operation_id,
            })
        
        return recovery_steps
    
    def _find_rollback_plan_by_operation(self, operation_id: str) -> Optional[RollbackPlan]:
        """根据操作ID查找回滚计划."""
        for plan in self.rollback_plans.values():
            if plan.operation_id == operation_id:
                return plan
        return None
    
    async def _execute_recovery_step(self, step: Dict[str, Any], connector: BaseDatabaseConnector, user: User) -> bool:
        """执行恢复步骤."""
        try:
            step_type = step.get("type", "")
            
            if step_type == "retry_with_timeout":
                # 重新执行原始操作（基于审计日志中的SQL），上层可在连接层配置更长超时时间
                operation_id = step.get("operation_id")
                if not operation_id:
                    logger.warning("retry_with_timeout 步骤缺少 operation_id")
                    return False

                audit_log = None
                for log in self.audit_logs.values():
                    if log.operation_id == operation_id:
                        audit_log = log
                        break

                if not audit_log:
                    logger.warning(f"未找到操作 {operation_id} 的审计日志，无法重试")
                    return False

                logger.info(f"开始重试执行原始操作({operation_id})，SQL条数: {len(audit_log.sql_statements)}")
                try:
                    for sql in audit_log.sql_statements:
                        await connector.execute_query(sql)
                    return True
                except Exception as e:  # noqa: BLE001
                    logger.error(f"重试执行原始操作失败: {e}")
                    return False
            elif step_type == "wait_for_locks":
                # 等待指定时间
                wait_time = step.get("wait_time", 60)
                await asyncio.sleep(wait_time)
                return True
            elif step_type == "check_permissions":
                # 调用 SafetyValidator 检查用户是否具备执行优化/回滚的权限
                if not self.safety_validator._can_execute_optimization(user):
                    logger.error(f"权限检查失败，用户 {user.username} 无权执行恢复/回滚")
                    return False
                logger.info(f"权限检查通过，用户 {user.username} 可执行恢复/回滚")
                return True
            elif step_type == "rollback_and_retry":
                # 先根据 operation_id 查找回滚计划并执行回滚，再尝试重试原始操作
                operation_id = step.get("operation_id")
                if not operation_id:
                    logger.warning("rollback_and_retry 步骤缺少 operation_id")
                    return False

                plan = self._find_rollback_plan_by_operation(operation_id)
                if not plan:
                    logger.warning(f"未找到操作 {operation_id} 的回滚计划，无法执行 rollback_and_retry")
                    return False

                rollback_success = await self.execute_rollback(plan.plan_id, connector, user)
                if not rollback_success:
                    logger.error(f"rollback_and_retry 中回滚失败, operation_id={operation_id}")
                    return False

                # 回滚成功后重试原始操作
                retry_step = {"type": "retry_with_timeout", "operation_id": operation_id}
                return await self._execute_recovery_step(retry_step, connector, user)
            else:
                logger.warning(f"未知的恢复步骤类型: {step_type}")
                return True
        except Exception as e:
            logger.error(f"执行恢复步骤失败: {e}")
            return False
    
    # 公共查询方法
    
    def get_snapshots(self, database: str = None, operation_id: str = None) -> List[DatabaseSnapshot]:
        """获取快照列表."""
        snapshots = list(self.snapshots.values())
        
        if database:
            snapshots = [s for s in snapshots if s.database == database]
        
        if operation_id:
            snapshots = [s for s in snapshots if s.operation_id == operation_id]
        
        return snapshots
    
    def get_rollback_plans(self, database: str = None, status: RecoveryStatus = None) -> List[RollbackPlan]:
        """获取回滚计划列表."""
        plans = list(self.rollback_plans.values())
        
        if database:
            plans = [p for p in plans if p.database == database]
        
        if status:
            plans = [p for p in plans if p.status == status]
        
        return plans
    
    def get_recovery_operations(self, operation_id: str = None, status: RecoveryStatus = None) -> List[RecoveryOperation]:
        """获取恢复操作列表."""
        operations = list(self.recovery_operations.values())
        
        if operation_id:
            operations = [o for o in operations if o.operation_id == operation_id]
        
        if status:
            operations = [o for o in operations if o.status == status]
        
        return operations
    
    def get_audit_logs(self, database: str = None, user_id: str = None, 
                      start_time: datetime = None, end_time: datetime = None) -> List[OperationAuditLog]:
        """获取审计日志列表."""
        logs = list(self.audit_logs.values())
        
        if database:
            logs = [l for l in logs if l.database == database]
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if start_time:
            logs = [l for l in logs if l.started_at >= start_time]
        
        if end_time:
            logs = [l for l in logs if l.started_at <= end_time]
        
        return logs
    
    def export_audit_logs(self, format: str = "json") -> str:
        """导出审计日志."""
        if format == "json":
            return json.dumps([log.dict() for log in self.audit_logs.values()], default=str, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")