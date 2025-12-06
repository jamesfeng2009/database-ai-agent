"""自动优化器模块 - 提供数据库自动化优化功能."""

import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

try:  # 可选依赖, 用于获取系统资源信息, 不存在时自动降级
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None

from ...database.connector_base import BaseDatabaseConnector
from ...database.adapters import DatabaseAdapter, DatabaseAdapterFactory
from ...tools import _detect_database_type
from .safety_validator import SafetyValidator, User, ValidationResult, RiskLevel
from .knowledge_service import KnowledgeService
from ..models.models import Task, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class OptimizationType(str, Enum):
    """优化类型枚举."""
    INDEX_CREATION = "index_creation"
    INDEX_DELETION = "index_deletion"
    STATISTICS_UPDATE = "statistics_update"
    QUERY_REWRITE = "query_rewrite"
    CONFIG_TUNING = "config_tuning"
    BATCH_OPTIMIZATION = "batch_optimization"


class OptimizationStatus(str, Enum):
    """优化状态枚举."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class IndexSpec(BaseModel):
    """索引规格模型."""
    table_name: str = Field(..., description="表名")
    columns: List[str] = Field(..., description="索引列")
    index_name: Optional[str] = Field(None, description="索引名称")
    index_type: str = Field(default="BTREE", description="索引类型")
    is_unique: bool = Field(default=False, description="是否唯一索引")
    is_partial: bool = Field(default=False, description="是否部分索引")
    where_clause: Optional[str] = Field(None, description="部分索引条件")
    storage_parameters: Dict[str, Any] = Field(default_factory=dict, description="存储参数")


class QueryRewriteSuggestion(BaseModel):
    """查询重写建议模型."""
    original_query: str = Field(..., description="原始查询")
    rewritten_query: str = Field(..., description="重写后查询")
    improvement_type: str = Field(..., description="改进类型")
    expected_improvement: str = Field(..., description="预期改进")
    confidence: float = Field(..., description="建议置信度")
    explanation: str = Field(..., description="重写说明")
    rule_name: Optional[str] = Field(None, description="触发该建议的重写规则名称")


class ConfigParameter(BaseModel):
    """配置参数模型."""
    parameter_name: str = Field(..., description="参数名称")
    current_value: Any = Field(..., description="当前值")
    recommended_value: Any = Field(..., description="推荐值")
    parameter_type: str = Field(..., description="参数类型")
    description: str = Field(..., description="参数描述")
    impact_level: str = Field(..., description="影响级别")
    requires_restart: bool = Field(default=False, description="是否需要重启")


class OptimizationResult(BaseModel):
    """优化结果模型."""
    optimization_id: str = Field(default_factory=lambda: str(uuid4()), description="优化ID")
    optimization_type: OptimizationType = Field(..., description="优化类型")
    status: OptimizationStatus = Field(..., description="优化状态")
    database: str = Field(..., description="数据库名")
    target_object: str = Field(..., description="目标对象")
    executed_sql: List[str] = Field(default_factory=list, description="执行的SQL")
    execution_time: float = Field(..., description="执行时间(秒)")
    performance_impact: Dict[str, Any] = Field(default_factory=dict, description="性能影响")
    rollback_sql: List[str] = Field(default_factory=list, description="回滚SQL")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class BatchOptimizationPlan(BaseModel):
    """批量优化计划模型."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()), description="计划ID")
    database: str = Field(..., description="数据库名")
    optimizations: List[Dict[str, Any]] = Field(..., description="优化操作列表")
    execution_order: List[str] = Field(..., description="执行顺序")
    rollback_plan: Dict[str, Any] = Field(..., description="回滚计划")
    estimated_duration: float = Field(..., description="预估执行时间")
    risk_assessment: RiskLevel = Field(..., description="风险评估")


class AutoOptimizer:
    """自动优化器 - 提供数据库自动化优化功能."""
    
    def __init__(self, safety_validator: SafetyValidator, knowledge_service: KnowledgeService | None = None):
        """初始化自动优化器."""
        self.safety_validator = safety_validator
        # 可选的知识服务, 用于将历史案例/规范附加到优化建议中
        self.knowledge_service: KnowledgeService | None = knowledge_service
        self.optimization_history: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationResult] = {}
        # schema 信息缓存: (db_type, table_name) -> [columns]
        self._schema_cache: Dict[Tuple[str, str], List[str]] = {}
        
        # 初始化优化规则
        self._init_optimization_rules()
    
    def _init_optimization_rules(self):
        """初始化优化规则."""
        # 索引创建规则
        self.index_creation_rules = {
            "where_clause_columns": {
                "priority": "high",
                "description": "为WHERE子句中的列创建索引"
            },
            "join_columns": {
                "priority": "high", 
                "description": "为JOIN条件中的列创建索引"
            },
            "order_by_columns": {
                "priority": "medium",
                "description": "为ORDER BY子句中的列创建索引"
            },
            "group_by_columns": {
                "priority": "medium",
                "description": "为GROUP BY子句中的列创建索引"
            }
        }
        
        # 查询重写规则
        self.query_rewrite_rules = {
            "subquery_to_join": {
                "pattern": r"SELECT.*FROM.*WHERE.*IN\s*\(SELECT",
                "description": "将子查询转换为JOIN"
            },
            "exists_to_join": {
                "pattern": r"WHERE\s+EXISTS\s*\(",
                "description": "将EXISTS转换为JOIN(提示型)"
            },
            "union_to_union_all": {
                "pattern": r"UNION(?!\s+ALL)",
                "description": "在结果无需去重时, 可考虑将UNION改为UNION ALL(提示型)"
            },
            "select_star_to_columns": {
                "pattern": r"SELECT\s+\*\s+FROM",
                "description": "将SELECT * 基于schema改写为具体列名(仅MySQL)"
            },
            "or_to_in_list": {
                "pattern": r"WHERE\s+.+\s*=\s*.+\s+OR\s+",
                "description": "将同一列的多个OR等值条件合并为IN列表(简化版)"
            },
            "implicit_join_to_explicit": {
                "pattern": r"FROM\s+[a-zA-Z_][a-zA-Z0-9_]*\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*",
                "description": "将FROM t1, t2隐式连接改写为显式JOIN(提示型)"
            },
        }
        
        # 配置参数优化规则
        self.config_tuning_rules = {
            "mysql": {
                "innodb_buffer_pool_size": {
                    "calculation": "memory * 0.7",
                    "min_value": "128M",
                    "description": "InnoDB缓冲池大小"
                },
                "query_cache_size": {
                    "calculation": "memory * 0.1",
                    "max_value": "256M",
                    "description": "查询缓存大小"
                }
            },
            "postgresql": {
                "shared_buffers": {
                    "calculation": "memory * 0.25",
                    "min_value": "128MB",
                    "description": "共享缓冲区大小"
                },
                "effective_cache_size": {
                    "calculation": "memory * 0.75",
                    "description": "有效缓存大小"
                }
            }
        }
    
    async def create_index(self, 
                          connector: BaseDatabaseConnector,
                          index_spec: IndexSpec,
                          user: User) -> OptimizationResult:
        """
        创建索引.
        
        Args:
            connector: 数据库连接器
            index_spec: 索引规格
            user: 执行用户
            
        Returns:
            OptimizationResult: 优化结果
        """
        optimization_id = str(uuid4())
        logger.info(f"开始创建索引，优化ID: {optimization_id}")
        
        start_time = datetime.now()
        result = OptimizationResult(
            optimization_id=optimization_id,
            optimization_type=OptimizationType.INDEX_CREATION,
            status=OptimizationStatus.ANALYZING,
            database=connector.config.database,
            target_object=f"{index_spec.table_name}.{index_spec.index_name or 'auto_idx'}",
            execution_time=0.0
        )
        
        try:
            # 1. 生成索引创建SQL
            create_sql = self._generate_create_index_sql(index_spec, connector.database_type)
            result.executed_sql = [create_sql]
            
            # 2. 安全验证
            validation_result = await self.safety_validator.validate_sql_operation(
                create_sql, user, connector.config.database
            )
            
            if not validation_result.is_valid:
                result.status = OptimizationStatus.FAILED
                result.error_message = f"安全验证失败: {', '.join(validation_result.violations)}"
                return result
            
            # 3. 检查索引是否已存在
            if await self._index_exists(connector, index_spec):
                result.status = OptimizationStatus.FAILED
                result.error_message = "索引已存在"
                return result
            
            # 4. 执行索引创建
            result.status = OptimizationStatus.EXECUTING
            self.active_optimizations[optimization_id] = result
            
            await connector.execute_query(create_sql)
            
            # 5. 生成回滚SQL
            drop_sql = self._generate_drop_index_sql(index_spec, connector.database_type)
            result.rollback_sql = [drop_sql]
            
            # 6. 评估性能影响
            performance_impact = await self._assess_index_performance_impact(
                connector, index_spec
            )
            result.performance_impact = performance_impact

            # 6.1 可选: 基于索引信息关联知识库
            if self.knowledge_service is not None:
                try:
                    ctx = {
                        "issue": "index_creation",
                        "table": index_spec.table_name,
                        "columns": ",".join(index_spec.columns),
                    }
                    desc = f"index on {index_spec.table_name}({','.join(index_spec.columns)})"
                    entries = await self.knowledge_service.related_to_sql(desc, ctx, limit=5)
                    if entries:
                        result.performance_impact["related_knowledge"] = [
                            {
                                "entry_id": e.entry_id,
                                "title": e.title,
                                "source": e.source,
                            }
                            for e in entries
                        ]
                except Exception as e:  # noqa: BLE001
                    logger.warning("关联索引相关知识失败: %s", e)
            
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            logger.info(f"索引创建成功，优化ID: {optimization_id}")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            logger.error(f"索引创建失败，优化ID: {optimization_id}, 错误: {e}")
        
        finally:
            self.optimization_history.append(result)
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
        
        return result
    
    async def drop_index(self,
                        connector: BaseDatabaseConnector,
                        index_name: str,
                        table_name: str,
                        user: User) -> OptimizationResult:
        """
        删除索引.
        
        Args:
            connector: 数据库连接器
            index_name: 索引名称
            table_name: 表名
            user: 执行用户
            
        Returns:
            OptimizationResult: 优化结果
        """
        optimization_id = str(uuid4())
        logger.info(f"开始删除索引，优化ID: {optimization_id}")
        
        start_time = datetime.now()
        result = OptimizationResult(
            optimization_id=optimization_id,
            optimization_type=OptimizationType.INDEX_DELETION,
            status=OptimizationStatus.ANALYZING,
            database=connector.config.database,
            target_object=f"{table_name}.{index_name}",
            execution_time=0.0
        )
        
        try:
            # 1. 获取索引信息用于回滚
            index_info = await self._get_index_info(connector, index_name, table_name)
            if not index_info:
                result.status = OptimizationStatus.FAILED
                result.error_message = "索引不存在"
                return result
            
            # 2. 生成删除SQL
            drop_sql = self._generate_drop_index_sql_by_name(
                index_name, table_name, connector.database_type
            )
            result.executed_sql = [drop_sql]
            
            # 3. 安全验证
            validation_result = await self.safety_validator.validate_sql_operation(
                drop_sql, user, connector.config.database
            )
            
            if not validation_result.is_valid:
                result.status = OptimizationStatus.FAILED
                result.error_message = f"安全验证失败: {', '.join(validation_result.violations)}"
                return result
            
            # 4. 执行索引删除
            result.status = OptimizationStatus.EXECUTING
            self.active_optimizations[optimization_id] = result
            
            await connector.execute_query(drop_sql)
            
            # 5. 生成回滚SQL（重新创建索引）
            recreate_sql = self._generate_recreate_index_sql(index_info, connector.database_type)
            result.rollback_sql = [recreate_sql]
            
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            logger.info(f"索引删除成功，优化ID: {optimization_id}")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            logger.error(f"索引删除失败，优化ID: {optimization_id}, 错误: {e}")
        
        finally:
            self.optimization_history.append(result)
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
        
        return result
    
    async def update_statistics(self,
                               connector: BaseDatabaseConnector,
                               tables: List[str],
                               user: User) -> OptimizationResult:
        """
        批量更新统计信息.
        
        Args:
            connector: 数据库连接器
            tables: 表名列表
            user: 执行用户
            
        Returns:
            OptimizationResult: 优化结果
        """
        optimization_id = str(uuid4())
        logger.info(f"开始更新统计信息，优化ID: {optimization_id}")
        
        start_time = datetime.now()
        result = OptimizationResult(
            optimization_id=optimization_id,
            optimization_type=OptimizationType.STATISTICS_UPDATE,
            status=OptimizationStatus.ANALYZING,
            database=connector.config.database,
            target_object=f"tables: {', '.join(tables)}",
            execution_time=0.0
        )
        
        try:
            # 1. 生成统计信息更新SQL
            update_sqls = self._generate_statistics_update_sql(tables, connector.database_type)
            result.executed_sql = update_sqls
            
            # 2. 安全验证
            for sql in update_sqls:
                validation_result = await self.safety_validator.validate_sql_operation(
                    sql, user, connector.config.database
                )
                
                if not validation_result.is_valid:
                    result.status = OptimizationStatus.FAILED
                    result.error_message = f"安全验证失败: {', '.join(validation_result.violations)}"
                    return result
            
            # 3. 执行统计信息更新
            result.status = OptimizationStatus.EXECUTING
            self.active_optimizations[optimization_id] = result
            
            for sql in update_sqls:
                await connector.execute_query(sql)
            
            # 4. 评估性能影响
            performance_impact = await self._assess_statistics_performance_impact(
                connector, tables
            )
            result.performance_impact = performance_impact

            # 4.1 可选: 关联统计信息相关知识
            if self.knowledge_service is not None:
                try:
                    ctx = {
                        "issue": "statistics_update",
                        "tables": ",".join(tables),
                    }
                    desc = f"statistics update on tables {','.join(tables)}"
                    entries = await self.knowledge_service.related_to_sql(desc, ctx, limit=5)
                    if entries:
                        result.performance_impact["related_knowledge"] = [
                            {
                                "entry_id": e.entry_id,
                                "title": e.title,
                                "source": e.source,
                            }
                            for e in entries
                        ]
                except Exception as e:  # noqa: BLE001
                    logger.warning("关联统计信息相关知识失败: %s", e)
            
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            logger.info(f"统计信息更新成功，优化ID: {optimization_id}")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            logger.error(f"统计信息更新失败，优化ID: {optimization_id}, 错误: {e}")
        
        finally:
            self.optimization_history.append(result)
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
        
        return result
    
    async def suggest_query_rewrite(self,
                                   sql: str,
                                   connector: BaseDatabaseConnector) -> List[QueryRewriteSuggestion]:
        """
        生成查询重写建议.
        
        Args:
            sql: 原始SQL查询
            connector: 数据库连接器
            
        Returns:
            List[QueryRewriteSuggestion]: 查询重写建议列表
        """
        logger.info("开始生成查询重写建议")
        
        suggestions = []
        
        try:
            # 1. 分析查询结构
            query_analysis = await self._analyze_query_structure(sql, connector)
            
            # 2. 应用重写规则
            for rule_name, rule_config in self.query_rewrite_rules.items():
                suggestion = await self._apply_rewrite_rule(
                    sql, rule_name, rule_config, query_analysis
                )
                if suggestion:
                    suggestions.append(suggestion)
            
            # 3. 基于执行计划的重写建议
            execution_plan_suggestions = await self._generate_execution_plan_based_suggestions(
                sql, connector
            )
            suggestions.extend(execution_plan_suggestions)
            
            # 4. 按置信度排序
            suggestions.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"生成了 {len(suggestions)} 个查询重写建议")
            
        except Exception as e:
            logger.error(f"生成查询重写建议失败: {e}")
        
        return suggestions
    
    async def optimize_configuration(self,
                                   connector: BaseDatabaseConnector,
                                   user: User) -> OptimizationResult:
        """
        优化数据库配置参数.
        
        Args:
            connector: 数据库连接器
            user: 执行用户
            
        Returns:
            OptimizationResult: 优化结果
        """
        optimization_id = str(uuid4())
        logger.info(f"开始优化数据库配置，优化ID: {optimization_id}")
        
        start_time = datetime.now()
        result = OptimizationResult(
            optimization_id=optimization_id,
            optimization_type=OptimizationType.CONFIG_TUNING,
            status=OptimizationStatus.ANALYZING,
            database=connector.config.database,
            target_object="database_configuration",
            execution_time=0.0
        )
        
        try:
            # 1. 获取当前配置
            current_config = await self._get_current_configuration(connector)
            
            # 2. 分析系统资源
            system_resources = await self._analyze_system_resources(connector)
            
            # 3. 生成配置建议
            config_recommendations = self._generate_config_recommendations(
                current_config, system_resources, connector.database_type
            )
            
            # 4. 生成配置更新SQL
            config_sqls = self._generate_config_update_sql(
                config_recommendations, connector.database_type
            )
            result.executed_sql = config_sqls
            
            # 5. 安全验证
            for sql in config_sqls:
                validation_result = await self.safety_validator.validate_sql_operation(
                    sql, user, connector.config.database
                )
                
                if not validation_result.is_valid:
                    result.status = OptimizationStatus.FAILED
                    result.error_message = f"安全验证失败: {', '.join(validation_result.violations)}"
                    return result
            
            # 6. 执行配置更新
            result.status = OptimizationStatus.EXECUTING
            self.active_optimizations[optimization_id] = result
            
            for sql in config_sqls:
                await connector.execute_query(sql)
            
            # 7. 生成回滚SQL
            rollback_sqls = self._generate_config_rollback_sql(
                current_config, config_recommendations, connector.database_type
            )
            result.rollback_sql = rollback_sqls
            
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            result.performance_impact = {
                "config_changes": len(config_recommendations),
                "requires_restart": any(rec.requires_restart for rec in config_recommendations)
            }

            # 8. 可选: 为配置调优结果关联知识库
            if self.knowledge_service is not None:
                try:
                    ctx = {
                        "issue": "config_tuning",
                        "db_type": connector.database_type,
                    }
                    desc = "database configuration tuning"
                    entries = await self.knowledge_service.related_to_sql(desc, ctx, limit=5)
                    if entries:
                        result.performance_impact["related_knowledge"] = [
                            {
                                "entry_id": e.entry_id,
                                "title": e.title,
                                "source": e.source,
                            }
                            for e in entries
                        ]
                except Exception as e:  # noqa: BLE001
                    logger.warning("关联配置调优相关知识失败: %s", e)
            
            logger.info(f"数据库配置优化成功，优化ID: {optimization_id}")
            
        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            logger.error(f"数据库配置优化失败，优化ID: {optimization_id}, 错误: {e}")
        
        finally:
            self.optimization_history.append(result)
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
        
        return result
    
    async def execute_batch_optimization(self,
                                       plan: BatchOptimizationPlan,
                                       connector: BaseDatabaseConnector,
                                       user: User) -> List[OptimizationResult]:
        """
        执行批量优化计划.
        
        Args:
            plan: 批量优化计划
            connector: 数据库连接器
            user: 执行用户
            
        Returns:
            List[OptimizationResult]: 优化结果列表
        """
        logger.info(f"开始执行批量优化计划，计划ID: {plan.plan_id}")
        
        results = []
        
        try:
            # 1. 安全验证整个计划
            plan_validation = await self._validate_batch_plan(plan, user)
            if not plan_validation.is_valid:
                logger.error(f"批量优化计划验证失败: {plan_validation.violations}")
                return results
            
            # 2. 按执行顺序执行优化
            for optimization_id in plan.execution_order:
                optimization_config = next(
                    (opt for opt in plan.optimizations if opt["id"] == optimization_id),
                    None
                )
                
                if not optimization_config:
                    logger.warning(f"未找到优化配置: {optimization_id}")
                    continue
                
                # 执行单个优化
                result = await self._execute_single_optimization(
                    optimization_config, connector, user
                )
                results.append(result)
                
                # 如果优化失败，根据策略决定是否继续
                if result.status == OptimizationStatus.FAILED:
                    if optimization_config.get("stop_on_failure", True):
                        logger.error(f"优化失败，停止执行: {optimization_id}")
                        break
            
            logger.info(f"批量优化计划执行完成，计划ID: {plan.plan_id}")
            
        except Exception as e:
            logger.error(f"批量优化计划执行失败，计划ID: {plan.plan_id}, 错误: {e}")
        
        return results
    
    async def rollback_optimization(self,
                                   optimization_id: str,
                                   connector: BaseDatabaseConnector,
                                   user: User) -> bool:
        """
        回滚优化操作.
        
        Args:
            optimization_id: 优化ID
            connector: 数据库连接器
            user: 执行用户
            
        Returns:
            bool: 回滚是否成功
        """
        logger.info(f"开始回滚优化操作，优化ID: {optimization_id}")
        
        try:
            # 1. 查找优化记录
            optimization_result = next(
                (result for result in self.optimization_history 
                 if result.optimization_id == optimization_id),
                None
            )
            
            if not optimization_result:
                logger.error(f"未找到优化记录: {optimization_id}")
                return False
            
            if not optimization_result.rollback_sql:
                logger.error(f"优化记录没有回滚SQL: {optimization_id}")
                return False
            
            # 2. 安全验证回滚操作
            for sql in optimization_result.rollback_sql:
                validation_result = await self.safety_validator.validate_sql_operation(
                    sql, user, connector.config.database
                )
                
                if not validation_result.is_valid:
                    logger.error(f"回滚操作安全验证失败: {validation_result.violations}")
                    return False
            
            # 3. 执行回滚
            for sql in optimization_result.rollback_sql:
                await connector.execute_query(sql)
            
            # 4. 更新优化状态
            optimization_result.status = OptimizationStatus.ROLLED_BACK
            
            logger.info(f"优化操作回滚成功，优化ID: {optimization_id}")
            return True
            
        except Exception as e:
            logger.error(f"优化操作回滚失败，优化ID: {optimization_id}, 错误: {e}")
            return False
    
    def get_optimization_history(self,
                               database: str = None,
                               optimization_type: OptimizationType = None,
                               status: OptimizationStatus = None) -> List[OptimizationResult]:
        """
        获取优化历史记录.
        
        Args:
            database: 数据库名（可选）
            optimization_type: 优化类型（可选）
            status: 优化状态（可选）
            
        Returns:
            List[OptimizationResult]: 优化历史记录
        """
        results = self.optimization_history
        
        if database:
            results = [r for r in results if r.database == database]
        
        if optimization_type:
            results = [r for r in results if r.optimization_type == optimization_type]
        
        if status:
            results = [r for r in results if r.status == status]
        
        return results
    
    def get_active_optimizations(self) -> Dict[str, OptimizationResult]:
        """获取当前活跃的优化操作."""
        return self.active_optimizations.copy()

    async def optimize_query(
        self,
        sql: str,
        connector: BaseDatabaseConnector,
        user: User,
        auto_apply: bool = False,
    ) -> Dict[str, Any]:
        """对单条查询执行自动优化流程.

        当前实现只生成重写候选并基于 EXPLAIN 评估成本, 不直接在数据库中
        执行任何结构性变更或自动替换原始 SQL。
        """

        logger.info("开始自动优化查询")

        # 1. 生成重写候选
        rewrite_suggestions = await self.suggest_query_rewrite(sql, connector)
        candidate_sqls: List[str] = []
        for s in rewrite_suggestions:
            if s.rewritten_query and s.rewritten_query.strip() and s.rewritten_query.strip() != sql.strip():
                candidate_sqls.append(s.rewritten_query)

        # 确保至少评估原始 SQL
        all_sqls = [sql] + [c for c in candidate_sqls if c.strip()]

        # 2. 评估所有候选
        eval_results = await self.evaluate_query_candidates(connector, sql, all_sqls)
        if not eval_results:
            # 没有可用评估结果时，构造一个保守的 planned_actions，仅包含配置调优和可选的统计更新
            import re  # 局部导入，避免破坏现有模块结构

            base_sql = sql
            table_name = None
            m = re.search(r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)", base_sql, re.IGNORECASE)
            if m:
                table_name = m.group(1)

            planned_actions: list[dict[str, Any]] = []
            if table_name:
                planned_actions.append(
                    {
                        "id": f"stats-{table_name}",
                        "type": "statistics_update",
                        "tables": [table_name],
                    }
                )

            planned_actions.append(
                {
                    "id": "config-tuning",
                    "type": "config_tuning",
                }
            )

            return {
                "original_sql": sql,
                "best_sql": sql,
                "improvement": 0.0,
                "candidates": [],
                "rewrite_suggestions": [s.dict() for s in rewrite_suggestions],
                "auto_applied": False,
                "planned_actions": planned_actions,
            }

        # 找到原始 SQL 和最佳候选
        orig_entry = next((e for e in eval_results if e["sql"] == sql), None)
        best_entry = min(eval_results, key=lambda e: e["cost"])

        orig_cost = orig_entry["cost"] if orig_entry else best_entry["cost"]
        best_cost = best_entry["cost"]
        improvement = max(0.0, orig_cost - best_cost)

        # 尝试找到与最佳 SQL 对应的重写建议, 以便向上层解释原因
        best_suggestion = None
        for s in rewrite_suggestions:
            try:
                if s.rewritten_query and s.rewritten_query.strip() == best_entry["sql"].strip():
                    best_suggestion = s
                    break
            except Exception:
                continue

        best_reason = None
        if best_suggestion is not None:
            best_reason = {
                "rule_name": best_suggestion.rule_name,
                "explanation": best_suggestion.explanation,
                "improvement_type": best_suggestion.improvement_type,
            }

        # 根据 auto_apply 开关和安全验证结果决定是否认为可以“自动使用” best_sql
        auto_applied = False
        if auto_apply and best_entry["sql"].strip() != sql.strip():
            try:
                validation = await self.safety_validator.validate_sql_operation(
                    best_entry["sql"], user, connector.config.database
                )
                auto_applied = bool(validation.is_valid)
            except Exception as e:  # noqa: BLE001
                logger.warning("auto_apply 验证失败: %s", e)
                auto_applied = False

        # 基于 best_sql 构造一组保守的 planned_actions，供后续任务执行阶段消费
        import re  # 局部导入，避免影响模块其它部分

        base_sql = best_entry["sql"] or sql
        table_name = None
        where_column = None

        try:
            m = re.search(r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)", base_sql, re.IGNORECASE)
            if m:
                table_name = m.group(1)

            mw = re.search(r"where\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=", base_sql, re.IGNORECASE)
            if mw:
                where_column = mw.group(1)
        except Exception as e:  # noqa: BLE001
            logger.warning("解析 planned_actions 所需的表/列信息失败: %s", e)

        planned_actions: list[dict[str, Any]] = []

        # 1) 如果 best_sql 与原始 sql 不同，给出一个 execute_sql 动作
        if best_entry["sql"].strip() != sql.strip():
            planned_actions.append(
                {
                    "id": "rewrite-best-sql",
                    "type": "execute_sql",
                    "sqls": [best_entry["sql"]],
                }
            )

        # 2) 如果能解析出表名，给出 statistics_update 动作
        if table_name:
            planned_actions.append(
                {
                    "id": f"stats-{table_name}",
                    "type": "statistics_update",
                    "tables": [table_name],
                }
            )

        # 3) 如果能解析出 where 列，则为该列构造一个单列索引创建动作
        if table_name and where_column:
            planned_actions.append(
                {
                    "id": f"idx-{table_name}-{where_column}",
                    "type": "index_creation",
                    "spec": {
                        "table_name": table_name,
                        "columns": [where_column],
                        "index_name": f"idx_{table_name}_{where_column}",
                        "index_type": "BTREE",
                        "is_unique": False,
                        "is_partial": False,
                        "where_clause": None,
                        "storage_parameters": {},
                    },
                }
            )

        # 4) 始终附加一个配置调优动作
        planned_actions.append(
            {
                "id": "config-tuning",
                "type": "config_tuning",
            }
        )

        return {
            "original_sql": sql,
            "best_sql": best_entry["sql"],
            "best_cost": best_cost,
            "original_cost": orig_cost,
            "improvement": improvement,
            "best_reason": best_reason,
            "candidates": eval_results,
            "rewrite_suggestions": [s.dict() for s in rewrite_suggestions],
            "auto_applied": auto_applied,
            "planned_actions": planned_actions,
        }

    async def evaluate_query_candidates(
        self,
        connector: BaseDatabaseConnector,
        original_sql: str,
        candidate_sqls: List[str],
    ) -> List[Dict[str, Any]]:
        """对给定的一组 SQL 候选进行 EXPLAIN 评估并打分.

        返回列表中每个元素包含: sql / role / cost / explain_results。
        cost 只是相对指标, 仅用于在同一批候选中比较优劣。
        """

        results: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for sql in candidate_sqls:
            normalized = sql.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            try:
                explain_results = await connector.execute_explain(sql)
                cost = self._estimate_query_cost(explain_results)
                results.append(
                    {
                        "sql": sql,
                        "role": "original" if sql == original_sql else "candidate",
                        "cost": cost,
                        "explain_results": [r.dict() for r in explain_results],
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("评估候选 SQL 失败: %s", e)
                continue

        # 按成本从低到高排序, 成本越低认为越优
        results.sort(key=lambda x: x["cost"])
        return results

    def _estimate_query_cost(self, explain_results: List[Any]) -> float:
        """基于 EXPLAIN 结果估算查询成本 (简化版).

        成本主要由预估扫描行数、是否存在全表扫描、是否使用临时表/文件排序等决定。
        返回的成本值仅在同一批候选间比较有意义。
        """

        if not explain_results:
            return float("inf")

        try:
            db_type = _detect_database_type(explain_results)
            adapter: DatabaseAdapter = DatabaseAdapterFactory.create_adapter(db_type)
        except Exception:
            # 如果适配器创建失败, 退化为简单统计 rows
            total_rows = 0
            for r in explain_results:
                rows = getattr(r, "rows", None)
                if isinstance(rows, int):
                    total_rows += rows
            return float(total_rows or 1)

        total_rows = 0
        has_full_scan = False
        has_temp = False
        has_filesort = False

        for r in explain_results:
            try:
                rows = adapter.get_scan_rows(r)
            except Exception:
                rows = getattr(r, "rows", 0) or 0
            total_rows += rows

            try:
                if adapter.is_full_table_scan(r):
                    has_full_scan = True
            except Exception:
                pass

            try:
                extra = adapter.get_extra_info(r) or ""
            except Exception:
                extra = getattr(r, "extra", "") or ""

            if "Using temporary" in str(extra):
                has_temp = True
            if "Using filesort" in str(extra):
                has_filesort = True

        # 基础成本 = 总扫描行数, 加上一些惩罚项
        cost = float(max(total_rows, 1))
        if has_full_scan:
            cost *= 5.0
        if has_temp:
            cost *= 1.5
        if has_filesort:
            cost *= 1.5

        return cost

    # 私有辅助方法
    
    def _generate_create_index_sql(self, index_spec: IndexSpec, database_type: str) -> str:
        """生成创建索引的SQL."""
        index_name = index_spec.index_name or f"idx_{index_spec.table_name}_{'_'.join(index_spec.columns)}"
        columns_str = ', '.join(index_spec.columns)
        
        if database_type.lower() == "mysql":
            sql = f"CREATE"
            if index_spec.is_unique:
                sql += " UNIQUE"
            sql += f" INDEX {index_name} ON {index_spec.table_name} ({columns_str})"
            if index_spec.index_type != "BTREE":
                sql += f" USING {index_spec.index_type}"
        
        elif database_type.lower() == "postgresql":
            sql = f"CREATE"
            if index_spec.is_unique:
                sql += " UNIQUE"
            sql += f" INDEX {index_name} ON {index_spec.table_name}"
            if index_spec.index_type != "BTREE":
                sql += f" USING {index_spec.index_type}"
            sql += f" ({columns_str})"
            if index_spec.where_clause:
                sql += f" WHERE {index_spec.where_clause}"
        
        else:
            raise ValueError(f"不支持的数据库类型: {database_type}")
        
        return sql
    
    def _generate_drop_index_sql(self, index_spec: IndexSpec, database_type: str) -> str:
        """生成删除索引的SQL."""
        index_name = index_spec.index_name or f"idx_{index_spec.table_name}_{'_'.join(index_spec.columns)}"
        
        if database_type.lower() == "mysql":
            return f"DROP INDEX {index_name} ON {index_spec.table_name}"
        elif database_type.lower() == "postgresql":
            return f"DROP INDEX {index_name}"
        else:
            raise ValueError(f"不支持的数据库类型: {database_type}")
    
    def _generate_drop_index_sql_by_name(self, index_name: str, table_name: str, database_type: str) -> str:
        """根据索引名生成删除索引的SQL."""
        if database_type.lower() == "mysql":
            return f"DROP INDEX {index_name} ON {table_name}"
        elif database_type.lower() == "postgresql":
            return f"DROP INDEX {index_name}"
        else:
            raise ValueError(f"不支持的数据库类型: {database_type}")
    
    def _generate_statistics_update_sql(self, tables: List[str], database_type: str) -> List[str]:
        """生成统计信息更新SQL."""
        sqls = []
        
        if database_type.lower() == "mysql":
            for table in tables:
                sqls.append(f"ANALYZE TABLE {table}")
        elif database_type.lower() == "postgresql":
            for table in tables:
                sqls.append(f"ANALYZE {table}")
        else:
            raise ValueError(f"不支持的数据库类型: {database_type}")
        
        return sqls
    
    async def _index_exists(self, connector: BaseDatabaseConnector, index_spec: IndexSpec) -> bool:
        """检查索引是否已存在."""
        index_name = index_spec.index_name or f"idx_{index_spec.table_name}_{'_'.join(index_spec.columns)}"
        
        if connector.database_type.lower() == "mysql":
            sql = f"""
            SELECT COUNT(*) as count 
            FROM information_schema.statistics 
            WHERE table_schema = DATABASE() 
            AND table_name = '{index_spec.table_name}' 
            AND index_name = '{index_name}'
            """
        elif connector.database_type.lower() == "postgresql":
            sql = f"""
            SELECT COUNT(*) as count 
            FROM pg_indexes 
            WHERE tablename = '{index_spec.table_name}' 
            AND indexname = '{index_name}'
            """
        else:
            return False
        
        try:
            result = await connector.execute_query(sql)
            return result[0]["count"] > 0 if result else False
        except Exception:
            return False
    
    async def _get_index_info(self, connector: BaseDatabaseConnector, index_name: str, table_name: str) -> Optional[Dict[str, Any]]:
        """获取索引信息.

        对 MySQL 返回 information_schema.statistics 行列表(同一索引的多列),
        对 PostgreSQL 返回 pg_indexes 的单行记录。
        """
        if connector.database_type.lower() == "mysql":
            sql = f"""
            SELECT * FROM information_schema.statistics 
            WHERE table_schema = DATABASE() 
            AND table_name = '{table_name}' 
            AND index_name = '{index_name}'
            ORDER BY seq_in_index
            """
        elif connector.database_type.lower() == "postgresql":
            sql = f"""
            SELECT * FROM pg_indexes 
            WHERE tablename = '{table_name}' 
            AND indexname = '{index_name}'
            """
        else:
            return None
        
        try:
            result = await connector.execute_query(sql)
            if connector.database_type.lower() == "mysql":
                # MySQL: 返回多行(每行一列), 交给后续逻辑分组
                return result or None
            # PostgreSQL: 只需一行 indexdef
            return result[0] if result else None
        except Exception:
            return None
    
    def _generate_recreate_index_sql(self, index_info: Dict[str, Any], database_type: str) -> str:
        """根据索引信息生成重新创建索引的SQL."""
        if not index_info:
            return "-- 无法根据空索引信息生成重建SQL"

        db_type = database_type.lower()

        # MySQL: index_info 为 information_schema.statistics 的多行
        if db_type == "mysql":
            rows: List[Dict[str, Any]]
            if isinstance(index_info, list):  # type: ignore[assignment]
                rows = index_info  # type: ignore[assignment]
            else:
                rows = [index_info]

            if not rows:
                return "-- 无索引行信息, 无法生成重建SQL"

            # 按约定: 所有行属于同一个 (table_name, index_name)
            first = rows[0]
            table_name = first.get("table_name") or first.get("TABLE_NAME")
            index_name = first.get("index_name") or first.get("INDEX_NAME")
            non_unique = first.get("non_unique") if "non_unique" in first else first.get("NON_UNIQUE")
            index_type = first.get("index_type") or first.get("INDEX_TYPE") or "BTREE"

            if not table_name or not index_name:
                return "-- 索引信息缺少表名或索引名, 无法生成重建SQL"

            if str(index_name).upper() == "PRIMARY":
                return "-- PRIMARY KEY 由表结构维护, 不在这里重建"

            columns: List[str] = []
            for r in rows:
                col = r.get("column_name") or r.get("COLUMN_NAME")
                if col:
                    columns.append(str(col))

            if not columns:
                return f"-- 未找到索引 {index_name} 的列信息, 无法生成重建SQL"

            is_unique = str(non_unique or "1") == "0"
            index_type = str(index_type or "BTREE").upper()
            col_list = ", ".join(f"`{c}`" for c in columns)

            sql = "CREATE"
            if is_unique:
                sql += " UNIQUE"
            sql += f" INDEX `{index_name}` ON `{table_name}` ({col_list})"
            if index_type and index_type != "BTREE":
                sql += f" USING {index_type}"

            return sql

        # PostgreSQL: index_info 为 pg_indexes 的单行, 直接使用 indexdef
        if db_type == "postgresql":
            if isinstance(index_info, list) and index_info:
                row = index_info[0]
            else:
                row = index_info
            indexdef = row.get("indexdef") or row.get("INDEXDEF")
            if not indexdef:
                return "-- 未找到 PostgreSQL 索引定义(indexdef), 无法生成重建SQL"
            return str(indexdef)

        return f"-- 当前不支持的数据库类型: {database_type}, 无法生成索引重建SQL"
    
    async def _assess_index_performance_impact(self, connector: BaseDatabaseConnector, index_spec: IndexSpec) -> Dict[str, Any]:
        """评估索引性能影响."""
        db_type = connector.database_type.lower()

        impact: Dict[str, Any] = {
            "index_size_estimate_bytes": None,
            "row_count_estimate": None,
            "query_performance_improvement": "unknown",
            "maintenance_overhead": "unknown",
        }

        try:
            if db_type == "mysql":
                # 使用 SHOW TABLE STATUS 获取大致行数和数据/索引大小
                status_sql = f"SHOW TABLE STATUS LIKE '{index_spec.table_name}'"
                rows = await connector.execute_query(status_sql)
                if rows:
                    r = rows[0]
                    data_length = int(r.get("Data_length", 0) or 0)
                    index_length = int(r.get("Index_length", 0) or 0)
                    row_count = int(r.get("Rows", 0) or 0)
                    impact["row_count_estimate"] = row_count

                    # 粗略估计: 新索引大小 ~ 行数 * 每行每列约 64 字节
                    per_row_per_col = 64
                    est_size = row_count * per_row_per_col * max(len(index_spec.columns), 1)
                    # 不让估计值小于当前 index_length 的 5%
                    min_extra = int(index_length * 0.05) if index_length else 0
                    if est_size < min_extra:
                        est_size = min_extra or est_size
                    impact["index_size_estimate_bytes"] = est_size

                    # 粗略评估查询收益: 行数越多, 新索引带来的收益越高
                    if row_count > 1_000_000:
                        impact["query_performance_improvement"] = "high"
                    elif row_count > 100_000:
                        impact["query_performance_improvement"] = "medium"
                    else:
                        impact["query_performance_improvement"] = "low"

                    # 维护成本: 列数越多、表越大, 成本越高
                    if len(index_spec.columns) >= 3 or row_count > 1_000_000:
                        impact["maintenance_overhead"] = "high"
                    elif len(index_spec.columns) == 2 or row_count > 100_000:
                        impact["maintenance_overhead"] = "medium"
                    else:
                        impact["maintenance_overhead"] = "low"

            elif db_type == "postgresql":
                # 使用 pg_catalog.pg_class 获取行数与表大小估计
                sql = (
                    "SELECT reltuples::bigint AS row_estimate, "
                    "pg_relation_size(relid) AS table_bytes "
                    "FROM pg_catalog.pg_statio_user_tables "
                    f"WHERE relname = '{index_spec.table_name}' LIMIT 1"
                )
                rows = await connector.execute_query(sql)
                if rows:
                    r = rows[0]
                    row_count = int(r.get("row_estimate", 0) or 0)
                    table_bytes = int(r.get("table_bytes", 0) or 0)
                    impact["row_count_estimate"] = row_count

                    per_row_per_col = 64
                    est_size = row_count * per_row_per_col * max(len(index_spec.columns), 1)
                    if est_size > table_bytes:
                        est_size = int(table_bytes * 0.75) if table_bytes else est_size
                    impact["index_size_estimate_bytes"] = est_size

                    if row_count > 1_000_000:
                        impact["query_performance_improvement"] = "high"
                    elif row_count > 100_000:
                        impact["query_performance_improvement"] = "medium"
                    else:
                        impact["query_performance_improvement"] = "low"

                    if len(index_spec.columns) >= 3 or row_count > 1_000_000:
                        impact["maintenance_overhead"] = "high"
                    elif len(index_spec.columns) == 2 or row_count > 100_000:
                        impact["maintenance_overhead"] = "medium"
                    else:
                        impact["maintenance_overhead"] = "low"

        except Exception as e:  # noqa: BLE001
            logger.warning("评估索引性能影响时出错: %s", e)

        return impact
    
    async def _assess_statistics_performance_impact(self, connector: BaseDatabaseConnector, tables: List[str]) -> Dict[str, Any]:
        """评估统计信息更新的性能影响."""
        db_type = connector.database_type.lower()

        table_rows: Dict[str, int] = {}
        try:
            for table in tables:
                if db_type == "mysql":
                    sql = f"SELECT TABLE_ROWS as row_count FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = '{table}'"
                elif db_type == "postgresql":
                    sql = (
                        "SELECT reltuples::bigint AS row_count FROM pg_class "
                        f"WHERE relname = '{table}' LIMIT 1"
                    )
                else:
                    continue

                try:
                    rows = await connector.execute_query(sql)
                    if rows:
                        rc = rows[0].get("row_count") or rows[0].get("TABLE_ROWS")
                        if rc is not None:
                            table_rows[table] = int(rc)
                except Exception:
                    continue
        except Exception as e:  # noqa: BLE001
            logger.warning("统计信息性能影响评估中获取行数失败: %s", e)

        total_rows = sum(table_rows.values()) if table_rows else None

        return {
            "tables_updated": len(tables),
            "tables_row_estimates": table_rows,
            "total_row_estimate": total_rows,
            "query_plan_improvements": "expected",  # 语义化提示
            "update_duration": "proportional_to_total_rows",
        }

    async def _get_mysql_table_columns(
        self,
        connector: BaseDatabaseConnector,
        table_name: str,
    ) -> List[str]:
        """获取 MySQL 表的列名列表, 使用简单缓存."""

        db_type = getattr(connector, "database_type", "").lower()
        if db_type != "mysql":
            return []

        cache_key = (db_type, table_name)
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        sql = (
            "SELECT COLUMN_NAME FROM information_schema.columns "
            "WHERE table_schema = DATABASE() "
            f"AND table_name = '{table_name}' "
            "ORDER BY ORDINAL_POSITION"
        )
        try:
            rows = await connector.execute_query(sql)
            cols = [row["COLUMN_NAME"] for row in rows if "COLUMN_NAME" in row]
            self._schema_cache[cache_key] = cols
            return cols
        except Exception as e:  # noqa: BLE001
            logger.warning("获取表列信息失败: %s", e)
            return []

    async def _analyze_query_structure(
        self,
        sql: str,
        connector: BaseDatabaseConnector,
    ) -> Dict[str, Any]:
        """分析查询结构, 并尝试基于schema解析 SELECT * (当前仅 MySQL)."""

        import re

        analysis: Dict[str, Any] = {
            "has_subquery": "IN (" in sql.upper(),
            "has_exists": "EXISTS" in sql.upper(),
            "has_union": "UNION" in sql.upper(),
            "has_join": "JOIN" in sql.upper(),
        }

        select_star_columns: Dict[str, List[str]] = {}
        try:
            db_type = getattr(connector, "database_type", "").lower()
            if db_type == "mysql":
                # 简单匹配: SELECT * FROM table_name ...
                m = re.search(
                    r"select\s*\*\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                    sql,
                    re.IGNORECASE,
                )
                if m:
                    table_name = m.group(1)
                    cols = await self._get_mysql_table_columns(connector, table_name)
                    if cols:
                        select_star_columns[table_name] = cols
        except Exception as e:  # noqa: BLE001
            logger.warning("解析 SELECT * schema 信息失败: %s", e)

        if select_star_columns:
            analysis["select_star_columns"] = select_star_columns

        return analysis

    async def _apply_rewrite_rule(
        self,
        sql: str,
        rule_name: str,
        rule_config: Dict[str, Any],
        query_analysis: Dict[str, Any],
    ) -> Optional[QueryRewriteSuggestion]:
        """应用查询重写规则."""
        import re

        pattern = rule_config.get("pattern", "")
        if not pattern or not re.search(pattern, sql, re.IGNORECASE):
            return None

        # 1) 子查询 IN (...) → EXISTS(...) (示例性, 可参与候选)
        if rule_name == "subquery_to_join" and query_analysis.get("has_subquery"):
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql.replace("IN (SELECT", "EXISTS (SELECT"),
                improvement_type="subquery_optimization",
                expected_improvement="将IN子查询改为EXISTS可能提升性能(需人工确认语义等价)",
                confidence=0.7,
                explanation="检测到 IN 子查询, 已示例性改写为 EXISTS, 请确认语义等价后再采用。",
                rule_name=rule_name,
            )

        # 2) EXISTS 提示 + 示例 JOIN 重写 (不参与候选)
        if rule_name == "exists_to_join" and query_analysis.get("has_exists"):
            example = (
                "示例:\n"
                "原始:\n"
                "  SELECT ... FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.x = t1.x AND 条件)\n"
                "可以考虑改写为:\n"
                "  SELECT DISTINCT t1.* FROM t1 JOIN t2 ON t2.x = t1.x AND 条件\n"
            )
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql,
                improvement_type="exists_join_hint",
                expected_improvement="在部分场景下将 EXISTS 改写为 JOIN 可获得更好的执行计划",
                confidence=0.5,
                explanation=(
                    "检测到 EXISTS 子查询, 建议在确认语义等价的前提下改写为 JOIN 以提升可读性和潜在性能。\n"
                    + example
                ),
                rule_name=rule_name,
            )

        # 3) UNION → UNION ALL 提示 (不参与候选)
        if rule_name == "union_to_union_all" and query_analysis.get("has_union"):
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql,
                improvement_type="union_all_hint",
                expected_improvement="在结果无需去重时, 使用 UNION ALL 可避免不必要的去重开销",
                confidence=0.5,
                explanation="检测到 UNION, 如结果不需要去重, 建议手动改为 UNION ALL 以降低排序/去重成本。",
                rule_name=rule_name,
            )

        # 4) schema 感知的 SELECT * → 显式列名 (MySQL), 否则仅提示
        if rule_name == "select_star_to_columns":
            cols_map = query_analysis.get("select_star_columns") or {}
            rewritten_sql = sql

            if cols_map:
                # 只处理第一个匹配的表
                table_name, columns = next(iter(cols_map.items()))
                if columns:
                    column_list = ", ".join(columns)
                    try:
                        rewritten_sql = re.sub(
                            r"SELECT\s*\*\s*FROM",
                            f"SELECT {column_list} FROM",
                            sql,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                    except re.error:  # noqa: BLE001
                        rewritten_sql = sql

                    return QueryRewriteSuggestion(
                        original_query=sql,
                        rewritten_query=rewritten_sql,
                        improvement_type="select_star_rewrite",
                        expected_improvement="显式列出字段可减少不必要的数据扫描和传输",
                        confidence=0.8,
                        explanation=(
                            f"检测到 SELECT *, 已基于 information_schema 将表 {table_name} 的所有字段替换为显式字段列表, "
                            "请确认列集是否符合业务需求后采用。"
                        ),
                        rule_name=rule_name,
                    )

            # 没有 schema 信息时, 退回提示型建议
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql,
                improvement_type="select_star_warning",
                expected_improvement="显式列出需要的字段可减少不必要的数据传输和扫描",
                confidence=0.6,
                explanation="检测到 SELECT *, 建议将 * 替换为实际需要的字段列表, 以便优化索引使用和减少传输量。",
                rule_name=rule_name,
            )

        # 5) 单表多 OR 条件 → IN 列表 (简化版, 作为候选)
        if rule_name == "or_to_in_list":
            where_match = re.search(r"(WHERE\s+)(.+)", sql, re.IGNORECASE | re.DOTALL)
            if not where_match:
                return None

            where_clause = where_match.group(2)
            pattern_eq = re.compile(
                r"([a-zA-Z_][a-zA-Z0-9_\.]*)\s*=\s*([^\s\)]+)",
                re.IGNORECASE,
            )
            matches = pattern_eq.findall(where_clause)
            if not matches:
                return None

            col_values: Dict[str, List[str]] = {}
            for col, val in matches:
                col_values.setdefault(col, []).append(val)

            target_col = None
            values: List[str] = []
            for col, vs in col_values.items():
                if len(vs) >= 3:
                    target_col = col
                    values = vs
                    break

            if not target_col:
                return None

            in_list = f"{target_col} IN ({', '.join(values)})"

            or_pattern = re.compile(
                rf"{re.escape(target_col)}\s*=\s*[^\s\)]+(\s+OR\s+{re.escape(target_col)}\s*=\s*[^\s\)]+)+",
                re.IGNORECASE,
            )
            new_where_clause, count = or_pattern.subn(in_list, where_clause, count=1)
            if count == 0:
                return None

            rewritten_sql = sql[: where_match.start(2)] + new_where_clause

            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=rewritten_sql,
                improvement_type="or_to_in_list",
                expected_improvement="将多次 OR 等值比较转换为 IN 列表, 可使优化器更容易利用索引",
                confidence=0.6,
                explanation="检测到同一列上的多 OR 等值条件, 已尝试重写为 IN 列表, 请确认语义等价后采用。",
                rule_name=rule_name,
            )

        # 6) FROM t1, t2 → 显式 JOIN (提示型)
        if rule_name == "implicit_join_to_explicit":
            example = (
                "示例:\n"
                "原始:\n"
                "  SELECT ... FROM t1, t2 WHERE t1.id = t2.id AND ...\n"
                "可以考虑改写为:\n"
                "  SELECT ... FROM t1 JOIN t2 ON t1.id = t2.id WHERE ...\n"
            )
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql,
                improvement_type="implicit_join_hint",
                expected_improvement="显式 JOIN 语法更清晰, 也更容易让优化器选择合适的计划",
                confidence=0.5,
                explanation="检测到 FROM t1, t2 形式的隐式连接, 建议改写为显式 JOIN 语法。\n" + example,
                rule_name=rule_name,
            )

        return None
    
    async def _generate_execution_plan_based_suggestions(self, sql: str, connector: BaseDatabaseConnector) -> List[QueryRewriteSuggestion]:
        """基于执行计划生成重写建议."""
        suggestions = []
        
        try:
            # 获取执行计划
            explain_results = await connector.execute_explain(sql)
            
            # 基于执行计划分析生成建议
            for result in explain_results:
                # 当前实现聚焦 MySQL 的全表扫描(type = ALL)场景, 尝试根据 possible_keys 生成 FORCE INDEX 提示性重写
                if hasattr(result, "type") and str(getattr(result, "type", "")).upper() == "ALL":
                    table = getattr(result, "table", None)
                    possible_keys = getattr(result, "possible_keys", None)

                    rewritten_sql = sql
                    explanation = "检测到全表扫描, 建议为过滤条件列创建索引或优化WHERE条件。"

                    if table and possible_keys:
                        # possible_keys 形如 'idx_a,idx_b'
                        first_key = str(possible_keys).split(",")[0].strip()
                        if first_key:
                            import re

                            pattern = rf"FROM\s+{re.escape(str(table))}\b"
                            replacement = f"FROM {table} FORCE INDEX ({first_key})"
                            try:
                                rewritten_sql, count = re.subn(
                                    pattern,
                                    replacement,
                                    sql,
                                    count=1,
                                    flags=re.IGNORECASE,
                                )
                                if count > 0:
                                    explanation = (
                                        "检测到表 {table} 上的全表扫描(type=ALL), 且存在可用索引 {idx}, "
                                        "已示例性将 FROM 子句改写为使用 FORCE INDEX 提示, 请在确认索引选择合理后再采用。"
                                    ).format(table=table, idx=first_key)
                            except re.error:  # noqa: BLE001
                                rewritten_sql = sql

                    suggestions.append(
                        QueryRewriteSuggestion(
                            original_query=sql,
                            rewritten_query=rewritten_sql,
                            improvement_type="full_table_scan_optimization",
                            expected_improvement="避免或减轻全表扫描开销",
                            confidence=0.8,
                            explanation=explanation,
                            rule_name="execution_plan_full_table_scan",
                        )
                    )
        
        except Exception as e:
            logger.warning(f"生成基于执行计划的建议失败: {e}")
        
        return suggestions
    
    async def _get_current_configuration(self, connector: BaseDatabaseConnector) -> Dict[str, Any]:
        """获取当前数据库配置."""
        config = {}
        
        try:
            if connector.database_type.lower() == "mysql":
                result = await connector.execute_query("SHOW VARIABLES")
                config = {row["Variable_name"]: row["Value"] for row in result}
            elif connector.database_type.lower() == "postgresql":
                result = await connector.execute_query("SELECT name, setting FROM pg_settings")
                config = {row["name"]: row["setting"] for row in result}
        
        except Exception as e:
            logger.warning(f"获取数据库配置失败: {e}")
        
        return config
    
    async def _analyze_system_resources(self, connector: BaseDatabaseConnector) -> Dict[str, Any]:
        """分析系统资源."""
        resources: Dict[str, Any] = {
            "total_memory_bytes": None,
            "available_memory_bytes": None,
            "cpu_cores": None,
            "disk_total_bytes": None,
            "disk_free_bytes": None,
        }

        # 1. 尝试使用 psutil 获取主机资源信息
        try:
            if psutil is not None:
                vm = psutil.virtual_memory()
                resources["total_memory_bytes"] = int(vm.total)
                resources["available_memory_bytes"] = int(vm.available)
                resources["cpu_cores"] = int(psutil.cpu_count(logical=True) or 0)

                try:
                    du = psutil.disk_usage("/")
                    resources["disk_total_bytes"] = int(du.total)
                    resources["disk_free_bytes"] = int(du.free)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            logger.warning("通过 psutil 获取系统资源失败: %s", e)

        # 2. 可选: 结合数据库级别的资源配置进行补充
        try:
            if connector.database_type.lower() == "mysql":
                # 示例: 记录 innodb_buffer_pool_size 作为内存利用的一个参考
                rows = await connector.execute_query(
                    "SHOW VARIABLES LIKE 'innodb_buffer_pool_size'"
                )
                if rows:
                    resources["innodb_buffer_pool_size"] = int(rows[0].get("Value", 0) or 0)
            elif connector.database_type.lower() == "postgresql":
                rows = await connector.execute_query(
                    "SELECT setting::bigint AS shared_buffers_bytes FROM pg_settings WHERE name = 'shared_buffers'"
                )
                if rows:
                    resources["shared_buffers_bytes"] = int(
                        rows[0].get("shared_buffers_bytes", 0) or 0
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("获取数据库资源配置失败: %s", e)

        return resources
    
    def _generate_config_recommendations(self, current_config: Dict[str, Any], system_resources: Dict[str, Any], database_type: str) -> List[ConfigParameter]:
        """生成配置建议."""
        recommendations = []

        def _parse_size(value: Any) -> Optional[int]:
            """将类似 '128M' / '1G' / '4096MB' 或纯数字转换为字节数."""
            if value is None:
                return None
            try:
                # 已经是纯数字字符串或整数
                if isinstance(value, (int, float)):
                    return int(value)
                s = str(value).strip().upper()
                if s.isdigit():
                    return int(s)
                multipliers = {
                    "K": 1024,
                    "KB": 1024,
                    "M": 1024**2,
                    "MB": 1024**2,
                    "G": 1024**3,
                    "GB": 1024**3,
                }
                for suffix, mul in multipliers.items():
                    if s.endswith(suffix):
                        num = float(s[: -len(suffix)].strip())
                        return int(num * mul)
            except Exception:  # noqa: BLE001
                return None
            return None

        def _format_size(num_bytes: int) -> str:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if num_bytes < 1024 or unit == "TB":
                    return f"{num_bytes:.0f}{unit}" if unit != "B" else str(num_bytes)
                num_bytes /= 1024
            return str(num_bytes)

        total_mem = system_resources.get("total_memory_bytes")
        if not isinstance(total_mem, int) or total_mem <= 0:
            # 无法获取系统内存时, 不做激进调参, 避免误伤
            logger.warning("系统总内存未知, 跳过自动配置推荐")
            return recommendations

        db_type = database_type.lower()

        if db_type == "mysql":
            # 基于预定义规则和系统内存生成建议
            rules = self.config_tuning_rules.get("mysql", {})
            for param_name, rule in rules.items():
                if param_name not in current_config:
                    continue

                calc_expr = rule.get("calculation", "")  # 如 "memory * 0.7"
                factor = 0.0
                if "*" in calc_expr:
                    try:
                        factor = float(calc_expr.split("*")[-1].strip())
                    except Exception:  # noqa: BLE001
                        factor = 0.0

                if factor <= 0:
                    continue

                target_bytes = int(total_mem * factor)

                min_value = rule.get("min_value")
                max_value = rule.get("max_value")
                if min_value:
                    min_bytes = _parse_size(min_value) or 0
                    if target_bytes < min_bytes:
                        target_bytes = min_bytes
                if max_value:
                    max_bytes = _parse_size(max_value) or 0
                    if max_bytes and target_bytes > max_bytes:
                        target_bytes = max_bytes

                current_bytes = _parse_size(current_config.get(param_name))

                # 建议与当前值相差不明显则不建议调整
                if current_bytes and abs(target_bytes - current_bytes) < current_bytes * 0.1:
                    continue

                recommendations.append(
                    ConfigParameter(
                        parameter_name=param_name,
                        current_value=current_config.get(param_name),
                        recommended_value=_format_size(target_bytes),
                        parameter_type="memory",
                        description=rule.get("description", ""),
                        impact_level="high" if factor >= 0.5 else "medium",
                        requires_restart=True,
                    )
                )

        elif db_type == "postgresql":
            rules = self.config_tuning_rules.get("postgresql", {})
            for param_name, rule in rules.items():
                if param_name not in current_config:
                    continue

                calc_expr = rule.get("calculation", "")
                factor = 0.0
                if "*" in calc_expr:
                    try:
                        factor = float(calc_expr.split("*")[-1].strip())
                    except Exception:  # noqa: BLE001
                        factor = 0.0

                if factor <= 0:
                    continue

                target_bytes = int(total_mem * factor)

                min_value = rule.get("min_value")
                if min_value:
                    min_bytes = _parse_size(min_value) or 0
                    if target_bytes < min_bytes:
                        target_bytes = min_bytes

                current_bytes = _parse_size(current_config.get(param_name))
                if current_bytes and abs(target_bytes - current_bytes) < current_bytes * 0.1:
                    continue

                recommendations.append(
                    ConfigParameter(
                        parameter_name=param_name,
                        current_value=current_config.get(param_name),
                        recommended_value=_format_size(target_bytes),
                        parameter_type="memory",
                        description=rule.get("description", ""),
                        impact_level="high" if factor >= 0.5 else "medium",
                        requires_restart=True,
                    )
                )

        return recommendations
    
    def _generate_config_update_sql(self, recommendations: List[ConfigParameter], database_type: str) -> List[str]:
        """生成配置更新SQL."""
        sqls = []
        
        for rec in recommendations:
            if database_type.lower() == "mysql":
                sqls.append(f"SET GLOBAL {rec.parameter_name} = '{rec.recommended_value}'")
            elif database_type.lower() == "postgresql":
                sqls.append(f"ALTER SYSTEM SET {rec.parameter_name} = '{rec.recommended_value}'")
        
        return sqls
    
    def _generate_config_rollback_sql(self, current_config: Dict[str, Any], recommendations: List[ConfigParameter], database_type: str) -> List[str]:
        """生成配置回滚SQL."""
        sqls = []
        
        for rec in recommendations:
            if database_type.lower() == "mysql":
                sqls.append(f"SET GLOBAL {rec.parameter_name} = '{rec.current_value}'")
            elif database_type.lower() == "postgresql":
                sqls.append(f"ALTER SYSTEM SET {rec.parameter_name} = '{rec.current_value}'")
        
        return sqls
    
    async def _validate_batch_plan(self, plan: BatchOptimizationPlan, user: User) -> ValidationResult:
        """验证批量优化计划."""
        # 使用安全验证器验证整个计划
        from .safety_validator import OptimizationPlan
        
        optimization_plan = OptimizationPlan(
            database=plan.database,
            operations=plan.optimizations,
            estimated_impact=f"批量优化，包含{len(plan.optimizations)}个操作",
            rollback_plan=plan.rollback_plan
        )
        
        return await self.safety_validator.validate_optimization_plan(optimization_plan, user)
    
    async def _execute_single_optimization(self, optimization_config: Dict[str, Any], connector: BaseDatabaseConnector, user: User) -> OptimizationResult:
        """执行单个优化操作."""
        optimization_type = optimization_config.get("type")
        
        if optimization_type == "index_creation":
            index_spec = IndexSpec(**optimization_config["spec"])
            return await self.create_index(connector, index_spec, user)
        elif optimization_type == "statistics_update":
            tables = optimization_config.get("tables", [])
            return await self.update_statistics(connector, tables, user)
        elif optimization_type == "config_tuning":
            # 配置调优：直接调用 optimize_configuration
            return await self.optimize_configuration(connector, user)
        elif optimization_type == "index_deletion":
            # 索引删除：需要提供 table_name 和 index_name
            table_name = optimization_config.get("table_name")
            index_name = optimization_config.get("index_name")
            if not table_name or not index_name:
                return OptimizationResult(
                    optimization_type=OptimizationType.BATCH_OPTIMIZATION,
                    status=OptimizationStatus.FAILED,
                    database=connector.config.database,
                    target_object=optimization_config.get("id", "unknown"),
                    execution_time=0.0,
                    error_message="index_deletion 需要 table_name 和 index_name",
                )
            return await self.drop_index(connector, index_name, table_name, user)
        elif optimization_type == "query_rewrite":
            # 查询重写：只返回评估结果，不执行变更
            sql = optimization_config.get("sql")
            if not sql:
                return OptimizationResult(
                    optimization_type=OptimizationType.BATCH_OPTIMIZATION,
                    status=OptimizationStatus.FAILED,
                    database=connector.config.database,
                    target_object=optimization_config.get("id", "unknown"),
                    execution_time=0.0,
                    error_message="query_rewrite 需要 sql 字段",
                )

            start_time = datetime.now()
            result_dict = await self.optimize_query(sql, connector, user, auto_apply=False)
            completed_at = datetime.now()

            return OptimizationResult(
                optimization_type=OptimizationType.QUERY_REWRITE,
                status=OptimizationStatus.COMPLETED,
                database=connector.config.database,
                target_object=optimization_config.get("id", "unknown"),
                executed_sql=[],
                execution_time=(completed_at - start_time).total_seconds(),
                performance_impact={
                    "original_cost": result_dict.get("original_cost"),
                    "best_cost": result_dict.get("best_cost"),
                    "improvement": result_dict.get("improvement"),
                },
            )
        else:
            # 创建失败结果
            return OptimizationResult(
                optimization_type=OptimizationType.BATCH_OPTIMIZATION,
                status=OptimizationStatus.FAILED,
                database=connector.config.database,
                target_object=optimization_config.get("id", "unknown"),
                execution_time=0.0,
                error_message=f"不支持的优化类型: {optimization_type}"
            )