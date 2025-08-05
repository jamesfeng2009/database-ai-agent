"""自动优化器模块 - 提供数据库自动化优化功能."""

import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ...database.connector_base import BaseDatabaseConnector
from ...database.adapters import DatabaseAdapter, DatabaseAdapterFactory
from .safety_validator import SafetyValidator, User, ValidationResult, RiskLevel
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
    
    def __init__(self, safety_validator: SafetyValidator):
        """初始化自动优化器."""
        self.safety_validator = safety_validator
        self.optimization_history: List[OptimizationResult] = []
        self.active_optimizations: Dict[str, OptimizationResult] = {}
        
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
                "description": "将EXISTS转换为JOIN"
            },
            "union_to_union_all": {
                "pattern": r"UNION(?!\s+ALL)",
                "description": "将UNION转换为UNION ALL（如果适用）"
            }
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
        """获取索引信息."""
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
            return result[0] if result else None
        except Exception:
            return None
    
    def _generate_recreate_index_sql(self, index_info: Dict[str, Any], database_type: str) -> str:
        """根据索引信息生成重新创建索引的SQL."""
        # 这里需要根据具体的索引信息格式来实现
        # 简化实现，实际应该解析索引信息
        return f"-- 重新创建索引的SQL需要根据具体索引信息实现"
    
    async def _assess_index_performance_impact(self, connector: BaseDatabaseConnector, index_spec: IndexSpec) -> Dict[str, Any]:
        """评估索引性能影响."""
        return {
            "index_size_estimate": "unknown",
            "query_performance_improvement": "estimated_medium",
            "maintenance_overhead": "low"
        }
    
    async def _assess_statistics_performance_impact(self, connector: BaseDatabaseConnector, tables: List[str]) -> Dict[str, Any]:
        """评估统计信息更新的性能影响."""
        return {
            "tables_updated": len(tables),
            "query_plan_improvements": "expected",
            "update_duration": "varies_by_table_size"
        }
    
    async def _analyze_query_structure(self, sql: str, connector: BaseDatabaseConnector) -> Dict[str, Any]:
        """分析查询结构."""
        return {
            "has_subquery": "IN (" in sql.upper(),
            "has_exists": "EXISTS" in sql.upper(),
            "has_union": "UNION" in sql.upper(),
            "has_join": "JOIN" in sql.upper()
        }
    
    async def _apply_rewrite_rule(self, sql: str, rule_name: str, rule_config: Dict[str, Any], query_analysis: Dict[str, Any]) -> Optional[QueryRewriteSuggestion]:
        """应用查询重写规则."""
        import re
        
        pattern = rule_config.get("pattern", "")
        if not pattern or not re.search(pattern, sql, re.IGNORECASE):
            return None
        
        # 简化的重写逻辑，实际应该更复杂
        if rule_name == "subquery_to_join" and query_analysis.get("has_subquery"):
            return QueryRewriteSuggestion(
                original_query=sql,
                rewritten_query=sql.replace("IN (SELECT", "EXISTS (SELECT"),  # 简化示例
                improvement_type="subquery_optimization",
                expected_improvement="可能提升查询性能",
                confidence=0.7,
                explanation="将IN子查询转换为EXISTS可能提升性能"
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
                if hasattr(result, 'type') and result.type == "ALL":
                    suggestions.append(QueryRewriteSuggestion(
                        original_query=sql,
                        rewritten_query=sql,  # 实际应该生成优化后的查询
                        improvement_type="full_table_scan_optimization",
                        expected_improvement="避免全表扫描",
                        confidence=0.8,
                        explanation="检测到全表扫描，建议添加索引或优化WHERE条件"
                    ))
        
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
        return {
            "total_memory": "unknown",
            "available_memory": "unknown",
            "cpu_cores": "unknown",
            "disk_space": "unknown"
        }
    
    def _generate_config_recommendations(self, current_config: Dict[str, Any], system_resources: Dict[str, Any], database_type: str) -> List[ConfigParameter]:
        """生成配置建议."""
        recommendations = []
        
        # 简化的配置建议逻辑
        if database_type.lower() == "mysql":
            if "innodb_buffer_pool_size" in current_config:
                recommendations.append(ConfigParameter(
                    parameter_name="innodb_buffer_pool_size",
                    current_value=current_config["innodb_buffer_pool_size"],
                    recommended_value="1G",  # 实际应该基于系统内存计算
                    parameter_type="memory",
                    description="InnoDB缓冲池大小",
                    impact_level="high",
                    requires_restart=True
                ))
        
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