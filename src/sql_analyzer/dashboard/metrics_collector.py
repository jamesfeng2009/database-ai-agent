"""性能指标收集器."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..database.database_manager import DatabaseManager
from ..database.models import DatabaseType
from .models import PerformanceMetrics, MetricDefinition, MetricType, MetricUnit

logger = logging.getLogger(__name__)


class MetricsCollector:
    """性能指标收集器 - 从多个数据库收集性能指标."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.metric_definitions = {}
        self.collection_tasks = {}
        self.metrics_storage = {}  # 简单的内存存储，实际应用中应使用时序数据库
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """初始化默认指标定义."""
        default_metrics = [
            MetricDefinition(
                metric_id="cpu_usage",
                name="CPU使用率",
                description="数据库服务器CPU使用率",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=30
            ),
            MetricDefinition(
                metric_id="memory_usage",
                name="内存使用率",
                description="数据库服务器内存使用率",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=30
            ),
            MetricDefinition(
                metric_id="connections_active",
                name="活跃连接数",
                description="当前活跃的数据库连接数",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.COUNT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=15
            ),
            MetricDefinition(
                metric_id="queries_per_second",
                name="每秒查询数",
                description="数据库每秒处理的查询数量",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.RATE,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=15
            ),
            MetricDefinition(
                metric_id="slow_queries",
                name="慢查询数量",
                description="慢查询的数量",
                metric_type=MetricType.COUNTER,
                unit=MetricUnit.COUNT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=60
            ),
            MetricDefinition(
                metric_id="response_time_avg",
                name="平均响应时间",
                description="查询的平均响应时间",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.MILLISECONDS,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=30
            ),
            MetricDefinition(
                metric_id="buffer_hit_ratio",
                name="缓冲区命中率",
                description="缓冲区的命中率",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=60
            ),
            MetricDefinition(
                metric_id="lock_waits",
                name="锁等待数",
                description="当前锁等待的数量",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.COUNT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=30
            ),
            MetricDefinition(
                metric_id="deadlocks",
                name="死锁数",
                description="死锁的数量",
                metric_type=MetricType.COUNTER,
                unit=MetricUnit.COUNT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=60
            ),
            MetricDefinition(
                metric_id="disk_usage",
                name="磁盘使用率",
                description="数据库磁盘使用率",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.PERCENT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=300
            ),
            MetricDefinition(
                metric_id="response_time_p95",
                name="95%响应时间",
                description="95%分位数响应时间",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.MILLISECONDS,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=60
            ),
            MetricDefinition(
                metric_id="response_time_p99",
                name="99%响应时间",
                description="99%分位数响应时间",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.MILLISECONDS,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=60
            ),
            MetricDefinition(
                metric_id="connections_total",
                name="总连接数",
                description="数据库总连接数",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.COUNT,
                database_types=["mysql", "postgresql", "oracle", "sqlserver"],
                collection_interval=30
            )
        ]
        
        for metric in default_metrics:
            self.metric_definitions[metric.metric_id] = metric
    
    def register_metric(self, metric_definition: MetricDefinition):
        """注册新的指标定义."""
        self.metric_definitions[metric_definition.metric_id] = metric_definition
        logger.info(f"已注册指标: {metric_definition.name}")
    
    def get_metric_definition(self, metric_id: str) -> Optional[MetricDefinition]:
        """获取指标定义."""
        return self.metric_definitions.get(metric_id)
    
    def list_metrics(self, database_type: Optional[str] = None) -> List[MetricDefinition]:
        """列出指标定义."""
        if database_type:
            return [
                metric for metric in self.metric_definitions.values()
                if database_type in metric.database_types
            ]
        return list(self.metric_definitions.values())
    
    async def start_collection(self, database_ids: Optional[List[str]] = None):
        """启动指标收集."""
        if not database_ids:
            # 获取所有健康的数据库连接
            database_ids = self.database_manager.get_healthy_connections()
        
        for database_id in database_ids:
            if database_id not in self.collection_tasks:
                task = asyncio.create_task(self._collect_metrics_loop(database_id))
                self.collection_tasks[database_id] = task
                logger.info(f"已启动数据库 {database_id} 的指标收集")
    
    async def stop_collection(self, database_ids: Optional[List[str]] = None):
        """停止指标收集."""
        if not database_ids:
            database_ids = list(self.collection_tasks.keys())
        
        for database_id in database_ids:
            if database_id in self.collection_tasks:
                task = self.collection_tasks[database_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.collection_tasks[database_id]
                logger.info(f"已停止数据库 {database_id} 的指标收集")
    
    async def _collect_metrics_loop(self, database_id: str):
        """指标收集循环."""
        try:
            while True:
                try:
                    await self._collect_database_metrics(database_id)
                except Exception as e:
                    logger.error(f"收集数据库 {database_id} 指标失败: {e}")
                
                # 等待最小收集间隔
                await asyncio.sleep(15)  # 15秒的基础间隔
                
        except asyncio.CancelledError:
            logger.info(f"数据库 {database_id} 的指标收集已取消")
            raise
    
    async def _collect_database_metrics(self, database_id: str):
        """收集单个数据库的指标."""
        try:
            # 获取数据库配置
            config = self.database_manager.get_database_config(database_id)
            if not config:
                logger.warning(f"未找到数据库配置: {database_id}")
                return
            
            database_type = config.get_database_type().value
            
            # 获取适用的指标
            applicable_metrics = [
                metric for metric in self.metric_definitions.values()
                if database_type in metric.database_types
            ]
            
            # 收集指标数据
            metrics_data = {}
            
            for metric in applicable_metrics:
                try:
                    value = await self._collect_single_metric(database_id, database_type, metric)
                    if value is not None:
                        metrics_data[metric.metric_id] = value
                except Exception as e:
                    logger.error(f"收集指标 {metric.metric_id} 失败: {e}")
            
            # 创建性能指标对象
            performance_metrics = PerformanceMetrics(
                database_id=database_id,
                database_type=database_type,
                timestamp=datetime.now(),
                metrics=metrics_data,
                **{k: v for k, v in metrics_data.items() if hasattr(PerformanceMetrics, k)}
            )
            
            # 存储指标数据
            await self._store_metrics(performance_metrics)
            
        except Exception as e:
            logger.error(f"收集数据库 {database_id} 指标时发生错误: {e}")
    
    async def _collect_single_metric(self, database_id: str, database_type: str, metric: MetricDefinition) -> Optional[float]:
        """收集单个指标."""
        try:
            if database_type == "mysql":
                return await self._collect_mysql_metric(database_id, metric)
            elif database_type == "postgresql":
                return await self._collect_postgresql_metric(database_id, metric)
            elif database_type == "oracle":
                return await self._collect_oracle_metric(database_id, metric)
            elif database_type == "sqlserver":
                return await self._collect_sqlserver_metric(database_id, metric)
            else:
                logger.warning(f"不支持的数据库类型: {database_type}")
                return None
                
        except Exception as e:
            logger.error(f"收集指标 {metric.metric_id} 失败: {e}")
            return None
    
    async def _collect_mysql_metric(self, database_id: str, metric: MetricDefinition) -> Optional[float]:
        """收集MySQL指标."""
        metric_queries = {
            "connections_active": "SHOW STATUS LIKE 'Threads_connected'",
            "queries_per_second": "SHOW STATUS LIKE 'Queries'",
            "slow_queries": "SHOW STATUS LIKE 'Slow_queries'",
            "buffer_hit_ratio": """
                SELECT ROUND(
                    (1 - (Innodb_buffer_pool_reads / Innodb_buffer_pool_read_requests)) * 100, 2
                ) as hit_ratio
                FROM (
                    SELECT 
                        VARIABLE_VALUE as Innodb_buffer_pool_reads
                    FROM INFORMATION_SCHEMA.GLOBAL_STATUS 
                    WHERE VARIABLE_NAME = 'Innodb_buffer_pool_reads'
                ) a,
                (
                    SELECT 
                        VARIABLE_VALUE as Innodb_buffer_pool_read_requests
                    FROM INFORMATION_SCHEMA.GLOBAL_STATUS 
                    WHERE VARIABLE_NAME = 'Innodb_buffer_pool_read_requests'
                ) b
            """,
            "lock_waits": "SELECT COUNT(*) FROM INFORMATION_SCHEMA.INNODB_LOCKS",
            "deadlocks": "SHOW STATUS LIKE 'Innodb_deadlocks'"
        }
        
        query = metric_queries.get(metric.metric_id)
        if not query:
            # 对于系统级指标（CPU、内存等），返回模拟数据
            return await self._get_system_metric(metric.metric_id)
        
        try:
            result = await self.database_manager.execute_query(query, connection_id=database_id)
            if result and len(result) > 0:
                if metric.metric_id == "buffer_hit_ratio":
                    return float(result[0].get('hit_ratio', 0))
                else:
                    # 对于SHOW STATUS查询，结果格式为[{'Variable_name': 'xxx', 'Value': 'yyy'}]
                    return float(result[0].get('Value', 0))
            return 0.0
        except Exception as e:
            logger.error(f"执行MySQL查询失败: {e}")
            return None
    
    async def _collect_postgresql_metric(self, database_id: str, metric: MetricDefinition) -> Optional[float]:
        """收集PostgreSQL指标."""
        metric_queries = {
            "connections_active": "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'",
            "queries_per_second": "SELECT sum(calls) FROM pg_stat_statements",
            "buffer_hit_ratio": """
                SELECT ROUND(
                    (blks_hit::float / (blks_hit + blks_read)) * 100, 2
                ) as hit_ratio
                FROM pg_stat_database 
                WHERE datname = current_database()
            """,
            "lock_waits": "SELECT count(*) FROM pg_locks WHERE NOT granted",
            "deadlocks": "SELECT sum(deadlocks) FROM pg_stat_database"
        }
        
        query = metric_queries.get(metric.metric_id)
        if not query:
            return await self._get_system_metric(metric.metric_id)
        
        try:
            result = await self.database_manager.execute_query(query, connection_id=database_id)
            if result and len(result) > 0:
                # PostgreSQL查询结果通常是字典格式
                value = list(result[0].values())[0]
                return float(value) if value is not None else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"执行PostgreSQL查询失败: {e}")
            return None
    
    async def _collect_oracle_metric(self, database_id: str, metric: MetricDefinition) -> Optional[float]:
        """收集Oracle指标."""
        metric_queries = {
            "connections_active": "SELECT count(*) FROM v$session WHERE status = 'ACTIVE'",
            "buffer_hit_ratio": """
                SELECT ROUND(
                    (1 - (phy.value / (cur.value + con.value))) * 100, 2
                ) as hit_ratio
                FROM v$sysstat cur, v$sysstat con, v$sysstat phy
                WHERE cur.name = 'db block gets'
                AND con.name = 'consistent gets'
                AND phy.name = 'physical reads'
            """,
            "lock_waits": "SELECT count(*) FROM v$lock WHERE block > 0"
        }
        
        query = metric_queries.get(metric.metric_id)
        if not query:
            return await self._get_system_metric(metric.metric_id)
        
        try:
            result = await self.database_manager.execute_query(query, connection_id=database_id)
            if result and len(result) > 0:
                value = list(result[0].values())[0]
                return float(value) if value is not None else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"执行Oracle查询失败: {e}")
            return None
    
    async def _collect_sqlserver_metric(self, database_id: str, metric: MetricDefinition) -> Optional[float]:
        """收集SQL Server指标."""
        metric_queries = {
            "connections_active": """
                SELECT count(*) 
                FROM sys.dm_exec_sessions 
                WHERE is_user_process = 1 AND status = 'running'
            """,
            "buffer_hit_ratio": """
                SELECT 
                    (cntr_value * 100.0) / 
                    (SELECT cntr_value FROM sys.dm_os_performance_counters 
                     WHERE counter_name = 'Buffer cache hit ratio base') as hit_ratio
                FROM sys.dm_os_performance_counters 
                WHERE counter_name = 'Buffer cache hit ratio'
            """,
            "lock_waits": """
                SELECT count(*) 
                FROM sys.dm_tran_locks 
                WHERE request_status = 'WAIT'
            """
        }
        
        query = metric_queries.get(metric.metric_id)
        if not query:
            return await self._get_system_metric(metric.metric_id)
        
        try:
            result = await self.database_manager.execute_query(query, connection_id=database_id)
            if result and len(result) > 0:
                value = list(result[0].values())[0]
                return float(value) if value is not None else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"执行SQL Server查询失败: {e}")
            return None
    
    async def _get_system_metric(self, metric_id: str) -> Optional[float]:
        """获取系统级指标（模拟数据）."""
        import random
        
        # 在实际应用中，这里应该调用系统监控API获取真实数据
        system_metrics = {
            "cpu_usage": random.uniform(10, 80),
            "memory_usage": random.uniform(30, 90),
            "disk_usage": random.uniform(20, 70),
            "response_time_avg": random.uniform(10, 100),
            "response_time_p95": random.uniform(50, 200),
            "response_time_p99": random.uniform(100, 500)
        }
        
        return system_metrics.get(metric_id)
    
    async def _store_metrics(self, metrics: PerformanceMetrics):
        """存储指标数据."""
        # 简单的内存存储实现
        database_id = metrics.database_id
        if database_id not in self.metrics_storage:
            self.metrics_storage[database_id] = []
        
        self.metrics_storage[database_id].append(metrics)
        
        # 保持最近1000条记录
        if len(self.metrics_storage[database_id]) > 1000:
            self.metrics_storage[database_id] = self.metrics_storage[database_id][-1000:]
        
        logger.debug(f"已存储数据库 {database_id} 的指标数据")
    
    async def get_metrics(
        self,
        database_ids: Optional[List[str]] = None,
        metric_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """获取指标数据."""
        results = []
        
        # 确定要查询的数据库
        if database_ids is None:
            database_ids = list(self.metrics_storage.keys())
        
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)
        
        for database_id in database_ids:
            if database_id not in self.metrics_storage:
                continue
            
            database_metrics = self.metrics_storage[database_id]
            
            # 时间过滤
            filtered_metrics = [
                m for m in database_metrics
                if start_time <= m.timestamp <= end_time
            ]
            
            # 指标过滤
            if metric_ids:
                for metrics in filtered_metrics:
                    filtered_data = {
                        k: v for k, v in metrics.metrics.items()
                        if k in metric_ids
                    }
                    if filtered_data:
                        # 创建新的指标对象，只包含请求的指标
                        filtered_metrics_obj = PerformanceMetrics(
                            database_id=metrics.database_id,
                            database_type=metrics.database_type,
                            timestamp=metrics.timestamp,
                            metrics=filtered_data
                        )
                        results.append(filtered_metrics_obj)
            else:
                results.extend(filtered_metrics)
        
        # 按时间排序
        results.sort(key=lambda x: x.timestamp)
        
        # 限制结果数量
        if limit:
            results = results[-limit:]
        
        return results
    
    async def get_latest_metrics(self, database_ids: Optional[List[str]] = None) -> Dict[str, PerformanceMetrics]:
        """获取最新的指标数据."""
        latest_metrics = {}
        
        if database_ids is None:
            database_ids = list(self.metrics_storage.keys())
        
        for database_id in database_ids:
            if database_id in self.metrics_storage and self.metrics_storage[database_id]:
                latest_metrics[database_id] = self.metrics_storage[database_id][-1]
        
        return latest_metrics
    
    async def get_aggregated_metrics(
        self,
        database_ids: Optional[List[str]] = None,
        metric_ids: Optional[List[str]] = None,
        aggregation: str = "avg",
        time_range: int = 3600,
        group_by_database: bool = True
    ) -> Dict[str, Any]:
        """获取聚合指标数据."""
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=time_range)
        
        metrics_data = await self.get_metrics(
            database_ids=database_ids,
            metric_ids=metric_ids,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics_data:
            return {}
        
        # 按数据库分组聚合
        if group_by_database:
            aggregated = {}
            for database_id in (database_ids or list(self.metrics_storage.keys())):
                db_metrics = [m for m in metrics_data if m.database_id == database_id]
                if db_metrics:
                    aggregated[database_id] = self._aggregate_metrics_list(db_metrics, aggregation)
            return aggregated
        else:
            # 全局聚合
            return self._aggregate_metrics_list(metrics_data, aggregation)
    
    def _aggregate_metrics_list(self, metrics_list: List[PerformanceMetrics], aggregation: str) -> Dict[str, float]:
        """聚合指标列表."""
        if not metrics_list:
            return {}
        
        # 收集所有指标值
        metric_values = {}
        for metrics in metrics_list:
            for metric_id, value in metrics.metrics.items():
                if isinstance(value, (int, float)):
                    if metric_id not in metric_values:
                        metric_values[metric_id] = []
                    metric_values[metric_id].append(value)
        
        # 执行聚合
        aggregated = {}
        for metric_id, values in metric_values.items():
            if aggregation == "avg":
                aggregated[metric_id] = sum(values) / len(values)
            elif aggregation == "sum":
                aggregated[metric_id] = sum(values)
            elif aggregation == "min":
                aggregated[metric_id] = min(values)
            elif aggregation == "max":
                aggregated[metric_id] = max(values)
            elif aggregation == "count":
                aggregated[metric_id] = len(values)
            else:
                aggregated[metric_id] = sum(values) / len(values)  # 默认平均值
        
        return aggregated