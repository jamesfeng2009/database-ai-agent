"""跨数据库事务性能监控服务."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .database_manager import DatabaseManager
from .cross_database_analyzer import CrossDatabaseAnalyzer

logger = logging.getLogger(__name__)


class TransactionStatus(str, Enum):
    """事务状态枚举."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"
    PREPARING = "preparing"
    PREPARED = "prepared"


class AlertSeverity(str, Enum):
    """告警严重程度枚举."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CrossDatabaseTransaction:
    """跨数据库事务."""
    transaction_id: str
    involved_databases: List[str]
    start_time: datetime
    status: TransactionStatus
    coordinator_database: str
    participant_databases: List[str]
    isolation_level: str
    lock_count: int = 0
    data_size: int = 0  # bytes
    network_round_trips: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetric:
    """性能指标."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    database_id: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """性能告警."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    database_id: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringRule:
    """监控规则."""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==
    threshold: float
    severity: AlertSeverity
    duration: int  # 持续时间（秒）
    enabled: bool = True
    description: str = ""


class CrossDatabaseMonitor:
    """跨数据库事务性能监控器."""
    
    def __init__(self, database_manager: DatabaseManager, cross_db_analyzer: CrossDatabaseAnalyzer):
        self.database_manager = database_manager
        self.cross_db_analyzer = cross_db_analyzer
        
        # 监控数据存储
        self.active_transactions: Dict[str, CrossDatabaseTransaction] = {}
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        
        # 监控配置
        self.monitoring_interval = 30  # 秒
        self.metric_retention_hours = 24
        self.alert_retention_hours = 72
        
        # 监控任务
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 性能统计
        self.stats = {
            "total_transactions_monitored": 0,
            "active_transaction_count": 0,
            "total_alerts_generated": 0,
            "active_alert_count": 0,
            "monitoring_start_time": None
        }
    
    async def start_monitoring(self):
        """启动监控服务."""
        if self._running:
            return
        
        try:
            # 初始化依赖组件
            await self.database_manager.initialize()
            await self.cross_db_analyzer.initialize()
            
            # 加载默认监控规则
            await self._load_default_monitoring_rules()
            
            # 启动监控任务
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.stats["monitoring_start_time"] = datetime.now()
            logger.info("跨数据库事务监控服务已启动")
            
        except Exception as e:
            logger.error(f"启动监控服务失败: {e}")
            raise
    
    async def stop_monitoring(self):
        """停止监控服务."""
        if not self._running:
            return
        
        self._running = False
        
        # 取消监控任务
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("跨数据库事务监控服务已停止")
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据."""
        try:
            # 更新统计信息
            await self._update_stats()
            
            # 获取最新性能指标
            latest_metrics = await self._get_latest_metrics()
            
            # 获取活跃告警
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            
            # 获取事务统计
            transaction_stats = await self._get_transaction_statistics()
            
            # 生成性能趋势
            performance_trends = await self._generate_performance_trends()
            
            return {
                "overview": {
                    "monitoring_status": "running" if self._running else "stopped",
                    "monitored_databases": len(self.database_manager.get_healthy_connections()),
                    "active_transactions": len(self.active_transactions),
                    "active_alerts": len(active_alerts),
                    "monitoring_uptime": self._calculate_uptime()
                },
                "statistics": self.stats,
                "latest_metrics": latest_metrics,
                "active_alerts": [self._serialize_alert(alert) for alert in active_alerts],
                "transaction_statistics": transaction_stats,
                "performance_trends": performance_trends,
                "health_summary": await self._generate_health_summary()
            }
            
        except Exception as e:
            logger.error(f"获取监控仪表板数据失败: {e}")
            return {"error": str(e)}
    
    async def add_monitoring_rule(self, rule: MonitoringRule) -> bool:
        """添加监控规则."""
        try:
            self.monitoring_rules[rule.rule_id] = rule
            logger.info(f"已添加监控规则: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"添加监控规则失败: {e}")
            return False
    
    async def remove_monitoring_rule(self, rule_id: str) -> bool:
        """移除监控规则."""
        try:
            if rule_id in self.monitoring_rules:
                del self.monitoring_rules[rule_id]
                logger.info(f"已移除监控规则: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除监控规则失败: {e}")
            return False
    
    async def get_transaction_details(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """获取事务详细信息."""
        try:
            if transaction_id not in self.active_transactions:
                return None
            
            transaction = self.active_transactions[transaction_id]
            
            # 获取事务相关的性能指标
            transaction_metrics = await self._get_transaction_metrics(transaction_id)
            
            # 获取事务依赖关系
            dependencies = await self._analyze_transaction_dependencies(transaction)
            
            return {
                "transaction_id": transaction.transaction_id,
                "involved_databases": transaction.involved_databases,
                "start_time": transaction.start_time.isoformat(),
                "status": transaction.status.value,
                "coordinator_database": transaction.coordinator_database,
                "participant_databases": transaction.participant_databases,
                "isolation_level": transaction.isolation_level,
                "duration": (datetime.now() - transaction.start_time).total_seconds(),
                "lock_count": transaction.lock_count,
                "data_size": transaction.data_size,
                "network_round_trips": transaction.network_round_trips,
                "last_activity": transaction.last_activity.isoformat(),
                "performance_metrics": transaction_metrics,
                "dependencies": dependencies,
                "risk_assessment": await self._assess_transaction_risk(transaction)
            }
            
        except Exception as e:
            logger.error(f"获取事务详细信息失败: {e}")
            return None
    
    async def _monitoring_loop(self):
        """监控主循环."""
        try:
            while self._running:
                await self._collect_performance_metrics()
                await self._monitor_transactions()
                await self._check_alert_conditions()
                await asyncio.sleep(self.monitoring_interval)
        except asyncio.CancelledError:
            logger.info("监控循环已取消")
            raise
        except Exception as e:
            logger.error(f"监控循环错误: {e}")
    
    async def _cleanup_loop(self):
        """清理循环 - 清理过期数据."""
        try:
            while self._running:
                await self._cleanup_expired_data()
                await asyncio.sleep(3600)  # 每小时清理一次
        except asyncio.CancelledError:
            logger.info("清理循环已取消")
            raise
        except Exception as e:
            logger.error(f"清理循环错误: {e}")
    
    async def _collect_performance_metrics(self):
        """收集性能指标."""
        try:
            healthy_connections = self.database_manager.get_healthy_connections()
            
            for db_id in healthy_connections:
                # 收集基础性能指标
                metrics = await self._collect_database_metrics(db_id)
                
                for metric_name, value in metrics.items():
                    metric = PerformanceMetric(
                        metric_name=metric_name,
                        value=value,
                        unit=self._get_metric_unit(metric_name),
                        timestamp=datetime.now(),
                        database_id=db_id
                    )
                    
                    self.performance_metrics[f"{db_id}_{metric_name}"].append(metric)
                
                # 收集跨数据库特定指标
                cross_db_metrics = await self._collect_cross_database_metrics(db_id)
                
                for metric_name, value in cross_db_metrics.items():
                    metric = PerformanceMetric(
                        metric_name=metric_name,
                        value=value,
                        unit=self._get_metric_unit(metric_name),
                        timestamp=datetime.now(),
                        database_id=db_id,
                        tags={"type": "cross_database"}
                    )
                    
                    self.performance_metrics[f"{db_id}_{metric_name}"].append(metric)
                    
        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
    
    async def _monitor_transactions(self):
        """监控事务状态."""
        try:
            healthy_connections = self.database_manager.get_healthy_connections()
            
            for db_id in healthy_connections:
                # 获取活跃事务
                active_txns = await self._get_active_transactions_from_db(db_id)
                
                for txn_info in active_txns:
                    txn_id = txn_info.get("transaction_id")
                    if not txn_id:
                        continue
                    
                    # 检查是否为跨数据库事务
                    if await self._is_cross_database_transaction(txn_info):
                        await self._update_transaction_info(txn_id, txn_info, db_id)
            
            # 清理已完成的事务
            await self._cleanup_completed_transactions()
            
        except Exception as e:
            logger.error(f"监控事务失败: {e}")
    
    async def _check_alert_conditions(self):
        """检查告警条件."""
        try:
            for rule_id, rule in self.monitoring_rules.items():
                if not rule.enabled:
                    continue
                
                # 获取相关指标
                relevant_metrics = await self._get_relevant_metrics(rule)
                
                for db_id, metrics in relevant_metrics.items():
                    if not metrics:
                        continue
                    
                    latest_metric = metrics[-1]
                    
                    # 检查告警条件
                    if self._check_condition(latest_metric.value, rule.condition, rule.threshold):
                        # 检查持续时间
                        if await self._check_duration(rule, db_id, metrics):
                            await self._generate_alert(rule, db_id, latest_metric)
                    else:
                        # 检查是否需要解决告警
                        await self._resolve_alert_if_exists(rule, db_id)
                        
        except Exception as e:
            logger.error(f"检查告警条件失败: {e}")
    
    async def _load_default_monitoring_rules(self):
        """加载默认监控规则."""
        default_rules = [
            MonitoringRule(
                rule_id="high_active_transactions",
                name="活跃事务数过高",
                metric_name="active_transactions",
                condition=">",
                threshold=100,
                severity=AlertSeverity.WARNING,
                duration=300,  # 5分钟
                description="活跃事务数超过阈值"
            ),
            MonitoringRule(
                rule_id="long_running_transaction",
                name="长时间运行事务",
                metric_name="max_transaction_duration",
                condition=">",
                threshold=3600,  # 1小时
                severity=AlertSeverity.ERROR,
                duration=60,
                description="存在运行时间超过1小时的事务"
            ),
            MonitoringRule(
                rule_id="high_lock_waits",
                name="锁等待过多",
                metric_name="lock_waits_per_second",
                condition=">",
                threshold=10,
                severity=AlertSeverity.WARNING,
                duration=180,  # 3分钟
                description="锁等待频率过高"
            ),
            MonitoringRule(
                rule_id="deadlock_detected",
                name="检测到死锁",
                metric_name="deadlocks_per_minute",
                condition=">",
                threshold=0,
                severity=AlertSeverity.ERROR,
                duration=0,  # 立即告警
                description="检测到数据库死锁"
            ),
            MonitoringRule(
                rule_id="high_cross_db_latency",
                name="跨数据库延迟过高",
                metric_name="cross_db_avg_latency",
                condition=">",
                threshold=500,  # 500ms
                severity=AlertSeverity.WARNING,
                duration=300,
                description="跨数据库操作平均延迟过高"
            )
        ]
        
        for rule in default_rules:
            self.monitoring_rules[rule.rule_id] = rule
    
    async def _collect_database_metrics(self, db_id: str) -> Dict[str, float]:
        """收集数据库基础指标."""
        metrics = {}
        
        try:
            # 获取连接状态
            connection_status = self.database_manager.get_connection_status()
            if db_id in connection_status:
                health = connection_status[db_id]
                metrics["response_time"] = health.response_time_ms
                metrics["connection_failures"] = health.consecutive_failures
            
            # 获取连接统计
            connection_stats = self.database_manager.get_connection_stats()
            if db_id in connection_stats:
                stats = connection_stats[db_id]
                metrics["connection_count"] = stats.get("connection_count", 0)
            
            # 模拟其他指标（在实际实现中应该从数据库查询）
            metrics.update({
                "active_transactions": self._simulate_metric("active_transactions", 0, 200),
                "lock_waits_per_second": self._simulate_metric("lock_waits", 0, 20),
                "deadlocks_per_minute": self._simulate_metric("deadlocks", 0, 5),
                "queries_per_second": self._simulate_metric("qps", 10, 1000),
                "cpu_usage": self._simulate_metric("cpu", 10, 90),
                "memory_usage": self._simulate_metric("memory", 20, 85)
            })
            
        except Exception as e:
            logger.error(f"收集数据库 {db_id} 指标失败: {e}")
        
        return metrics
    
    async def _collect_cross_database_metrics(self, db_id: str) -> Dict[str, float]:
        """收集跨数据库特定指标."""
        metrics = {}
        
        try:
            # 跨数据库连接数
            cross_db_connections = len([
                txn for txn in self.active_transactions.values()
                if db_id in txn.involved_databases and len(txn.involved_databases) > 1
            ])
            metrics["cross_db_connections"] = cross_db_connections
            
            # 跨数据库事务平均延迟
            cross_db_latencies = []
            for txn in self.active_transactions.values():
                if db_id in txn.involved_databases and len(txn.involved_databases) > 1:
                    duration = (datetime.now() - txn.start_time).total_seconds() * 1000
                    cross_db_latencies.append(duration)
            
            if cross_db_latencies:
                metrics["cross_db_avg_latency"] = sum(cross_db_latencies) / len(cross_db_latencies)
                metrics["cross_db_max_latency"] = max(cross_db_latencies)
            else:
                metrics["cross_db_avg_latency"] = 0
                metrics["cross_db_max_latency"] = 0
            
            # 跨数据库数据传输量
            total_data_size = sum(
                txn.data_size for txn in self.active_transactions.values()
                if db_id in txn.involved_databases and len(txn.involved_databases) > 1
            )
            metrics["cross_db_data_transfer"] = total_data_size
            
            # 网络往返次数
            total_round_trips = sum(
                txn.network_round_trips for txn in self.active_transactions.values()
                if db_id in txn.involved_databases and len(txn.involved_databases) > 1
            )
            metrics["cross_db_network_round_trips"] = total_round_trips
            
        except Exception as e:
            logger.error(f"收集数据库 {db_id} 跨数据库指标失败: {e}")
        
        return metrics
    
    def _simulate_metric(self, metric_type: str, min_val: float, max_val: float) -> float:
        """模拟指标数据（用于演示）."""
        import random
        
        # 基于时间的波动
        time_factor = time.time() % 3600 / 3600  # 小时内的位置
        base_value = min_val + (max_val - min_val) * (0.5 + 0.3 * time_factor)
        
        # 添加随机噪声
        noise = random.uniform(-0.1, 0.1) * (max_val - min_val)
        
        return max(min_val, min(max_val, base_value + noise))
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """获取指标单位."""
        unit_map = {
            "response_time": "ms",
            "cross_db_avg_latency": "ms",
            "cross_db_max_latency": "ms",
            "connection_count": "count",
            "active_transactions": "count",
            "lock_waits_per_second": "count/s",
            "deadlocks_per_minute": "count/min",
            "queries_per_second": "count/s",
            "cpu_usage": "%",
            "memory_usage": "%",
            "cross_db_data_transfer": "bytes",
            "cross_db_network_round_trips": "count"
        }
        
        return unit_map.get(metric_name, "")
    
    async def _get_active_transactions_from_db(self, db_id: str) -> List[Dict[str, Any]]:
        """从数据库获取活跃事务信息."""
        # 在实际实现中，这里应该查询数据库的系统表
        # 这里返回模拟数据
        import random
        
        transaction_count = random.randint(0, 5)
        transactions = []
        
        for i in range(transaction_count):
            transactions.append({
                "transaction_id": f"txn_{db_id}_{i}_{int(time.time())}",
                "start_time": datetime.now() - timedelta(seconds=random.randint(1, 3600)),
                "isolation_level": random.choice(["READ_committed", "repeatable_read", "serializable"]),
                "lock_count": random.randint(0, 50),
                "data_size": random.randint(1024, 1024*1024),  # 1KB to 1MB
                "status": "active"
            })
        
        return transactions
    
    async def _is_cross_database_transaction(self, txn_info: Dict[str, Any]) -> bool:
        """判断是否为跨数据库事务."""
        # 简化的判断逻辑
        # 在实际实现中，需要分析事务涉及的表和数据库
        import random
        return random.random() < 0.3  # 30%的概率是跨数据库事务
    
    async def _update_transaction_info(self, txn_id: str, txn_info: Dict[str, Any], db_id: str):
        """更新事务信息."""
        if txn_id in self.active_transactions:
            # 更新现有事务
            transaction = self.active_transactions[txn_id]
            transaction.last_activity = datetime.now()
            transaction.lock_count = txn_info.get("lock_count", transaction.lock_count)
            transaction.data_size = txn_info.get("data_size", transaction.data_size)
        else:
            # 创建新事务记录
            transaction = CrossDatabaseTransaction(
                transaction_id=txn_id,
                involved_databases=[db_id],  # 初始只有一个数据库
                start_time=txn_info.get("start_time", datetime.now()),
                status=TransactionStatus.ACTIVE,
                coordinator_database=db_id,
                participant_databases=[],
                isolation_level=txn_info.get("isolation_level", "read_committed"),
                lock_count=txn_info.get("lock_count", 0),
                data_size=txn_info.get("data_size", 0)
            )
            
            self.active_transactions[txn_id] = transaction
            self.stats["total_transactions_monitored"] += 1
    
    async def _cleanup_completed_transactions(self):
        """清理已完成的事务."""
        current_time = datetime.now()
        completed_transactions = []
        
        for txn_id, transaction in self.active_transactions.items():
            # 如果事务超过1小时没有活动，认为已完成
            if (current_time - transaction.last_activity).total_seconds() > 3600:
                completed_transactions.append(txn_id)
        
        for txn_id in completed_transactions:
            del self.active_transactions[txn_id]
    
    async def _get_relevant_metrics(self, rule: MonitoringRule) -> Dict[str, List[PerformanceMetric]]:
        """获取与规则相关的指标."""
        relevant_metrics = {}
        
        for key, metrics_deque in self.performance_metrics.items():
            if rule.metric_name in key:
                db_id = key.split("_")[0]
                if db_id not in relevant_metrics:
                    relevant_metrics[db_id] = []
                relevant_metrics[db_id].extend(list(metrics_deque))
        
        return relevant_metrics
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """检查条件是否满足."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        else:
            return False
    
    async def _check_duration(self, rule: MonitoringRule, db_id: str, metrics: List[PerformanceMetric]) -> bool:
        """检查条件持续时间是否满足."""
        if rule.duration == 0:
            return True  # 立即告警
        
        current_time = datetime.now()
        duration_start = current_time - timedelta(seconds=rule.duration)
        
        # 检查在持续时间内是否一直满足条件
        relevant_metrics = [
            m for m in metrics 
            if m.timestamp >= duration_start and m.database_id == db_id
        ]
        
        if not relevant_metrics:
            return False
        
        # 检查所有相关指标是否都满足条件
        return all(
            self._check_condition(m.value, rule.condition, rule.threshold)
            for m in relevant_metrics
        )
    
    async def _generate_alert(self, rule: MonitoringRule, db_id: str, metric: PerformanceMetric):
        """生成告警."""
        alert_id = f"{rule.rule_id}_{db_id}_{int(time.time())}"
        
        # 检查是否已存在相同的活跃告警
        existing_alert_key = f"{rule.rule_id}_{db_id}"
        if any(alert.alert_id.startswith(existing_alert_key) and not alert.resolved 
               for alert in self.alerts.values()):
            return  # 避免重复告警
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            alert_type=rule.rule_id,
            severity=rule.severity,
            message=f"{rule.name}: {rule.description}",
            database_id=db_id,
            metric_name=rule.metric_name,
            threshold=rule.threshold,
            current_value=metric.value,
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        self.stats["total_alerts_generated"] += 1
        
        logger.warning(f"生成告警: {alert.message} (数据库: {db_id}, 当前值: {metric.value}, 阈值: {rule.threshold})")
    
    async def _resolve_alert_if_exists(self, rule: MonitoringRule, db_id: str):
        """如果存在相关告警则解决它."""
        for alert in self.alerts.values():
            if (alert.alert_type == rule.rule_id and 
                alert.database_id == db_id and 
                not alert.resolved):
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"告警已解决: {alert.message}")
    
    async def _cleanup_expired_data(self):
        """清理过期数据."""
        current_time = datetime.now()
        
        # 清理过期的性能指标
        metric_cutoff = current_time - timedelta(hours=self.metric_retention_hours)
        for key, metrics_deque in self.performance_metrics.items():
            # 移除过期指标
            while metrics_deque and metrics_deque[0].timestamp < metric_cutoff:
                metrics_deque.popleft()
        
        # 清理过期的告警
        alert_cutoff = current_time - timedelta(hours=self.alert_retention_hours)
        expired_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.timestamp < alert_cutoff and alert.resolved
        ]
        
        for alert_id in expired_alerts:
            del self.alerts[alert_id]
    
    async def _update_stats(self):
        """更新统计信息."""
        self.stats["active_transaction_count"] = len(self.active_transactions)
        self.stats["active_alert_count"] = len([
            alert for alert in self.alerts.values() if not alert.resolved
        ])
    
    async def _get_latest_metrics(self) -> Dict[str, Any]:
        """获取最新的性能指标."""
        latest_metrics = {}
        
        for key, metrics_deque in self.performance_metrics.items():
            if metrics_deque:
                latest_metric = metrics_deque[-1]
                latest_metrics[key] = {
                    "value": latest_metric.value,
                    "unit": latest_metric.unit,
                    "timestamp": latest_metric.timestamp.isoformat(),
                    "database_id": latest_metric.database_id
                }
        
        return latest_metrics
    
    async def _get_transaction_statistics(self) -> Dict[str, Any]:
        """获取事务统计信息."""
        if not self.active_transactions:
            return {
                "total_active": 0,
                "cross_database_count": 0,
                "average_duration": 0,
                "longest_running": None
            }
        
        current_time = datetime.now()
        cross_db_count = len([
            txn for txn in self.active_transactions.values()
            if len(txn.involved_databases) > 1
        ])
        
        durations = [
            (current_time - txn.start_time).total_seconds()
            for txn in self.active_transactions.values()
        ]
        
        longest_txn = max(self.active_transactions.values(), 
                         key=lambda x: (current_time - x.start_time).total_seconds())
        
        return {
            "total_active": len(self.active_transactions),
            "cross_database_count": cross_db_count,
            "average_duration": sum(durations) / len(durations),
            "longest_running": {
                "transaction_id": longest_txn.transaction_id,
                "duration": (current_time - longest_txn.start_time).total_seconds(),
                "involved_databases": longest_txn.involved_databases
            }
        }
    
    async def _generate_performance_trends(self) -> Dict[str, Any]:
        """生成性能趋势数据."""
        trends = {}
        
        # 选择关键指标生成趋势
        key_metrics = ["active_transactions", "cross_db_avg_latency", "lock_waits_per_second"]
        
        for metric_name in key_metrics:
            metric_data = []
            
            for key, metrics_deque in self.performance_metrics.items():
                if metric_name in key:
                    for metric in list(metrics_deque)[-20:]:  # 最近20个数据点
                        metric_data.append({
                            "timestamp": metric.timestamp.isoformat(),
                            "value": metric.value,
                            "database_id": metric.database_id
                        })
            
            trends[metric_name] = metric_data
        
        return trends
    
    async def _generate_health_summary(self) -> Dict[str, Any]:
        """生成健康摘要."""
        healthy_dbs = len(self.database_manager.get_healthy_connections())
        total_dbs = len(self.database_manager.list_database_configs())
        
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        critical_alerts = [alert for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL]
        error_alerts = [alert for alert in active_alerts if alert.severity == AlertSeverity.ERROR]
        
        # 计算健康评分
        health_score = 100
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
        if error_alerts:
            health_score -= len(error_alerts) * 10
        if len(active_alerts) > 5:
            health_score -= 10
        
        health_score = max(0, health_score)
        
        return {
            "overall_health_score": health_score,
            "healthy_databases": healthy_dbs,
            "total_databases": total_dbs,
            "database_health_ratio": healthy_dbs / total_dbs if total_dbs > 0 else 0,
            "alert_summary": {
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
            },
            "status": self._determine_overall_status(health_score, critical_alerts, error_alerts)
        }
    
    def _determine_overall_status(self, health_score: int, critical_alerts: List, error_alerts: List) -> str:
        """确定整体状态."""
        if critical_alerts:
            return "critical"
        elif error_alerts:
            return "error"
        elif health_score < 70:
            return "warning"
        else:
            return "healthy"
    
    def _calculate_uptime(self) -> float:
        """计算监控服务运行时间（小时）."""
        if self.stats["monitoring_start_time"]:
            uptime = datetime.now() - self.stats["monitoring_start_time"]
            return uptime.total_seconds() / 3600
        return 0.0
    
    def _serialize_alert(self, alert: PerformanceAlert) -> Dict[str, Any]:
        """序列化告警对象."""
        return {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "message": alert.message,
            "database_id": alert.database_id,
            "metric_name": alert.metric_name,
            "threshold": alert.threshold,
            "current_value": alert.current_value,
            "timestamp": alert.timestamp.isoformat(),
            "resolved": alert.resolved,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
        }
    
    # 事务分析相关方法
    async def _get_transaction_metrics(self, transaction_id: str) -> Dict[str, Any]:
        """获取事务相关的性能指标."""
        # 这里应该返回与特定事务相关的指标
        return {
            "duration": 0,
            "lock_wait_time": 0,
            "data_transfer_size": 0,
            "network_round_trips": 0
        }
    
    async def _analyze_transaction_dependencies(self, transaction: CrossDatabaseTransaction) -> List[Dict[str, Any]]:
        """分析事务依赖关系."""
        dependencies = []
        
        # 分析事务涉及的数据库间依赖
        for i, db1 in enumerate(transaction.involved_databases):
            for db2 in transaction.involved_databases[i+1:]:
                # 获取数据库间的依赖关系
                db_dependencies = await self.cross_db_analyzer.get_database_dependencies(db1)
                
                for dep in db_dependencies:
                    if dep.target_database.database_id == db2:
                        dependencies.append({
                            "source_database": db1,
                            "target_database": db2,
                            "dependency_type": dep.dependency_type.value,
                            "strength": dep.strength,
                            "performance_impact": dep.performance_impact
                        })
        
        return dependencies
    
    async def _assess_transaction_risk(self, transaction: CrossDatabaseTransaction) -> Dict[str, float]:
        """评估事务风险."""
        risks = {
            "deadlock_risk": 0.0,
            "timeout_risk": 0.0,
            "consistency_risk": 0.0,
            "performance_risk": 0.0
        }
        
        # 基于事务特征评估风险
        duration = (datetime.now() - transaction.start_time).total_seconds()
        
        # 死锁风险
        if len(transaction.involved_databases) > 2:
            risks["deadlock_risk"] = 0.3
        if transaction.lock_count > 50:
            risks["deadlock_risk"] += 0.2
        
        # 超时风险
        if duration > 1800:  # 30分钟
            risks["timeout_risk"] = 0.6
        elif duration > 600:  # 10分钟
            risks["timeout_risk"] = 0.3
        
        # 一致性风险
        if len(transaction.involved_databases) > 1:
            risks["consistency_risk"] = 0.2
        
        # 性能风险
        if transaction.data_size > 10 * 1024 * 1024:  # 10MB
            risks["performance_risk"] = 0.4
        
        # 限制风险值在0-1范围内
        for key in risks:
            risks[key] = min(risks[key], 1.0)
        
        return risks