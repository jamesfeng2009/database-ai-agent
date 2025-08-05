"""数据库连接管理器 - 支持动态管理、负载均衡和故障转移."""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .connector_base import BaseDatabaseConnector
from .factory import create_database_connector
from .models import (
    ConnectionStatus,
    DatabaseCluster,
    DatabaseConfig,
    DatabaseConnectionPool,
    DatabaseHealthCheck,
    LoadBalancingStrategy
)

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """数据库连接管理器 - 支持多数据库、负载均衡和故障转移."""
    
    def __init__(self):
        self.connections: Dict[str, BaseDatabaseConnector] = {}
        self.connection_pools: Dict[str, DatabaseConnectionPool] = {}
        self.clusters: Dict[str, DatabaseCluster] = {}
        self.health_checks: Dict[str, DatabaseHealthCheck] = {}
        self.connection_weights: Dict[str, int] = {}
        self.connection_counters: Dict[str, int] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动连接管理器."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("数据库连接管理器已启动")
    
    async def stop(self):
        """停止连接管理器."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        for connection in self.connections.values():
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(f"关闭连接失败: {e}")
        
        self.connections.clear()
        logger.info("数据库连接管理器已停止")
    
    async def add_connection(self, connection_id: str, config: DatabaseConfig, weight: int = 1) -> bool:
        """添加数据库连接."""
        try:
            connector = create_database_connector(config)
            await connector.connect()
            
            self.connections[connection_id] = connector
            self.connection_weights[connection_id] = weight
            self.connection_counters[connection_id] = 0
            
            # 初始化健康检查
            health_check = DatabaseHealthCheck(
                connection_id=connection_id,
                database_type=config.get_database_type(),
                host=config.host,
                port=config.port,
                database=config.database,
                status=ConnectionStatus.HEALTHY,
                response_time_ms=0.0,
                last_check_time=time.time(),
                consecutive_failures=0
            )
            self.health_checks[connection_id] = health_check
            
            logger.info(f"已添加数据库连接: {connection_id} ({config.get_database_type()})")
            return True
            
        except Exception as e:
            logger.error(f"添加数据库连接失败: {connection_id}, 错误: {e}")
            return False
    
    async def remove_connection(self, connection_id: str) -> bool:
        """移除数据库连接."""
        try:
            if connection_id in self.connections:
                await self.connections[connection_id].disconnect()
                del self.connections[connection_id]
                del self.connection_weights[connection_id]
                del self.connection_counters[connection_id]
                del self.health_checks[connection_id]
                
                logger.info(f"已移除数据库连接: {connection_id}")
                return True
            else:
                logger.warning(f"连接不存在: {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"移除数据库连接失败: {connection_id}, 错误: {e}")
            return False
    
    async def get_connection(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN) -> Optional[BaseDatabaseConnector]:
        """根据负载均衡策略获取数据库连接."""
        healthy_connections = [
            conn_id for conn_id, health in self.health_checks.items()
            if health.status == ConnectionStatus.HEALTHY
        ]
        
        if not healthy_connections:
            logger.error("没有可用的健康数据库连接")
            return None
        
        connection_id = self._select_connection(healthy_connections, strategy)
        if connection_id:
            self.connection_counters[connection_id] += 1
            return self.connections[connection_id]
        
        return None
    
    def _select_connection(self, healthy_connections: List[str], strategy: LoadBalancingStrategy) -> Optional[str]:
        """根据策略选择连接."""
        if not healthy_connections:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_connections)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_connections)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_connections)
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_connections)
        else:
            return random.choice(healthy_connections)
    
    def _round_robin_selection(self, connections: List[str]) -> str:
        """轮询选择."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        connection = connections[self._round_robin_index % len(connections)]
        self._round_robin_index += 1
        return connection
    
    def _least_connections_selection(self, connections: List[str]) -> str:
        """最少连接选择."""
        return min(connections, key=lambda x: self.connection_counters.get(x, 0))
    
    def _weighted_round_robin_selection(self, connections: List[str]) -> str:
        """加权轮询选择."""
        weighted_connections = []
        for conn_id in connections:
            weight = self.connection_weights.get(conn_id, 1)
            weighted_connections.extend([conn_id] * weight)
        
        if not weighted_connections:
            return connections[0]
        
        if not hasattr(self, '_weighted_round_robin_index'):
            self._weighted_round_robin_index = 0
        
        connection = weighted_connections[self._weighted_round_robin_index % len(weighted_connections)]
        self._weighted_round_robin_index += 1
        return connection
    
    def _health_based_selection(self, connections: List[str]) -> str:
        """基于健康状态选择."""
        # 根据响应时间选择最快的连接
        return min(connections, key=lambda x: self.health_checks[x].response_time_ms)
    
    async def _health_check_loop(self):
        """健康检查循环."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # 每30秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """执行健康检查."""
        tasks = []
        for connection_id in list(self.connections.keys()):
            task = asyncio.create_task(self._check_connection_health(connection_id))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_connection_health(self, connection_id: str):
        """检查单个连接的健康状态."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        health_check = self.health_checks[connection_id]
        
        start_time = time.time()
        try:
            # 执行简单的健康检查查询
            success = await connection.test_connection()
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            if success:
                health_check.status = ConnectionStatus.HEALTHY
                health_check.consecutive_failures = 0
                health_check.error_message = None
            else:
                health_check.consecutive_failures += 1
                if health_check.consecutive_failures >= 3:
                    health_check.status = ConnectionStatus.UNHEALTHY
                else:
                    health_check.status = ConnectionStatus.DEGRADED
                health_check.error_message = "连接测试失败"
            
            health_check.response_time_ms = response_time
            health_check.last_check_time = time.time()
            
        except Exception as e:
            health_check.consecutive_failures += 1
            health_check.error_message = str(e)
            health_check.response_time_ms = (time.time() - start_time) * 1000
            health_check.last_check_time = time.time()
            
            if health_check.consecutive_failures >= 3:
                health_check.status = ConnectionStatus.UNHEALTHY
                logger.warning(f"连接 {connection_id} 标记为不健康: {e}")
            else:
                health_check.status = ConnectionStatus.DEGRADED
    
    async def create_connection_pool(self, pool_config: DatabaseConnectionPool) -> bool:
        """创建连接池."""
        try:
            self.connection_pools[pool_config.pool_id] = pool_config
            
            # 为池中的每个数据库创建连接
            for i, db_config in enumerate(pool_config.database_configs):
                connection_id = f"{pool_config.pool_id}_db_{i}"
                success = await self.add_connection(connection_id, db_config)
                if not success:
                    logger.error(f"创建连接池 {pool_config.pool_id} 时，连接 {connection_id} 创建失败")
            
            logger.info(f"已创建连接池: {pool_config.pool_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建连接池失败: {pool_config.pool_id}, 错误: {e}")
            return False
    
    async def create_cluster(self, cluster_config: DatabaseCluster) -> bool:
        """创建数据库集群."""
        try:
            self.clusters[cluster_config.cluster_id] = cluster_config
            
            # 创建主数据库连接
            primary_id = f"{cluster_config.cluster_id}_primary"
            success = await self.add_connection(primary_id, cluster_config.primary_config, weight=10)
            if not success:
                logger.error(f"创建集群 {cluster_config.cluster_id} 时，主数据库连接创建失败")
                return False
            
            # 创建从数据库连接
            for i, replica_config in enumerate(cluster_config.replica_configs):
                replica_id = f"{cluster_config.cluster_id}_replica_{i}"
                success = await self.add_connection(replica_id, replica_config, weight=5)
                if not success:
                    logger.warning(f"创建集群 {cluster_config.cluster_id} 时，从数据库连接 {replica_id} 创建失败")
            
            logger.info(f"已创建数据库集群: {cluster_config.cluster_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建数据库集群失败: {cluster_config.cluster_id}, 错误: {e}")
            return False
    
    async def get_cluster_connection(self, cluster_id: str, read_only: bool = False) -> Optional[BaseDatabaseConnector]:
        """从集群获取连接."""
        if cluster_id not in self.clusters:
            logger.error(f"集群不存在: {cluster_id}")
            return None
        
        cluster = self.clusters[cluster_id]
        
        if read_only and cluster.read_write_split:
            # 优先使用从数据库进行读操作
            replica_connections = [
                conn_id for conn_id in self.connections.keys()
                if conn_id.startswith(f"{cluster_id}_replica_") and 
                self.health_checks[conn_id].status == ConnectionStatus.HEALTHY
            ]
            
            if replica_connections:
                connection_id = random.choice(replica_connections)
                self.connection_counters[connection_id] += 1
                return self.connections[connection_id]
        
        # 使用主数据库
        primary_id = f"{cluster_id}_primary"
        if primary_id in self.connections and self.health_checks[primary_id].status == ConnectionStatus.HEALTHY:
            self.connection_counters[primary_id] += 1
            return self.connections[primary_id]
        
        # 如果主数据库不可用，尝试故障转移
        if cluster.auto_failover:
            return await self._perform_failover(cluster_id)
        
        return None
    
    async def _perform_failover(self, cluster_id: str) -> Optional[BaseDatabaseConnector]:
        """执行故障转移."""
        logger.warning(f"集群 {cluster_id} 执行故障转移")
        
        # 查找健康的从数据库
        replica_connections = [
            conn_id for conn_id in self.connections.keys()
            if conn_id.startswith(f"{cluster_id}_replica_") and 
            self.health_checks[conn_id].status == ConnectionStatus.HEALTHY
        ]
        
        if replica_connections:
            # 选择响应时间最短的从数据库作为新的主数据库
            best_replica = min(replica_connections, 
                             key=lambda x: self.health_checks[x].response_time_ms)
            
            logger.info(f"故障转移: 将 {best_replica} 提升为主数据库")
            self.connection_counters[best_replica] += 1
            return self.connections[best_replica]
        
        logger.error(f"集群 {cluster_id} 故障转移失败: 没有可用的从数据库")
        return None
    
    def get_connection_status(self) -> Dict[str, DatabaseHealthCheck]:
        """获取所有连接的状态."""
        return self.health_checks.copy()
    
    def get_connection_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取连接统计信息."""
        stats = {}
        for connection_id in self.connections.keys():
            health = self.health_checks[connection_id]
            stats[connection_id] = {
                "status": health.status,
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures,
                "connection_count": self.connection_counters[connection_id],
                "weight": self.connection_weights[connection_id],
                "last_check_time": health.last_check_time
            }
        return stats