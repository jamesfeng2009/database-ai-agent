"""数据库管理器 - 统一的数据库管理接口."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from .adapters import DatabaseAdapterFactory
from .config_manager import DatabaseConfigManager
from .connection_manager import DatabaseConnectionManager
from .models import (
    DatabaseConfig,
    DatabaseHealthCheck,
    LoadBalancingStrategy,
    ConnectionStatus
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器 - 提供统一的数据库管理接口."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_manager = DatabaseConfigManager(config_file)
        self.connection_manager = DatabaseConnectionManager()
        self.adapter_factory = DatabaseAdapterFactory()
        self._initialized = False
    
    async def initialize(self):
        """初始化数据库管理器."""
        if self._initialized:
            return
        
        try:
            # 启动连接管理器
            await self.connection_manager.start()
            
            # 自动加载配置中的数据库连接
            await self._load_configured_connections()
            
            self._initialized = True
            logger.info("数据库管理器初始化完成")
            
        except Exception as e:
            logger.error(f"数据库管理器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭数据库管理器."""
        if not self._initialized:
            return
        
        try:
            await self.connection_manager.stop()
            self._initialized = False
            logger.info("数据库管理器已关闭")
            
        except Exception as e:
            logger.error(f"数据库管理器关闭失败: {e}")
    
    async def _load_configured_connections(self):
        """加载配置中的数据库连接."""
        # 加载单个数据库连接
        for config_id, config in self.config_manager.list_database_configs().items():
            success = await self.connection_manager.add_connection(config_id, config)
            if success:
                logger.info(f"已加载数据库连接: {config_id}")
            else:
                logger.warning(f"加载数据库连接失败: {config_id}")
        
        # 加载连接池
        for pool_id, pool_config in self.config_manager.list_connection_pools().items():
            success = await self.connection_manager.create_connection_pool(pool_config)
            if success:
                logger.info(f"已加载连接池: {pool_id}")
            else:
                logger.warning(f"加载连接池失败: {pool_id}")
        
        # 加载集群
        for cluster_id, cluster_config in self.config_manager.list_clusters().items():
            success = await self.connection_manager.create_cluster(cluster_config)
            if success:
                logger.info(f"已加载数据库集群: {cluster_id}")
            else:
                logger.warning(f"加载数据库集群失败: {cluster_id}")
    
    # 数据库配置管理接口
    def add_database_config(self, config_id: str, config: DatabaseConfig) -> bool:
        """添加数据库配置."""
        return self.config_manager.add_database_config(config_id, config)
    
    def remove_database_config(self, config_id: str) -> bool:
        """移除数据库配置."""
        return self.config_manager.remove_database_config(config_id)
    
    def get_database_config(self, config_id: str) -> Optional[DatabaseConfig]:
        """获取数据库配置."""
        return self.config_manager.get_database_config(config_id)
    
    def list_database_configs(self) -> Dict[str, DatabaseConfig]:
        """列出所有数据库配置."""
        return self.config_manager.list_database_configs()
    
    def get_supported_database_types(self) -> List[str]:
        """获取支持的数据库类型."""
        return self.config_manager.get_supported_database_types()
    
    # 连接管理接口
    async def add_connection(self, connection_id: str, config: DatabaseConfig, weight: int = 1) -> bool:
        """添加数据库连接."""
        if not self._initialized:
            await self.initialize()
        
        return await self.connection_manager.add_connection(connection_id, config, weight)
    
    async def remove_connection(self, connection_id: str) -> bool:
        """移除数据库连接."""
        if not self._initialized:
            return False
        
        return await self.connection_manager.remove_connection(connection_id)
    
    async def get_connection(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """获取数据库连接."""
        if not self._initialized:
            await self.initialize()
        
        return await self.connection_manager.get_connection(strategy)
    
    async def get_cluster_connection(self, cluster_id: str, read_only: bool = False):
        """从集群获取连接."""
        if not self._initialized:
            await self.initialize()
        
        return await self.connection_manager.get_cluster_connection(cluster_id, read_only)
    
    # 健康检查接口
    def get_connection_status(self) -> Dict[str, DatabaseHealthCheck]:
        """获取所有连接的健康状态."""
        if not self._initialized:
            return {}
        
        return self.connection_manager.get_connection_status()
    
    def get_connection_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取连接统计信息."""
        if not self._initialized:
            return {}
        
        return self.connection_manager.get_connection_stats()
    
    def get_healthy_connections(self) -> List[str]:
        """获取健康的连接列表."""
        status = self.get_connection_status()
        return [
            conn_id for conn_id, health in status.items()
            if health.status == ConnectionStatus.HEALTHY
        ]
    
    def get_unhealthy_connections(self) -> List[str]:
        """获取不健康的连接列表."""
        status = self.get_connection_status()
        return [
            conn_id for conn_id, health in status.items()
            if health.status == ConnectionStatus.UNHEALTHY
        ]
    
    # 数据库适配器接口
    def get_database_adapter(self, database_type: str):
        """获取数据库适配器."""
        return self.adapter_factory.create_adapter(database_type)
    
    def register_database_adapter(self, database_type: str, adapter_class: type):
        """注册新的数据库适配器."""
        self.adapter_factory.register_adapter(database_type, adapter_class)
    
    # 便捷方法
    async def execute_query(self, sql: str, params=None, connection_id: str = None):
        """执行SQL查询."""
        if connection_id:
            # 使用指定连接
            if connection_id not in self.connection_manager.connections:
                raise ValueError(f"连接不存在: {connection_id}")
            connection = self.connection_manager.connections[connection_id]
        else:
            # 使用负载均衡选择连接
            connection = await self.get_connection()
            if not connection:
                raise RuntimeError("没有可用的数据库连接")
        
        return await connection.execute_query(sql, params)
    
    async def execute_explain(self, sql: str, connection_id: str = None):
        """执行EXPLAIN查询."""
        if connection_id:
            # 使用指定连接
            if connection_id not in self.connection_manager.connections:
                raise ValueError(f"连接不存在: {connection_id}")
            connection = self.connection_manager.connections[connection_id]
        else:
            # 使用负载均衡选择连接
            connection = await self.get_connection()
            if not connection:
                raise RuntimeError("没有可用的数据库连接")
        
        return await connection.execute_explain(sql)
    
    async def test_connection(self, connection_id: str) -> bool:
        """测试指定连接."""
        if connection_id not in self.connection_manager.connections:
            return False
        
        connection = self.connection_manager.connections[connection_id]
        return await connection.test_connection()
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """测试所有连接."""
        results = {}
        tasks = []
        
        for connection_id in self.connection_manager.connections.keys():
            task = asyncio.create_task(self.test_connection(connection_id))
            tasks.append((connection_id, task))
        
        for connection_id, task in tasks:
            try:
                results[connection_id] = await task
            except Exception as e:
                logger.error(f"测试连接 {connection_id} 失败: {e}")
                results[connection_id] = False
        
        return results
    
    # 配置导入导出
    def export_config(self, export_file: str) -> bool:
        """导出配置."""
        return self.config_manager.export_config(export_file)
    
    def import_config(self, import_file: str) -> bool:
        """导入配置."""
        success = self.config_manager.import_config(import_file)
        if success and self._initialized:
            # 重新加载连接
            asyncio.create_task(self._reload_connections())
        return success
    
    async def _reload_connections(self):
        """重新加载连接."""
        try:
            # 停止当前连接管理器
            await self.connection_manager.stop()
            
            # 重新创建连接管理器
            self.connection_manager = DatabaseConnectionManager()
            await self.connection_manager.start()
            
            # 重新加载配置的连接
            await self._load_configured_connections()
            
            logger.info("连接已重新加载")
            
        except Exception as e:
            logger.error(f"重新加载连接失败: {e}")
    
    # 监控和统计
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览."""
        connection_status = self.get_connection_status()
        connection_stats = self.get_connection_stats()
        
        healthy_count = sum(1 for health in connection_status.values() 
                          if health.status == ConnectionStatus.HEALTHY)
        unhealthy_count = sum(1 for health in connection_status.values() 
                            if health.status == ConnectionStatus.UNHEALTHY)
        degraded_count = sum(1 for health in connection_status.values() 
                           if health.status == ConnectionStatus.DEGRADED)
        
        total_connections = sum(stats.get('connection_count', 0) 
                              for stats in connection_stats.values())
        
        avg_response_time = 0
        if connection_status:
            avg_response_time = sum(health.response_time_ms 
                                  for health in connection_status.values()) / len(connection_status)
        
        return {
            "total_databases": len(self.config_manager.list_database_configs()),
            "total_pools": len(self.config_manager.list_connection_pools()),
            "total_clusters": len(self.config_manager.list_clusters()),
            "healthy_connections": healthy_count,
            "unhealthy_connections": unhealthy_count,
            "degraded_connections": degraded_count,
            "total_connection_count": total_connections,
            "average_response_time_ms": avg_response_time,
            "supported_database_types": self.get_supported_database_types()
        }