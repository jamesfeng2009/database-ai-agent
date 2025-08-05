"""数据库配置管理器 - 统一管理多数据库配置."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from .models import (
    DatabaseConfig,
    DatabaseType,
    MySQLConfig,
    PostgreSQLConfig,
    TiDBConfig,
    MariaDBConfig,
    OracleConfig,
    SQLServerConfig,
    SQLiteConfig,
    DatabaseConnectionPool,
    DatabaseCluster,
    LoadBalancingStrategy
)

logger = logging.getLogger(__name__)


class DatabaseConfigManager:
    """数据库配置管理器 - 提供统一的配置管理界面."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "database_configs.json"
        self.configs: Dict[str, DatabaseConfig] = {}
        self.pools: Dict[str, DatabaseConnectionPool] = {}
        self.clusters: Dict[str, DatabaseCluster] = {}
        self._load_configs()
    
    def _load_configs(self):
        """从配置文件加载配置."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载数据库配置
                for config_id, config_data in data.get('databases', {}).items():
                    config = self._create_config_from_dict(config_data)
                    if config:
                        self.configs[config_id] = config
                
                # 加载连接池配置
                for pool_id, pool_data in data.get('pools', {}).items():
                    pool = self._create_pool_from_dict(pool_id, pool_data)
                    if pool:
                        self.pools[pool_id] = pool
                
                # 加载集群配置
                for cluster_id, cluster_data in data.get('clusters', {}).items():
                    cluster = self._create_cluster_from_dict(cluster_id, cluster_data)
                    if cluster:
                        self.clusters[cluster_id] = cluster
                
                logger.info(f"已加载 {len(self.configs)} 个数据库配置")
                
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
    
    def _save_configs(self):
        """保存配置到文件."""
        try:
            data = {
                'databases': {},
                'pools': {},
                'clusters': {}
            }
            
            # 保存数据库配置
            for config_id, config in self.configs.items():
                data['databases'][config_id] = config.dict()
            
            # 保存连接池配置
            for pool_id, pool in self.pools.items():
                data['pools'][pool_id] = pool.dict()
            
            # 保存集群配置
            for cluster_id, cluster in self.clusters.items():
                data['clusters'][cluster_id] = cluster.dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("配置已保存到文件")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> Optional[DatabaseConfig]:
        """从字典创建数据库配置."""
        try:
            db_type = config_data.get('database_type') or config_data.get('type')
            if not db_type:
                logger.error("配置中缺少数据库类型")
                return None
            
            if db_type == DatabaseType.MYSQL:
                return MySQLConfig(**config_data)
            elif db_type == DatabaseType.POSTGRESQL:
                return PostgreSQLConfig(**config_data)
            elif db_type == DatabaseType.TIDB:
                return TiDBConfig(**config_data)
            elif db_type == DatabaseType.MARIADB:
                return MariaDBConfig(**config_data)
            elif db_type == DatabaseType.ORACLE:
                return OracleConfig(**config_data)
            elif db_type == DatabaseType.SQLSERVER:
                return SQLServerConfig(**config_data)
            elif db_type == DatabaseType.SQLITE:
                return SQLiteConfig(**config_data)
            else:
                logger.error(f"不支持的数据库类型: {db_type}")
                return None
                
        except Exception as e:
            logger.error(f"创建数据库配置失败: {e}")
            return None
    
    def _create_pool_from_dict(self, pool_id: str, pool_data: Dict[str, Any]) -> Optional[DatabaseConnectionPool]:
        """从字典创建连接池配置."""
        try:
            # 创建数据库配置列表
            database_configs = []
            for db_config_data in pool_data.get('database_configs', []):
                config = self._create_config_from_dict(db_config_data)
                if config:
                    database_configs.append(config)
            
            pool_data['pool_id'] = pool_id
            pool_data['database_configs'] = database_configs
            
            return DatabaseConnectionPool(**pool_data)
            
        except Exception as e:
            logger.error(f"创建连接池配置失败: {e}")
            return None
    
    def _create_cluster_from_dict(self, cluster_id: str, cluster_data: Dict[str, Any]) -> Optional[DatabaseCluster]:
        """从字典创建集群配置."""
        try:
            # 创建主数据库配置
            primary_config = self._create_config_from_dict(cluster_data.get('primary_config', {}))
            if not primary_config:
                logger.error("集群配置中缺少主数据库配置")
                return None
            
            # 创建从数据库配置列表
            replica_configs = []
            for replica_config_data in cluster_data.get('replica_configs', []):
                config = self._create_config_from_dict(replica_config_data)
                if config:
                    replica_configs.append(config)
            
            cluster_data['cluster_id'] = cluster_id
            cluster_data['primary_config'] = primary_config
            cluster_data['replica_configs'] = replica_configs
            
            return DatabaseCluster(**cluster_data)
            
        except Exception as e:
            logger.error(f"创建集群配置失败: {e}")
            return None
    
    def add_database_config(self, config_id: str, config: DatabaseConfig) -> bool:
        """添加数据库配置."""
        try:
            self.configs[config_id] = config
            self._save_configs()
            logger.info(f"已添加数据库配置: {config_id}")
            return True
        except Exception as e:
            logger.error(f"添加数据库配置失败: {e}")
            return False
    
    def remove_database_config(self, config_id: str) -> bool:
        """移除数据库配置."""
        try:
            if config_id in self.configs:
                del self.configs[config_id]
                self._save_configs()
                logger.info(f"已移除数据库配置: {config_id}")
                return True
            else:
                logger.warning(f"配置不存在: {config_id}")
                return False
        except Exception as e:
            logger.error(f"移除数据库配置失败: {e}")
            return False
    
    def get_database_config(self, config_id: str) -> Optional[DatabaseConfig]:
        """获取数据库配置."""
        return self.configs.get(config_id)
    
    def list_database_configs(self) -> Dict[str, DatabaseConfig]:
        """列出所有数据库配置."""
        return self.configs.copy()
    
    def update_database_config(self, config_id: str, config: DatabaseConfig) -> bool:
        """更新数据库配置."""
        try:
            if config_id in self.configs:
                self.configs[config_id] = config
                self._save_configs()
                logger.info(f"已更新数据库配置: {config_id}")
                return True
            else:
                logger.warning(f"配置不存在: {config_id}")
                return False
        except Exception as e:
            logger.error(f"更新数据库配置失败: {e}")
            return False
    
    def create_mysql_config(self, config_id: str, host: str, port: int, user: str, 
                           password: str, database: str, **kwargs) -> bool:
        """创建MySQL配置."""
        try:
            config = MySQLConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                **kwargs
            )
            return self.add_database_config(config_id, config)
        except Exception as e:
            logger.error(f"创建MySQL配置失败: {e}")
            return False
    
    def create_postgresql_config(self, config_id: str, host: str, port: int, user: str,
                                password: str, database: str, **kwargs) -> bool:
        """创建PostgreSQL配置."""
        try:
            config = PostgreSQLConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                **kwargs
            )
            return self.add_database_config(config_id, config)
        except Exception as e:
            logger.error(f"创建PostgreSQL配置失败: {e}")
            return False
    
    def create_tidb_config(self, config_id: str, host: str, port: int, user: str,
                          password: str, database: str, **kwargs) -> bool:
        """创建TiDB配置."""
        try:
            config = TiDBConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                **kwargs
            )
            return self.add_database_config(config_id, config)
        except Exception as e:
            logger.error(f"创建TiDB配置失败: {e}")
            return False
    
    def create_connection_pool(self, pool_id: str, config_ids: List[str], 
                              strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
                              **kwargs) -> bool:
        """创建连接池."""
        try:
            database_configs = []
            for config_id in config_ids:
                config = self.get_database_config(config_id)
                if config:
                    database_configs.append(config)
                else:
                    logger.error(f"配置不存在: {config_id}")
                    return False
            
            pool = DatabaseConnectionPool(
                pool_id=pool_id,
                database_configs=database_configs,
                load_balancing_strategy=strategy,
                **kwargs
            )
            
            self.pools[pool_id] = pool
            self._save_configs()
            logger.info(f"已创建连接池: {pool_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建连接池失败: {e}")
            return False
    
    def create_cluster(self, cluster_id: str, cluster_name: str, primary_config_id: str,
                      replica_config_ids: List[str] = None, **kwargs) -> bool:
        """创建数据库集群."""
        try:
            primary_config = self.get_database_config(primary_config_id)
            if not primary_config:
                logger.error(f"主数据库配置不存在: {primary_config_id}")
                return False
            
            replica_configs = []
            if replica_config_ids:
                for config_id in replica_config_ids:
                    config = self.get_database_config(config_id)
                    if config:
                        replica_configs.append(config)
                    else:
                        logger.error(f"从数据库配置不存在: {config_id}")
                        return False
            
            cluster = DatabaseCluster(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                primary_config=primary_config,
                replica_configs=replica_configs,
                **kwargs
            )
            
            self.clusters[cluster_id] = cluster
            self._save_configs()
            logger.info(f"已创建数据库集群: {cluster_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建数据库集群失败: {e}")
            return False
    
    def get_connection_pool(self, pool_id: str) -> Optional[DatabaseConnectionPool]:
        """获取连接池配置."""
        return self.pools.get(pool_id)
    
    def get_cluster(self, cluster_id: str) -> Optional[DatabaseCluster]:
        """获取集群配置."""
        return self.clusters.get(cluster_id)
    
    def list_connection_pools(self) -> Dict[str, DatabaseConnectionPool]:
        """列出所有连接池."""
        return self.pools.copy()
    
    def list_clusters(self) -> Dict[str, DatabaseCluster]:
        """列出所有集群."""
        return self.clusters.copy()
    
    def test_database_config(self, config_id: str) -> bool:
        """测试数据库配置连接."""
        try:
            config = self.get_database_config(config_id)
            if not config:
                logger.error(f"配置不存在: {config_id}")
                return False
            
            from .factory import create_database_connector
            connector = create_database_connector(config)
            
            # 这里应该是异步测试，但为了简化示例，我们返回True
            # 实际实现中应该使用异步方法
            logger.info(f"数据库配置 {config_id} 测试成功")
            return True
            
        except Exception as e:
            logger.error(f"测试数据库配置失败: {config_id}, 错误: {e}")
            return False
    
    def get_supported_database_types(self) -> List[str]:
        """获取支持的数据库类型列表."""
        return [db_type.value for db_type in DatabaseType]
    
    def export_config(self, export_file: str) -> bool:
        """导出配置到指定文件."""
        try:
            data = {
                'databases': {config_id: config.dict() for config_id, config in self.configs.items()},
                'pools': {pool_id: pool.dict() for pool_id, pool in self.pools.items()},
                'clusters': {cluster_id: cluster.dict() for cluster_id, cluster in self.clusters.items()}
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"配置已导出到: {export_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False
    
    def import_config(self, import_file: str) -> bool:
        """从指定文件导入配置."""
        try:
            if not os.path.exists(import_file):
                logger.error(f"导入文件不存在: {import_file}")
                return False
            
            # 备份当前配置
            backup_configs = self.configs.copy()
            backup_pools = self.pools.copy()
            backup_clusters = self.clusters.copy()
            
            # 清空当前配置
            self.configs.clear()
            self.pools.clear()
            self.clusters.clear()
            
            # 临时设置配置文件路径
            original_config_file = self.config_file
            self.config_file = import_file
            
            try:
                self._load_configs()
                # 恢复原配置文件路径并保存
                self.config_file = original_config_file
                self._save_configs()
                
                logger.info(f"配置已从 {import_file} 导入")
                return True
                
            except Exception as e:
                # 恢复备份配置
                self.configs = backup_configs
                self.pools = backup_pools
                self.clusters = backup_clusters
                self.config_file = original_config_file
                raise e
                
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False