#!/usr/bin/env python3
"""数据库适配器扩展使用示例."""

import asyncio
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sql_analyzer.database import (
    DatabaseManager,
    MySQLConfig,
    PostgreSQLConfig,
    SQLiteConfig,
    TiDBConfig,
    LoadBalancingStrategy,
    DatabaseType
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """基本使用示例."""
    logger.info("=== 基本使用示例 ===")
    
    # 创建数据库管理器
    db_manager = DatabaseManager("example_config.json")
    
    try:
        # 1. 添加不同类型的数据库配置
        logger.info("1. 添加数据库配置")
        
        # MySQL配置
        mysql_config = MySQLConfig(
            host="localhost",
            port=3306,
            user="root",
            password="password",
            database="test_db"
        )
        db_manager.add_database_config("mysql_primary", mysql_config)
        
        # PostgreSQL配置
        pg_config = PostgreSQLConfig(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database="test_db"
        )
        db_manager.add_database_config("pg_primary", pg_config)
        
        # SQLite配置（用于演示）
        sqlite_config = SQLiteConfig(database_path="test.db")
        db_manager.add_database_config("sqlite_local", sqlite_config)
        
        # 2. 查看支持的数据库类型
        logger.info("2. 支持的数据库类型")
        supported_types = db_manager.get_supported_database_types()
        logger.info(f"支持的数据库: {supported_types}")
        
        # 3. 获取数据库适配器
        logger.info("3. 获取数据库适配器")
        mysql_adapter = db_manager.get_database_adapter("mysql")
        logger.info(f"MySQL适配器: {mysql_adapter.get_database_name()}")
        
        # 获取优化建议
        suggestions = mysql_adapter.get_optimization_suggestions(["全表扫描"])
        logger.info(f"MySQL优化建议: {len(suggestions)} 条")
        
        # 4. 初始化数据库管理器
        logger.info("4. 初始化数据库管理器")
        await db_manager.initialize()
        
        # 5. 获取系统概览
        overview = db_manager.get_system_overview()
        logger.info(f"系统概览: {overview}")
        
        # 6. 导出配置
        logger.info("6. 导出配置")
        db_manager.export_config("exported_config.json")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
    
    finally:
        await db_manager.shutdown()


async def example_connection_pool():
    """连接池使用示例."""
    logger.info("=== 连接池使用示例 ===")
    
    db_manager = DatabaseManager()
    
    try:
        # 创建多个数据库配置
        configs = []
        for i in range(3):
            config = MySQLConfig(
                host=f"mysql-{i}.example.com",
                port=3306,
                user="app_user",
                password="app_password",
                database="app_db"
            )
            config_id = f"mysql_node_{i}"
            db_manager.add_database_config(config_id, config)
            configs.append(config_id)
        
        # 创建连接池
        success = db_manager.config_manager.create_connection_pool(
            pool_id="mysql_cluster",
            config_ids=configs,
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            max_connections_per_db=20,
            health_check_interval=30
        )
        
        if success:
            logger.info("连接池创建成功")
        
        # 获取连接池配置
        pool_config = db_manager.config_manager.get_connection_pool("mysql_cluster")
        if pool_config:
            logger.info(f"连接池配置: {pool_config.pool_id}, 策略: {pool_config.load_balancing_strategy}")
    
    except Exception as e:
        logger.error(f"连接池示例失败: {e}")
    
    finally:
        await db_manager.shutdown()


async def example_database_cluster():
    """数据库集群使用示例."""
    logger.info("=== 数据库集群使用示例 ===")
    
    db_manager = DatabaseManager()
    
    try:
        # 创建主数据库配置
        primary_config = MySQLConfig(
            host="mysql-primary.example.com",
            port=3306,
            user="app_user",
            password="app_password",
            database="app_db"
        )
        db_manager.add_database_config("mysql_primary", primary_config)
        
        # 创建从数据库配置
        replica_configs = []
        for i in range(2):
            config = MySQLConfig(
                host=f"mysql-replica-{i}.example.com",
                port=3306,
                user="app_user",
                password="app_password",
                database="app_db"
            )
            config_id = f"mysql_replica_{i}"
            db_manager.add_database_config(config_id, config)
            replica_configs.append(config_id)
        
        # 创建集群
        success = db_manager.config_manager.create_cluster(
            cluster_id="mysql_cluster",
            cluster_name="MySQL主从集群",
            primary_config_id="mysql_primary",
            replica_config_ids=replica_configs,
            read_write_split=True,
            auto_failover=True
        )
        
        if success:
            logger.info("数据库集群创建成功")
        
        # 获取集群配置
        cluster_config = db_manager.config_manager.get_cluster("mysql_cluster")
        if cluster_config:
            logger.info(f"集群配置: {cluster_config.cluster_name}")
            logger.info(f"读写分离: {cluster_config.read_write_split}")
            logger.info(f"自动故障转移: {cluster_config.auto_failover}")
    
    except Exception as e:
        logger.error(f"集群示例失败: {e}")
    
    finally:
        await db_manager.shutdown()


async def example_health_monitoring():
    """健康监控示例."""
    logger.info("=== 健康监控示例 ===")
    
    db_manager = DatabaseManager()
    
    try:
        # 添加一些数据库配置
        sqlite_config = SQLiteConfig(database_path=":memory:")
        db_manager.add_database_config("test_db", sqlite_config)
        
        await db_manager.initialize()
        
        # 等待一段时间让健康检查运行
        await asyncio.sleep(2)
        
        # 获取连接状态
        connection_status = db_manager.get_connection_status()
        logger.info(f"连接状态: {len(connection_status)} 个连接")
        
        for conn_id, health in connection_status.items():
            logger.info(f"连接 {conn_id}: {health.status}, 响应时间: {health.response_time_ms}ms")
        
        # 获取健康和不健康的连接
        healthy = db_manager.get_healthy_connections()
        unhealthy = db_manager.get_unhealthy_connections()
        
        logger.info(f"健康连接: {len(healthy)}")
        logger.info(f"不健康连接: {len(unhealthy)}")
        
        # 获取连接统计
        stats = db_manager.get_connection_stats()
        logger.info(f"连接统计: {stats}")
    
    except Exception as e:
        logger.error(f"健康监控示例失败: {e}")
    
    finally:
        await db_manager.shutdown()


async def main():
    """主函数."""
    logger.info("开始数据库适配器扩展使用示例")
    
    try:
        await example_basic_usage()
        await example_connection_pool()
        await example_database_cluster()
        await example_health_monitoring()
        
        logger.info("所有示例完成")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
    
    finally:
        # 清理示例文件
        for file_name in ["example_config.json", "exported_config.json", "test.db"]:
            file_path = Path(file_name)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"已删除示例文件: {file_name}")


if __name__ == "__main__":
    asyncio.run(main())