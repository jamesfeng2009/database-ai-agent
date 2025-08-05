"""仪表板管理器."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..database.database_manager import DatabaseManager
from .models import (
    DashboardConfig,
    ComponentConfig,
    DashboardSnapshot,
    MetricsQuery,
    MetricsQueryResult
)
from .metrics_collector import MetricsCollector
from .dashboard_components import ComponentFactory, DashboardComponent

logger = logging.getLogger(__name__)


class DashboardManager:
    """仪表板管理器 - 统一管理性能仪表板."""
    
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.metrics_collector = MetricsCollector(database_manager)
        self.dashboards = {}  # 简单的内存存储
        self.dashboard_components = {}  # 组件实例缓存
        self.real_time_tasks = {}  # 实时更新任务
        self._initialized = False
    
    async def initialize(self):
        """初始化仪表板管理器."""
        if self._initialized:
            return
        
        try:
            # 初始化数据库管理器
            await self.database_manager.initialize()
            
            # 启动指标收集
            await self.metrics_collector.start_collection()
            
            # 创建默认仪表板
            await self._create_default_dashboards()
            
            self._initialized = True
            logger.info("仪表板管理器初始化完成")
            
        except Exception as e:
            logger.error(f"仪表板管理器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭仪表板管理器."""
        if not self._initialized:
            return
        
        try:
            # 停止实时更新任务
            for task in self.real_time_tasks.values():
                task.cancel()
            
            # 停止指标收集
            await self.metrics_collector.stop_collection()
            
            # 关闭数据库管理器
            await self.database_manager.shutdown()
            
            self._initialized = False
            logger.info("仪表板管理器已关闭")
            
        except Exception as e:
            logger.error(f"仪表板管理器关闭失败: {e}")
    
    async def _create_default_dashboards(self):
        """创建默认仪表板."""
        # 系统概览仪表板
        overview_dashboard = DashboardConfig(
            dashboard_id="system_overview",
            name="系统概览",
            description="数据库系统整体性能概览",
            owner="system",
            is_public=True,
            components=[
                ComponentConfig(
                    component_id="cpu_memory_chart",
                    component_type="chart",
                    title="CPU和内存使用率",
                    position={"x": 0, "y": 0, "width": 6, "height": 4},
                    chart_type="line",
                    metrics=["cpu_usage", "memory_usage"],
                    time_range=3600,
                    refresh_interval=30
                ),
                ComponentConfig(
                    component_id="connections_chart",
                    component_type="chart",
                    title="数据库连接数",
                    position={"x": 6, "y": 0, "width": 6, "height": 4},
                    chart_type="area",
                    metrics=["connections_active"],
                    time_range=3600,
                    refresh_interval=30
                ),
                ComponentConfig(
                    component_id="performance_table",
                    component_type="table",
                    title="性能指标表",
                    position={"x": 0, "y": 4, "width": 8, "height": 4},
                    metrics=["cpu_usage", "memory_usage", "connections_active", "queries_per_second"],
                    refresh_interval=30
                ),
                ComponentConfig(
                    component_id="alert_panel",
                    component_type="alert",
                    title="告警面板",
                    position={"x": 8, "y": 4, "width": 4, "height": 4},
                    alert_rules=[
                        {
                            "rule_id": "high_cpu",
                            "metric": "cpu_usage",
                            "condition": ">",
                            "threshold": 80,
                            "level": "warning"
                        },
                        {
                            "rule_id": "high_memory",
                            "metric": "memory_usage",
                            "condition": ">",
                            "threshold": 90,
                            "level": "error"
                        }
                    ],
                    refresh_interval=15
                )
            ],
            data_sources=self.database_manager.get_healthy_connections(),
            default_time_range=3600,
            auto_refresh=True,
            refresh_interval=30
        )
        
        await self.create_dashboard(overview_dashboard)
        
        # 数据库对比仪表板
        comparison_dashboard = DashboardConfig(
            dashboard_id="database_comparison",
            name="数据库对比",
            description="多数据库性能对比分析",
            owner="system",
            is_public=True,
            components=[
                ComponentConfig(
                    component_id="db_comparison",
                    component_type="comparison",
                    title="数据库性能对比",
                    position={"x": 0, "y": 0, "width": 12, "height": 6},
                    metrics=["cpu_usage", "memory_usage", "connections_active", "queries_per_second", "response_time_avg"],
                    time_range=3600,
                    refresh_interval=60
                ),
                ComponentConfig(
                    component_id="response_time_chart",
                    component_type="chart",
                    title="响应时间对比",
                    position={"x": 0, "y": 6, "width": 12, "height": 4},
                    chart_type="bar",
                    metrics=["response_time_avg", "response_time_p95", "response_time_p99"],
                    time_range=3600,
                    refresh_interval=60
                )
            ],
            data_sources=self.database_manager.get_healthy_connections(),
            default_time_range=3600,
            auto_refresh=True,
            refresh_interval=60
        )
        
        await self.create_dashboard(comparison_dashboard)
    
    async def create_dashboard(self, config: DashboardConfig) -> str:
        """创建仪表板."""
        try:
            # 验证配置
            await self._validate_dashboard_config(config)
            
            # 存储配置
            self.dashboards[config.dashboard_id] = config
            
            # 创建组件实例
            await self._create_dashboard_components(config)
            
            # 启动实时更新（如果启用）
            if config.auto_refresh:
                await self._start_real_time_updates(config.dashboard_id)
            
            logger.info(f"已创建仪表板: {config.name}")
            return config.dashboard_id
            
        except Exception as e:
            logger.error(f"创建仪表板失败: {e}")
            raise
    
    async def update_dashboard(self, dashboard_id: str, config: DashboardConfig) -> bool:
        """更新仪表板."""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"仪表板不存在: {dashboard_id}")
            
            # 验证配置
            await self._validate_dashboard_config(config)
            
            # 停止旧的实时更新
            await self._stop_real_time_updates(dashboard_id)
            
            # 更新配置
            config.updated_at = datetime.now()
            config.version += 1
            self.dashboards[dashboard_id] = config
            
            # 重新创建组件实例
            await self._create_dashboard_components(config)
            
            # 重新启动实时更新
            if config.auto_refresh:
                await self._start_real_time_updates(dashboard_id)
            
            logger.info(f"已更新仪表板: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"更新仪表板失败: {e}")
            return False
    
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """删除仪表板."""
        try:
            if dashboard_id not in self.dashboards:
                return False
            
            # 停止实时更新
            await self._stop_real_time_updates(dashboard_id)
            
            # 清理组件实例
            if dashboard_id in self.dashboard_components:
                del self.dashboard_components[dashboard_id]
            
            # 删除配置
            del self.dashboards[dashboard_id]
            
            logger.info(f"已删除仪表板: {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除仪表板失败: {e}")
            return False
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """获取仪表板配置."""
        return self.dashboards.get(dashboard_id)
    
    async def list_dashboards(self, owner: Optional[str] = None, is_public: Optional[bool] = None) -> List[DashboardConfig]:
        """列出仪表板."""
        dashboards = list(self.dashboards.values())
        
        if owner is not None:
            dashboards = [d for d in dashboards if d.owner == owner]
        
        if is_public is not None:
            dashboards = [d for d in dashboards if d.is_public == is_public]
        
        return dashboards
    
    async def render_dashboard(self, dashboard_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """渲染仪表板数据."""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"仪表板不存在: {dashboard_id}")
            
            config = self.dashboards[dashboard_id]
            components_data = []
            
            # 渲染所有组件
            if dashboard_id in self.dashboard_components:
                for component in self.dashboard_components[dashboard_id]:
                    component_data = await component.get_data(force_refresh=force_refresh)
                    components_data.append(component_data)
            
            return {
                "dashboard_id": dashboard_id,
                "name": config.name,
                "description": config.description,
                "layout": config.layout,
                "components": components_data,
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "auto_refresh": config.auto_refresh,
                    "refresh_interval": config.refresh_interval,
                    "data_sources": config.data_sources,
                    "version": config.version
                }
            }
            
        except Exception as e:
            logger.error(f"渲染仪表板失败: {e}")
            return {
                "dashboard_id": dashboard_id,
                "error": str(e)
            }
    
    async def render_component(self, dashboard_id: str, component_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """渲染单个组件."""
        try:
            if dashboard_id not in self.dashboard_components:
                raise ValueError(f"仪表板不存在: {dashboard_id}")
            
            # 查找组件
            component = None
            for comp in self.dashboard_components[dashboard_id]:
                if comp.config.component_id == component_id:
                    component = comp
                    break
            
            if not component:
                raise ValueError(f"组件不存在: {component_id}")
            
            return await component.get_data(force_refresh=force_refresh)
            
        except Exception as e:
            logger.error(f"渲染组件失败: {e}")
            return {
                "component_id": component_id,
                "error": str(e)
            }
    
    async def query_metrics(self, query: MetricsQuery) -> MetricsQueryResult:
        """查询指标数据."""
        try:
            start_time = datetime.now()
            
            # 执行查询
            metrics_data = await self.metrics_collector.get_metrics(
                database_ids=query.database_ids,
                metric_ids=query.metrics,
                start_time=query.start_time,
                end_time=query.end_time,
                limit=query.limit
            )
            
            # 转换为查询结果格式
            result_data = []
            for metrics in metrics_data:
                data_point = {
                    "database_id": metrics.database_id,
                    "database_type": metrics.database_type,
                    "timestamp": metrics.timestamp.isoformat(),
                    **metrics.metrics
                }
                result_data.append(data_point)
            
            # 应用过滤和排序
            if query.filters:
                result_data = self._apply_query_filters(result_data, query.filters)
            
            if query.order_by:
                reverse = query.order_by.startswith("-")
                order_field = query.order_by.lstrip("-")
                result_data.sort(key=lambda x: x.get(order_field, 0), reverse=reverse)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MetricsQueryResult(
                query=query,
                data=result_data,
                total_records=len(result_data),
                execution_time_ms=execution_time,
                metadata={
                    "data_sources": query.database_ids or self.database_manager.get_healthy_connections(),
                    "query_time": start_time.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"查询指标失败: {e}")
            raise
    
    async def create_snapshot(self, dashboard_id: str, name: str, description: Optional[str] = None) -> str:
        """创建仪表板快照."""
        try:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"仪表板不存在: {dashboard_id}")
            
            # 渲染当前仪表板数据
            dashboard_data = await self.render_dashboard(dashboard_id, force_refresh=True)
            
            # 创建快照
            snapshot = DashboardSnapshot(
                snapshot_id=str(uuid.uuid4()),
                dashboard_id=dashboard_id,
                name=name,
                description=description,
                config=self.dashboards[dashboard_id],
                data=dashboard_data,
                created_by="system"
            )
            
            # 这里应该存储到持久化存储中
            logger.info(f"已创建仪表板快照: {name}")
            return snapshot.snapshot_id
            
        except Exception as e:
            logger.error(f"创建快照失败: {e}")
            raise
    
    async def _validate_dashboard_config(self, config: DashboardConfig):
        """验证仪表板配置."""
        # 验证数据源
        healthy_connections = self.database_manager.get_healthy_connections()
        for data_source in config.data_sources:
            if data_source not in healthy_connections:
                logger.warning(f"数据源不健康或不存在: {data_source}")
        
        # 验证组件配置
        for component_config in config.components:
            # 验证指标是否存在
            for metric_id in component_config.metrics:
                if not self.metrics_collector.get_metric_definition(metric_id):
                    logger.warning(f"未知指标: {metric_id}")
    
    async def _create_dashboard_components(self, config: DashboardConfig):
        """创建仪表板组件实例."""
        components = []
        
        for component_config in config.components:
            try:
                component = ComponentFactory.create_component(
                    component_config,
                    self.metrics_collector
                )
                components.append(component)
            except Exception as e:
                logger.error(f"创建组件失败: {e}")
        
        self.dashboard_components[config.dashboard_id] = components
    
    async def _start_real_time_updates(self, dashboard_id: str):
        """启动实时更新任务."""
        if dashboard_id in self.real_time_tasks:
            return
        
        config = self.dashboards[dashboard_id]
        task = asyncio.create_task(self._real_time_update_loop(dashboard_id, config.refresh_interval))
        self.real_time_tasks[dashboard_id] = task
        logger.info(f"已启动仪表板 {dashboard_id} 的实时更新")
    
    async def _stop_real_time_updates(self, dashboard_id: str):
        """停止实时更新任务."""
        if dashboard_id in self.real_time_tasks:
            task = self.real_time_tasks[dashboard_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.real_time_tasks[dashboard_id]
            logger.info(f"已停止仪表板 {dashboard_id} 的实时更新")
    
    async def _real_time_update_loop(self, dashboard_id: str, refresh_interval: int):
        """实时更新循环."""
        try:
            while True:
                await asyncio.sleep(refresh_interval)
                
                # 强制刷新组件数据
                if dashboard_id in self.dashboard_components:
                    for component in self.dashboard_components[dashboard_id]:
                        try:
                            await component.get_data(force_refresh=True)
                        except Exception as e:
                            logger.error(f"更新组件数据失败: {e}")
                
        except asyncio.CancelledError:
            logger.info(f"仪表板 {dashboard_id} 的实时更新已取消")
            raise
    
    def _apply_query_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用查询过滤条件."""
        filtered_data = []
        for item in data:
            match = True
            for key, value in filters.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                filtered_data.append(item)
        return filtered_data
    
    # 便捷方法
    async def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览."""
        try:
            # 获取数据库管理器概览
            db_overview = self.database_manager.get_system_overview()
            
            # 获取最新指标数据
            latest_metrics = await self.metrics_collector.get_latest_metrics()
            
            # 计算聚合指标
            if latest_metrics:
                total_cpu = sum(m.cpu_usage or 0 for m in latest_metrics.values())
                total_memory = sum(m.memory_usage or 0 for m in latest_metrics.values())
                total_connections = sum(m.connections_active or 0 for m in latest_metrics.values())
                avg_response_time = sum(m.response_time_avg or 0 for m in latest_metrics.values()) / len(latest_metrics)
                
                metrics_overview = {
                    "average_cpu_usage": total_cpu / len(latest_metrics),
                    "average_memory_usage": total_memory / len(latest_metrics),
                    "total_active_connections": total_connections,
                    "average_response_time": avg_response_time,
                    "monitored_databases": len(latest_metrics)
                }
            else:
                metrics_overview = {}
            
            return {
                "database_overview": db_overview,
                "metrics_overview": metrics_overview,
                "dashboard_count": len(self.dashboards),
                "active_real_time_updates": len(self.real_time_tasks),
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取系统概览失败: {e}")
            return {"error": str(e)}