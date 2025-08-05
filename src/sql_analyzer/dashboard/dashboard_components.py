"""仪表板组件实现."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import (
    ComponentConfig,
    ComponentType,
    ChartType,
    PerformanceMetrics,
    MetricsQuery,
    MetricsQueryResult,
    Alert,
    AlertLevel
)
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class DashboardComponent(ABC):
    """仪表板组件基类."""
    
    def __init__(self, config: ComponentConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        self.last_update = None
        self.cached_data = None
    
    @abstractmethod
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染组件数据."""
        pass
    
    @abstractmethod
    def get_component_type(self) -> ComponentType:
        """获取组件类型."""
        pass
    
    async def should_refresh(self) -> bool:
        """判断是否需要刷新数据."""
        if self.last_update is None:
            return True
        
        refresh_interval = self.config.refresh_interval
        return (datetime.now() - self.last_update).total_seconds() >= refresh_interval
    
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取组件数据."""
        if force_refresh or await self.should_refresh():
            self.cached_data = await self.render()
            self.last_update = datetime.now()
        
        return self.cached_data or {}
    
    def _apply_filters(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用过滤条件."""
        if not self.config.filters:
            return data
        
        filtered_data = []
        for item in data:
            match = True
            for key, value in self.config.filters.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                filtered_data.append(item)
        
        return filtered_data
    
    def _group_data(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按指定字段分组数据."""
        if not self.config.group_by:
            return {"default": data}
        
        grouped = {}
        for item in data:
            group_key = "_".join([str(item.get(field, "unknown")) for field in self.config.group_by])
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(item)
        
        return grouped


class MetricsChart(DashboardComponent):
    """指标图表组件."""
    
    def get_component_type(self) -> ComponentType:
        return ComponentType.CHART
    
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染图表数据."""
        try:
            # 获取时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=self.config.time_range)
            
            # 查询指标数据
            metrics_data = await self.metrics_collector.get_metrics(
                metric_ids=self.config.metrics,
                start_time=start_time,
                end_time=end_time
            )
            
            # 转换为图表数据格式
            chart_data = self._convert_to_chart_data(metrics_data)
            
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "chart_type": self.config.chart_type.value if self.config.chart_type else ChartType.LINE.value,
                "data": chart_data,
                "metadata": {
                    "time_range": self.config.time_range,
                    "metrics": self.config.metrics,
                    "last_update": datetime.now().isoformat(),
                    "data_points": len(chart_data.get("series", [{}])[0].get("data", []))
                }
            }
            
        except Exception as e:
            logger.error(f"渲染图表组件失败: {e}")
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "error": str(e)
            }
    
    def _convert_to_chart_data(self, metrics_data: List[PerformanceMetrics]) -> Dict[str, Any]:
        """转换为图表数据格式."""
        if not metrics_data:
            return {"series": [], "categories": []}
        
        # 按数据库分组
        database_groups = {}
        for metrics in metrics_data:
            if metrics.database_id not in database_groups:
                database_groups[metrics.database_id] = []
            database_groups[metrics.database_id].append(metrics)
        
        if not database_groups:
            return {"series": [], "categories": []}
        
        # 生成时间轴
        timestamps = sorted(list(set([m.timestamp for m in metrics_data])))
        if not timestamps:
            return {"series": [], "categories": []}
            
        categories = [ts.strftime("%H:%M:%S") for ts in timestamps]
        
        # 生成数据系列
        series = []
        
        for database_id, db_metrics in database_groups.items():
            for metric_id in self.config.metrics:
                # 创建数据点
                data_points = []
                for timestamp in timestamps:
                    # 查找对应时间点的数据
                    value = None
                    for metrics in db_metrics:
                        if metrics.timestamp == timestamp:
                            value = metrics.metrics.get(metric_id)
                            break
                    data_points.append(value if value is not None else 0)
                
                series.append({
                    "name": f"{database_id}_{metric_id}",
                    "data": data_points,
                    "database_id": database_id,
                    "metric_id": metric_id
                })
        
        return {
            "series": series,
            "categories": categories,
            "chart_type": self.config.chart_type.value if self.config.chart_type else ChartType.LINE.value
        }


class PerformanceTable(DashboardComponent):
    """性能表格组件."""
    
    def get_component_type(self) -> ComponentType:
        return ComponentType.TABLE
    
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染表格数据."""
        try:
            # 获取最新的指标数据
            latest_metrics = await self.metrics_collector.get_latest_metrics()
            
            # 转换为表格数据
            table_data = self._convert_to_table_data(latest_metrics)
            
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "data": table_data,
                "metadata": {
                    "metrics": self.config.metrics,
                    "last_update": datetime.now().isoformat(),
                    "row_count": len(table_data.get("rows", []))
                }
            }
            
        except Exception as e:
            logger.error(f"渲染表格组件失败: {e}")
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "error": str(e)
            }
    
    def _convert_to_table_data(self, latest_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """转换为表格数据格式."""
        if not latest_metrics:
            return {"columns": [], "rows": []}
        
        # 定义列
        columns = [
            {"key": "database_id", "title": "数据库ID", "type": "string"},
            {"key": "database_type", "title": "数据库类型", "type": "string"},
            {"key": "timestamp", "title": "更新时间", "type": "datetime"}
        ]
        
        # 添加指标列
        for metric_id in self.config.metrics:
            metric_def = self.metrics_collector.get_metric_definition(metric_id)
            columns.append({
                "key": metric_id,
                "title": metric_def.name if metric_def else metric_id,
                "type": "number",
                "unit": metric_def.unit.value if metric_def else ""
            })
        
        # 生成行数据
        rows = []
        for database_id, metrics in latest_metrics.items():
            row = {
                "database_id": database_id,
                "database_type": metrics.database_type,
                "timestamp": metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 添加指标值
            for metric_id in self.config.metrics:
                value = metrics.metrics.get(metric_id)
                if isinstance(value, float):
                    row[metric_id] = round(value, 2)
                else:
                    row[metric_id] = value
            
            rows.append(row)
        
        return {
            "columns": columns,
            "rows": rows
        }


class DatabaseComparison(DashboardComponent):
    """数据库对比组件."""
    
    def get_component_type(self) -> ComponentType:
        return ComponentType.COMPARISON
    
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染对比数据."""
        try:
            # 获取聚合指标数据
            aggregated_data = await self.metrics_collector.get_aggregated_metrics(
                metric_ids=self.config.metrics,
                time_range=self.config.time_range,
                aggregation="avg",
                group_by_database=True
            )
            
            # 转换为对比数据格式
            comparison_data = self._convert_to_comparison_data(aggregated_data)
            
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "data": comparison_data,
                "metadata": {
                    "metrics": self.config.metrics,
                    "time_range": self.config.time_range,
                    "last_update": datetime.now().isoformat(),
                    "database_count": len(aggregated_data)
                }
            }
            
        except Exception as e:
            logger.error(f"渲染对比组件失败: {e}")
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "error": str(e)
            }
    
    def _convert_to_comparison_data(self, aggregated_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """转换为对比数据格式."""
        if not aggregated_data:
            return {"databases": [], "metrics": [], "comparison_matrix": []}
        
        databases = list(aggregated_data.keys())
        metrics = self.config.metrics
        
        # 生成对比矩阵
        comparison_matrix = []
        for metric_id in metrics:
            metric_row = {"metric_id": metric_id}
            metric_def = self.metrics_collector.get_metric_definition(metric_id)
            metric_row["metric_name"] = metric_def.name if metric_def else metric_id
            metric_row["unit"] = metric_def.unit.value if metric_def else ""
            
            # 添加各数据库的值
            values = []
            for database_id in databases:
                value = aggregated_data[database_id].get(metric_id, 0)
                metric_row[database_id] = round(value, 2) if isinstance(value, float) else value
                values.append(value)
            
            # 计算统计信息
            if values:
                metric_row["min"] = min(values)
                metric_row["max"] = max(values)
                metric_row["avg"] = sum(values) / len(values)
                metric_row["range"] = max(values) - min(values)
            
            comparison_matrix.append(metric_row)
        
        return {
            "databases": databases,
            "metrics": metrics,
            "comparison_matrix": comparison_matrix
        }


class AlertPanel(DashboardComponent):
    """告警面板组件."""
    
    def __init__(self, config: ComponentConfig, metrics_collector: MetricsCollector):
        super().__init__(config, metrics_collector)
        self.active_alerts = []  # 简单的内存存储
    
    def get_component_type(self) -> ComponentType:
        return ComponentType.ALERT
    
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染告警数据."""
        try:
            # 检查告警规则
            await self._check_alert_rules()
            
            # 获取活跃告警
            active_alerts = self._get_active_alerts()
            
            # 按级别分组
            alerts_by_level = self._group_alerts_by_level(active_alerts)
            
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "data": {
                    "active_alerts": active_alerts,
                    "alerts_by_level": alerts_by_level,
                    "total_count": len(active_alerts)
                },
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "alert_rules_count": len(self.config.alert_rules)
                }
            }
            
        except Exception as e:
            logger.error(f"渲染告警组件失败: {e}")
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "error": str(e)
            }
    
    async def _check_alert_rules(self):
        """检查告警规则."""
        if not self.config.alert_rules:
            return
        
        # 获取最新指标数据
        latest_metrics = await self.metrics_collector.get_latest_metrics()
        
        for rule in self.config.alert_rules:
            await self._evaluate_alert_rule(rule, latest_metrics)
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any], latest_metrics: Dict[str, PerformanceMetrics]):
        """评估单个告警规则."""
        try:
            rule_id = rule.get("rule_id")
            metric = rule.get("metric")
            condition = rule.get("condition", ">")
            threshold = rule.get("threshold", 0)
            level = AlertLevel(rule.get("level", "warning"))
            
            for database_id, metrics in latest_metrics.items():
                current_value = metrics.metrics.get(metric)
                if current_value is None:
                    continue
                
                # 评估条件
                alert_triggered = False
                if condition == ">" and current_value > threshold:
                    alert_triggered = True
                elif condition == "<" and current_value < threshold:
                    alert_triggered = True
                elif condition == ">=" and current_value >= threshold:
                    alert_triggered = True
                elif condition == "<=" and current_value <= threshold:
                    alert_triggered = True
                elif condition == "==" and current_value == threshold:
                    alert_triggered = True
                
                if alert_triggered:
                    # 创建或更新告警
                    alert = Alert(
                        alert_id=f"{rule_id}_{database_id}_{metric}",
                        rule_id=rule_id,
                        database_id=database_id,
                        metric=metric,
                        current_value=current_value,
                        threshold=threshold,
                        level=level,
                        message=f"数据库 {database_id} 的 {metric} 指标异常: {current_value} {condition} {threshold}"
                    )
                    
                    # 检查是否已存在
                    existing_alert = None
                    for existing in self.active_alerts:
                        if existing.alert_id == alert.alert_id:
                            existing_alert = existing
                            break
                    
                    if existing_alert:
                        # 更新现有告警
                        existing_alert.current_value = current_value
                        existing_alert.last_seen = datetime.now()
                    else:
                        # 添加新告警
                        self.active_alerts.append(alert)
                        logger.warning(f"触发告警: {alert.message}")
                
        except Exception as e:
            logger.error(f"评估告警规则失败: {e}")
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警."""
        active_alerts = []
        current_time = datetime.now()
        
        # 清理过期告警（5分钟未更新的告警视为已解决）
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - alert.last_seen).total_seconds() < 300
        ]
        
        for alert in self.active_alerts:
            active_alerts.append({
                "alert_id": alert.alert_id,
                "database_id": alert.database_id,
                "metric": alert.metric,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "level": alert.level.value,
                "message": alert.message,
                "first_seen": alert.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": alert.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": str(current_time - alert.first_seen)
            })
        
        return active_alerts
    
    def _group_alerts_by_level(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """按级别分组告警."""
        grouped = {level.value: 0 for level in AlertLevel}
        
        for alert in alerts:
            level = alert.get("level", "info")
            if level in grouped:
                grouped[level] += 1
        
        return grouped


class GaugeComponent(DashboardComponent):
    """仪表盘组件."""
    
    def get_component_type(self) -> ComponentType:
        return ComponentType.GAUGE
    
    async def render(self, **kwargs) -> Dict[str, Any]:
        """渲染仪表盘数据."""
        try:
            # 获取最新的指标数据
            latest_metrics = await self.metrics_collector.get_latest_metrics()
            
            # 转换为仪表盘数据
            gauge_data = self._convert_to_gauge_data(latest_metrics)
            
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "data": gauge_data,
                "metadata": {
                    "metrics": self.config.metrics,
                    "last_update": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"渲染仪表盘组件失败: {e}")
            return {
                "component_id": self.config.component_id,
                "component_type": self.get_component_type().value,
                "title": self.config.title,
                "error": str(e)
            }
    
    def _convert_to_gauge_data(self, latest_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """转换为仪表盘数据格式."""
        if not latest_metrics or not self.config.metrics:
            return {"value": 0, "max": 100, "unit": ""}
        
        # 取第一个指标作为仪表盘显示
        metric_id = self.config.metrics[0]
        
        # 计算所有数据库的平均值
        values = []
        for metrics in latest_metrics.values():
            value = metrics.metrics.get(metric_id)
            if isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return {"value": 0, "max": 100, "unit": ""}
        
        avg_value = sum(values) / len(values)
        
        # 获取指标定义以确定单位和最大值
        metric_def = self.metrics_collector.get_metric_definition(metric_id)
        unit = metric_def.unit.value if metric_def else ""
        
        # 根据指标类型设置最大值
        if unit == "%":
            max_value = 100
        elif "usage" in metric_id.lower():
            max_value = 100
        else:
            max_value = max(values) * 1.2 if values else 100
        
        return {
            "value": round(avg_value, 2),
            "max": max_value,
            "unit": unit,
            "metric_name": metric_def.name if metric_def else metric_id
        }


class ComponentFactory:
    """组件工厂类."""
    
    _component_classes = {
        ComponentType.CHART: MetricsChart,
        ComponentType.TABLE: PerformanceTable,
        ComponentType.COMPARISON: DatabaseComparison,
        ComponentType.ALERT: AlertPanel,
        ComponentType.GAUGE: GaugeComponent
    }
    
    @classmethod
    def create_component(
        cls,
        config: ComponentConfig,
        metrics_collector: MetricsCollector
    ) -> DashboardComponent:
        """创建组件实例."""
        component_class = cls._component_classes.get(config.component_type)
        if not component_class:
            raise ValueError(f"不支持的组件类型: {config.component_type}")
        
        return component_class(config, metrics_collector)
    
    @classmethod
    def register_component(cls, component_type: ComponentType, component_class: type):
        """注册新的组件类型."""
        cls._component_classes[component_type] = component_class
    
    @classmethod
    def get_supported_types(cls) -> List[ComponentType]:
        """获取支持的组件类型."""
        return list(cls._component_classes.keys())