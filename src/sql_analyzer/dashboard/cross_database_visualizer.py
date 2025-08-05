"""跨数据库依赖关系可视化组件."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..database.cross_database_analyzer import CrossDatabaseAnalyzer, CrossDatabaseDependency
from .dashboard_components import DashboardComponent
from .models import ComponentConfig

logger = logging.getLogger(__name__)


class CrossDatabaseVisualizationComponent(DashboardComponent):
    """跨数据库依赖关系可视化组件."""
    
    def __init__(self, config: ComponentConfig, cross_db_analyzer: CrossDatabaseAnalyzer):
        super().__init__(config)
        self.cross_db_analyzer = cross_db_analyzer
        self.cached_data = None
        self.last_update = None
    
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取可视化数据."""
        try:
            # 检查是否需要刷新缓存
            if (not force_refresh and self.cached_data and self.last_update and 
                (datetime.now() - self.last_update).seconds < self.config.refresh_interval):
                return self.cached_data
            
            # 获取依赖关系可视化数据
            visualization_data = await self.cross_db_analyzer.visualize_database_dependencies()
            
            # 增强可视化数据
            enhanced_data = await self._enhance_visualization_data(visualization_data)
            
            # 构建组件数据
            component_data = {
                "component_id": self.config.component_id,
                "component_type": self.config.component_type,
                "title": self.config.title,
                "data": enhanced_data,
                "layout": self._generate_layout_config(),
                "interactions": self._generate_interaction_config(),
                "last_update": datetime.now().isoformat()
            }
            
            # 缓存数据
            self.cached_data = component_data
            self.last_update = datetime.now()
            
            return component_data
            
        except Exception as e:
            logger.error(f"获取跨数据库可视化数据失败: {e}")
            return {
                "component_id": self.config.component_id,
                "error": str(e)
            }
    
    async def _enhance_visualization_data(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强可视化数据."""
        enhanced_data = visualization_data.copy()
        
        # 增强节点数据
        for node in enhanced_data.get("nodes", []):
            node.update({
                "size": self._calculate_node_size(node),
                "color": self._get_node_color(node),
                "icon": self._get_database_icon(node.get("type")),
                "tooltip": self._generate_node_tooltip(node)
            })
        
        # 增强边数据
        for edge in enhanced_data.get("edges", []):
            edge.update({
                "width": self._calculate_edge_width(edge),
                "color": self._get_edge_color(edge),
                "style": self._get_edge_style(edge),
                "tooltip": self._generate_edge_tooltip(edge)
            })
        
        # 添加统计信息
        enhanced_data["statistics"] = await self._calculate_statistics(enhanced_data)
        
        # 添加性能指标
        enhanced_data["performance_metrics"] = await self._collect_performance_metrics(enhanced_data)
        
        return enhanced_data
    
    def _calculate_node_size(self, node: Dict[str, Any]) -> int:
        """计算节点大小."""
        base_size = 30
        
        # 根据数据库状态调整大小
        if node.get("status") == "healthy":
            return base_size + 10
        elif node.get("status") == "degraded":
            return base_size + 5
        else:
            return base_size
    
    def _get_node_color(self, node: Dict[str, Any]) -> str:
        """获取节点颜色."""
        status = node.get("status", "unknown")
        
        color_map = {
            "healthy": "#4CAF50",      # 绿色
            "degraded": "#FF9800",     # 橙色
            "unhealthy": "#F44336",    # 红色
            "unknown": "#9E9E9E"       # 灰色
        }
        
        return color_map.get(status, "#9E9E9E")
    
    def _get_database_icon(self, db_type: str) -> str:
        """获取数据库图标."""
        icon_map = {
            "mysql": "🐬",
            "postgresql": "🐘",
            "oracle": "🔶",
            "sqlserver": "🏢",
            "sqlite": "📁",
            "mongodb": "🍃",
            "redis": "🔴"
        }
        
        return icon_map.get(db_type, "💾")
    
    def _generate_node_tooltip(self, node: Dict[str, Any]) -> str:
        """生成节点提示信息."""
        return f"""
        数据库: {node.get('label', 'Unknown')}
        类型: {node.get('type', 'Unknown')}
        主机: {node.get('host', 'Unknown')}:{node.get('port', 'Unknown')}
        状态: {node.get('status', 'Unknown')}
        """
    
    def _calculate_edge_width(self, edge: Dict[str, Any]) -> int:
        """计算边宽度."""
        strength = edge.get("strength", 0.0)
        frequency = edge.get("frequency", 0)
        
        # 基于强度和频率计算宽度
        base_width = 2
        strength_factor = int(strength * 5)
        frequency_factor = min(int(frequency / 10), 5)
        
        return base_width + strength_factor + frequency_factor
    
    def _get_edge_color(self, edge: Dict[str, Any]) -> str:
        """获取边颜色."""
        performance_impact = edge.get("performance_impact", 0.0)
        
        if performance_impact > 0.7:
            return "#F44336"  # 红色 - 高影响
        elif performance_impact > 0.4:
            return "#FF9800"  # 橙色 - 中等影响
        else:
            return "#4CAF50"  # 绿色 - 低影响
    
    def _get_edge_style(self, edge: Dict[str, Any]) -> str:
        """获取边样式."""
        dependency_type = edge.get("type", "")
        
        style_map = {
            "foreign_key": "solid",
            "view_dependency": "dashed",
            "stored_procedure": "dotted",
            "data_flow": "solid"
        }
        
        return style_map.get(dependency_type, "solid")
    
    def _generate_edge_tooltip(self, edge: Dict[str, Any]) -> str:
        """生成边提示信息."""
        return f"""
        依赖类型: {edge.get('type', 'Unknown')}
        强度: {edge.get('strength', 0.0):.2f}
        频率: {edge.get('frequency', 0)}
        性能影响: {edge.get('performance_impact', 0.0):.2f}
        描述: {edge.get('description', 'No description')}
        """
    
    async def _calculate_statistics(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息."""
        nodes = visualization_data.get("nodes", [])
        edges = visualization_data.get("edges", [])
        
        # 数据库类型统计
        db_types = {}
        for node in nodes:
            db_type = node.get("type", "unknown")
            db_types[db_type] = db_types.get(db_type, 0) + 1
        
        # 状态统计
        status_stats = {}
        for node in nodes:
            status = node.get("status", "unknown")
            status_stats[status] = status_stats.get(status, 0) + 1
        
        # 依赖类型统计
        dependency_types = {}
        for edge in edges:
            dep_type = edge.get("type", "unknown")
            dependency_types[dep_type] = dependency_types.get(dep_type, 0) + 1
        
        # 性能影响统计
        high_impact_deps = len([e for e in edges if e.get("performance_impact", 0) > 0.7])
        medium_impact_deps = len([e for e in edges if 0.4 < e.get("performance_impact", 0) <= 0.7])
        low_impact_deps = len([e for e in edges if e.get("performance_impact", 0) <= 0.4])
        
        return {
            "total_databases": len(nodes),
            "total_dependencies": len(edges),
            "database_types": db_types,
            "status_distribution": status_stats,
            "dependency_types": dependency_types,
            "performance_impact_distribution": {
                "high": high_impact_deps,
                "medium": medium_impact_deps,
                "low": low_impact_deps
            }
        }
    
    async def _collect_performance_metrics(self, visualization_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集性能指标."""
        nodes = visualization_data.get("nodes", [])
        edges = visualization_data.get("edges", [])
        
        # 计算平均性能指标
        total_strength = sum(edge.get("strength", 0) for edge in edges)
        avg_strength = total_strength / len(edges) if edges else 0
        
        total_frequency = sum(edge.get("frequency", 0) for edge in edges)
        avg_frequency = total_frequency / len(edges) if edges else 0
        
        total_impact = sum(edge.get("performance_impact", 0) for edge in edges)
        avg_impact = total_impact / len(edges) if edges else 0
        
        # 识别关键路径
        critical_paths = self._identify_critical_paths(nodes, edges)
        
        return {
            "average_dependency_strength": avg_strength,
            "average_access_frequency": avg_frequency,
            "average_performance_impact": avg_impact,
            "critical_paths": critical_paths,
            "health_score": self._calculate_overall_health_score(nodes)
        }
    
    def _identify_critical_paths(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别关键路径."""
        critical_paths = []
        
        # 找出高影响的依赖链
        high_impact_edges = [e for e in edges if e.get("performance_impact", 0) > 0.7]
        
        for edge in high_impact_edges:
            path = {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "impact": edge.get("performance_impact", 0),
                "frequency": edge.get("frequency", 0),
                "risk_level": "high" if edge.get("performance_impact", 0) > 0.8 else "medium"
            }
            critical_paths.append(path)
        
        return critical_paths
    
    def _calculate_overall_health_score(self, nodes: List[Dict[str, Any]]) -> float:
        """计算整体健康评分."""
        if not nodes:
            return 0.0
        
        healthy_count = len([n for n in nodes if n.get("status") == "healthy"])
        degraded_count = len([n for n in nodes if n.get("status") == "degraded"])
        unhealthy_count = len([n for n in nodes if n.get("status") == "unhealthy"])
        
        # 加权计算健康评分
        health_score = (healthy_count * 1.0 + degraded_count * 0.5 + unhealthy_count * 0.0) / len(nodes)
        
        return health_score
    
    def _generate_layout_config(self) -> Dict[str, Any]:
        """生成布局配置."""
        return {
            "algorithm": "force-directed",
            "iterations": 100,
            "node_repulsion": 1000,
            "edge_attraction": 0.1,
            "gravity": 0.01,
            "enable_zoom": True,
            "enable_pan": True,
            "enable_drag": True,
            "show_labels": True,
            "label_threshold": 0.5
        }
    
    def _generate_interaction_config(self) -> Dict[str, Any]:
        """生成交互配置."""
        return {
            "node_click": {
                "action": "show_details",
                "panel": "node_details"
            },
            "edge_click": {
                "action": "show_dependency_details",
                "panel": "dependency_details"
            },
            "node_hover": {
                "action": "highlight_connections",
                "show_tooltip": True
            },
            "edge_hover": {
                "action": "highlight_path",
                "show_tooltip": True
            },
            "double_click": {
                "action": "focus_subgraph"
            }
        }


class CrossDatabasePerformanceComponent(DashboardComponent):
    """跨数据库性能监控组件."""
    
    def __init__(self, config: ComponentConfig, cross_db_analyzer: CrossDatabaseAnalyzer):
        super().__init__(config)
        self.cross_db_analyzer = cross_db_analyzer
        self.cached_data = None
        self.last_update = None
    
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取性能监控数据."""
        try:
            # 检查缓存
            if (not force_refresh and self.cached_data and self.last_update and 
                (datetime.now() - self.last_update).seconds < self.config.refresh_interval):
                return self.cached_data
            
            # 获取跨数据库事务监控数据
            monitoring_data = await self.cross_db_analyzer.monitor_cross_database_transactions()
            
            # 构建组件数据
            component_data = {
                "component_id": self.config.component_id,
                "component_type": self.config.component_type,
                "title": self.config.title,
                "data": {
                    "transaction_metrics": monitoring_data.get("performance_metrics", {}),
                    "active_transactions": len(monitoring_data.get("active_transactions", [])),
                    "alerts": monitoring_data.get("alerts", []),
                    "recommendations": monitoring_data.get("recommendations", []),
                    "charts": await self._generate_performance_charts(monitoring_data),
                    "summary": await self._generate_performance_summary(monitoring_data)
                },
                "last_update": datetime.now().isoformat()
            }
            
            # 缓存数据
            self.cached_data = component_data
            self.last_update = datetime.now()
            
            return component_data
            
        except Exception as e:
            logger.error(f"获取跨数据库性能监控数据失败: {e}")
            return {
                "component_id": self.config.component_id,
                "error": str(e)
            }
    
    async def _generate_performance_charts(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能图表数据."""
        charts = {}
        
        # 事务数量趋势图
        transaction_metrics = monitoring_data.get("performance_metrics", {})
        
        charts["transaction_count"] = {
            "type": "line",
            "title": "活跃事务数趋势",
            "data": [
                {
                    "database_id": db_id,
                    "value": metrics.get("active_transactions", 0),
                    "timestamp": datetime.now().isoformat()
                }
                for db_id, metrics in transaction_metrics.items()
            ]
        }
        
        # 平均事务时间图
        charts["transaction_time"] = {
            "type": "bar",
            "title": "平均事务执行时间",
            "data": [
                {
                    "database_id": db_id,
                    "value": metrics.get("avg_transaction_time", 0),
                    "unit": "ms"
                }
                for db_id, metrics in transaction_metrics.items()
            ]
        }
        
        # 锁等待和死锁统计
        charts["lock_statistics"] = {
            "type": "stacked_bar",
            "title": "锁等待和死锁统计",
            "data": [
                {
                    "database_id": db_id,
                    "lock_waits": metrics.get("lock_waits", 0),
                    "deadlocks": metrics.get("deadlocks", 0)
                }
                for db_id, metrics in transaction_metrics.items()
            ]
        }
        
        return charts
    
    async def _generate_performance_summary(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能摘要."""
        transaction_metrics = monitoring_data.get("performance_metrics", {})
        alerts = monitoring_data.get("alerts", [])
        
        # 计算总体指标
        total_active_transactions = sum(
            metrics.get("active_transactions", 0) 
            for metrics in transaction_metrics.values()
        )
        
        avg_transaction_time = sum(
            metrics.get("avg_transaction_time", 0) 
            for metrics in transaction_metrics.values()
        ) / len(transaction_metrics) if transaction_metrics else 0
        
        total_lock_waits = sum(
            metrics.get("lock_waits", 0) 
            for metrics in transaction_metrics.values()
        )
        
        total_deadlocks = sum(
            metrics.get("deadlocks", 0) 
            for metrics in transaction_metrics.values()
        )
        
        # 告警统计
        alert_counts = {}
        for alert in alerts:
            severity = alert.get("severity", "info")
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        return {
            "total_active_transactions": total_active_transactions,
            "average_transaction_time": avg_transaction_time,
            "total_lock_waits": total_lock_waits,
            "total_deadlocks": total_deadlocks,
            "alert_counts": alert_counts,
            "monitored_databases": len(transaction_metrics),
            "health_status": "healthy" if not alert_counts.get("error", 0) else "degraded"
        }


class CrossDatabaseQueryAnalysisComponent(DashboardComponent):
    """跨数据库查询分析组件."""
    
    def __init__(self, config: ComponentConfig, cross_db_analyzer: CrossDatabaseAnalyzer):
        super().__init__(config)
        self.cross_db_analyzer = cross_db_analyzer
        self.cached_data = None
        self.last_update = None
    
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取查询分析数据."""
        try:
            # 检查缓存
            if (not force_refresh and self.cached_data and self.last_update and 
                (datetime.now() - self.last_update).seconds < self.config.refresh_interval):
                return self.cached_data
            
            # 获取查询缓存中的分析结果
            query_analyses = list(self.cross_db_analyzer.query_cache.values())
            
            # 构建组件数据
            component_data = {
                "component_id": self.config.component_id,
                "component_type": self.config.component_type,
                "title": self.config.title,
                "data": {
                    "query_summary": await self._generate_query_summary(query_analyses),
                    "performance_distribution": await self._generate_performance_distribution(query_analyses),
                    "optimization_suggestions": await self._aggregate_optimization_suggestions(query_analyses),
                    "cost_analysis": await self._generate_cost_analysis(query_analyses),
                    "recent_queries": await self._get_recent_queries(query_analyses)
                },
                "last_update": datetime.now().isoformat()
            }
            
            # 缓存数据
            self.cached_data = component_data
            self.last_update = datetime.now()
            
            return component_data
            
        except Exception as e:
            logger.error(f"获取跨数据库查询分析数据失败: {e}")
            return {
                "component_id": self.config.component_id,
                "error": str(e)
            }
    
    async def _generate_query_summary(self, query_analyses: List) -> Dict[str, Any]:
        """生成查询摘要."""
        if not query_analyses:
            return {"total_queries": 0}
        
        # 查询类型统计
        query_types = {}
        for query in query_analyses:
            query_type = query.query_type.value
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        # 涉及数据库数量统计
        db_count_distribution = {}
        for query in query_analyses:
            db_count = len(query.involved_databases)
            db_count_distribution[str(db_count)] = db_count_distribution.get(str(db_count), 0) + 1
        
        # 平均成本
        avg_cost = sum(query.estimated_cost for query in query_analyses) / len(query_analyses)
        
        return {
            "total_queries": len(query_analyses),
            "query_types": query_types,
            "database_count_distribution": db_count_distribution,
            "average_estimated_cost": avg_cost
        }
    
    async def _generate_performance_distribution(self, query_analyses: List) -> Dict[str, Any]:
        """生成性能分布数据."""
        if not query_analyses:
            return {}
        
        costs = [query.estimated_cost for query in query_analyses]
        costs.sort()
        
        # 计算百分位数
        def percentile(data, p):
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        return {
            "cost_percentiles": {
                "p50": percentile(costs, 50),
                "p75": percentile(costs, 75),
                "p90": percentile(costs, 90),
                "p95": percentile(costs, 95),
                "p99": percentile(costs, 99)
            },
            "cost_histogram": self._generate_histogram(costs, 10)
        }
    
    def _generate_histogram(self, data: List[float], bins: int) -> List[Dict[str, Any]]:
        """生成直方图数据."""
        if not data:
            return []
        
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / bins
        
        histogram = []
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            count = len([x for x in data if bin_start <= x < bin_end])
            
            histogram.append({
                "range": f"{bin_start:.2f}-{bin_end:.2f}",
                "count": count
            })
        
        return histogram
    
    async def _aggregate_optimization_suggestions(self, query_analyses: List) -> Dict[str, Any]:
        """聚合优化建议."""
        all_suggestions = []
        for query in query_analyses:
            all_suggestions.extend(query.optimization_suggestions)
        
        # 统计建议频率
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # 按频率排序
        top_suggestions = sorted(
            suggestion_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_suggestions": len(all_suggestions),
            "unique_suggestions": len(suggestion_counts),
            "top_suggestions": [
                {"suggestion": suggestion, "frequency": count}
                for suggestion, count in top_suggestions
            ]
        }
    
    async def _generate_cost_analysis(self, query_analyses: List) -> Dict[str, Any]:
        """生成成本分析."""
        if not query_analyses:
            return {}
        
        # 按查询类型分组分析成本
        cost_by_type = {}
        for query in query_analyses:
            query_type = query.query_type.value
            if query_type not in cost_by_type:
                cost_by_type[query_type] = []
            cost_by_type[query_type].append(query.estimated_cost)
        
        # 计算每种类型的平均成本
        avg_cost_by_type = {}
        for query_type, costs in cost_by_type.items():
            avg_cost_by_type[query_type] = sum(costs) / len(costs)
        
        return {
            "cost_by_query_type": avg_cost_by_type,
            "highest_cost_queries": sorted(
                [(query.query_id, query.estimated_cost) for query in query_analyses],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    async def _get_recent_queries(self, query_analyses: List, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的查询."""
        # 由于我们没有时间戳，这里返回最后几个查询
        recent_queries = query_analyses[-limit:] if len(query_analyses) > limit else query_analyses
        
        return [
            {
                "query_id": query.query_id,
                "query_type": query.query_type.value,
                "involved_databases": len(query.involved_databases),
                "estimated_cost": query.estimated_cost,
                "optimization_count": len(query.optimization_suggestions)
            }
            for query in recent_queries
        ]