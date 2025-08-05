"""数据库领域词典."""

import json
import os
from typing import Dict, List, Optional, Set
from .models import DomainTerm


class DomainDictionary:
    """数据库领域词典类."""
    
    def __init__(self):
        """初始化领域词典."""
        self._terms: Dict[str, DomainTerm] = {}
        self._synonyms_map: Dict[str, str] = {}  # 同义词映射到标准术语
        self._categories: Set[str] = set()
        self._load_default_terms()
    
    def _load_default_terms(self) -> None:
        """加载默认的数据库术语."""
        default_terms = [
            # SQL操作相关
            DomainTerm(
                term="查询",
                category="sql_operation",
                synonyms=["query", "select", "检索", "搜索"],
                definition="从数据库中检索数据的操作",
                examples=["查询用户表", "执行查询语句"],
                weight=1.0
            ),
            DomainTerm(
                term="慢查询",
                category="performance",
                synonyms=["slow query", "慢SQL", "性能差的查询"],
                definition="执行时间超过阈值的SQL查询",
                examples=["分析慢查询", "优化慢查询"],
                weight=1.5
            ),
            DomainTerm(
                term="索引",
                category="database_object",
                synonyms=["index", "idx", "索引结构"],
                definition="提高查询性能的数据库结构",
                examples=["创建索引", "删除索引", "索引优化"],
                weight=1.2
            ),
            DomainTerm(
                term="执行计划",
                category="performance",
                synonyms=["execution plan", "查询计划", "执行路径"],
                definition="数据库执行SQL语句的详细步骤",
                examples=["查看执行计划", "分析执行计划"],
                weight=1.3
            ),
            DomainTerm(
                term="优化",
                category="operation",
                synonyms=["optimize", "调优", "改进", "提升"],
                definition="改善数据库或查询性能的过程",
                examples=["查询优化", "性能优化"],
                weight=1.4
            ),
            # 性能指标相关
            DomainTerm(
                term="响应时间",
                category="metric",
                synonyms=["response time", "执行时间", "耗时"],
                definition="查询从开始到完成所需的时间",
                examples=["响应时间过长", "平均响应时间"],
                weight=1.1
            ),
            DomainTerm(
                term="吞吐量",
                category="metric",
                synonyms=["throughput", "TPS", "QPS"],
                definition="单位时间内处理的事务或查询数量",
                examples=["提高吞吐量", "系统吞吐量"],
                weight=1.1
            ),
            DomainTerm(
                term="CPU使用率",
                category="metric",
                synonyms=["CPU usage", "CPU利用率", "处理器使用率"],
                definition="CPU资源的使用百分比",
                examples=["CPU使用率过高", "监控CPU使用率"],
                weight=1.0
            ),
            # 数据库对象相关
            DomainTerm(
                term="表",
                category="database_object",
                synonyms=["table", "数据表", "关系表"],
                definition="存储数据的基本结构",
                examples=["用户表", "订单表"],
                weight=1.0
            ),
            DomainTerm(
                term="字段",
                category="database_object",
                synonyms=["column", "列", "属性"],
                definition="表中的数据列",
                examples=["用户ID字段", "名称字段"],
                weight=1.0
            ),
            DomainTerm(
                term="数据库",
                category="database_object",
                synonyms=["database", "db", "库"],
                definition="存储和管理数据的系统",
                examples=["MySQL数据库", "PostgreSQL数据库"],
                weight=1.0
            ),
            # 操作类型相关
            DomainTerm(
                term="监控",
                category="operation",
                synonyms=["monitor", "监视", "观察"],
                definition="持续观察系统状态和性能",
                examples=["性能监控", "实时监控"],
                weight=1.2
            ),
            DomainTerm(
                term="分析",
                category="operation",
                synonyms=["analyze", "analysis", "解析"],
                definition="详细检查和评估数据或性能",
                examples=["性能分析", "查询分析"],
                weight=1.3
            ),
            DomainTerm(
                term="告警",
                category="operation",
                synonyms=["alert", "alarm", "警报", "通知"],
                definition="当系统出现异常时发出的通知",
                examples=["性能告警", "设置告警"],
                weight=1.1
            )
        ]
        
        for term in default_terms:
            self.add_term(term)
    
    def add_term(self, term: DomainTerm) -> None:
        """添加术语到词典."""
        self._terms[term.term] = term
        self._categories.add(term.category)
        
        # 建立同义词映射
        for synonym in term.synonyms:
            self._synonyms_map[synonym.lower()] = term.term
        self._synonyms_map[term.term.lower()] = term.term
    
    def get_term(self, term: str) -> Optional[DomainTerm]:
        """获取术语信息."""
        # 先尝试直接匹配
        if term in self._terms:
            return self._terms[term]
        
        # 再尝试同义词匹配
        normalized_term = self._synonyms_map.get(term.lower())
        if normalized_term and normalized_term in self._terms:
            return self._terms[normalized_term]
        
        return None
    
    def normalize_term(self, term: str) -> Optional[str]:
        """将术语标准化为标准形式."""
        return self._synonyms_map.get(term.lower())
    
    def get_terms_by_category(self, category: str) -> List[DomainTerm]:
        """根据类别获取术语列表."""
        return [term for term in self._terms.values() if term.category == category]
    
    def get_all_categories(self) -> List[str]:
        """获取所有类别."""
        return list(self._categories)
    
    def search_terms(self, query: str, category: Optional[str] = None) -> List[DomainTerm]:
        """搜索术语."""
        query_lower = query.lower()
        results = []
        
        for term in self._terms.values():
            if category and term.category != category:
                continue
            
            # 检查术语本身
            if query_lower in term.term.lower():
                results.append(term)
                continue
            
            # 检查同义词
            if any(query_lower in synonym.lower() for synonym in term.synonyms):
                results.append(term)
                continue
            
            # 检查定义
            if query_lower in term.definition.lower():
                results.append(term)
        
        # 按权重排序
        results.sort(key=lambda x: x.weight, reverse=True)
        return results
    
    def get_related_terms(self, term: str, max_results: int = 5) -> List[DomainTerm]:
        """获取相关术语."""
        base_term = self.get_term(term)
        if not base_term:
            return []
        
        # 获取同类别的术语
        related = self.get_terms_by_category(base_term.category)
        
        # 排除自身
        related = [t for t in related if t.term != base_term.term]
        
        # 按权重排序并限制数量
        related.sort(key=lambda x: x.weight, reverse=True)
        return related[:max_results]
    
    def export_to_json(self, filepath: str) -> None:
        """导出词典到JSON文件."""
        data = {
            "terms": [term.model_dump() for term in self._terms.values()],
            "version": "1.0"
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_json(self, filepath: str) -> None:
        """从JSON文件加载词典."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for term_data in data.get("terms", []):
            term = DomainTerm(**term_data)
            self.add_term(term)
    
    def get_statistics(self) -> Dict[str, int]:
        """获取词典统计信息."""
        return {
            "total_terms": len(self._terms),
            "total_synonyms": len(self._synonyms_map),
            "categories": len(self._categories),
            "terms_by_category": {
                category: len(self.get_terms_by_category(category))
                for category in self._categories
            }
        }