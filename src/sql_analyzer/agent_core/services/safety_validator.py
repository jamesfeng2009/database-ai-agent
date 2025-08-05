"""安全验证器模块 - 提供SQL操作安全性检查和权限验证功能."""

import re
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """风险等级枚举."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserRole(str, Enum):
    """用户角色枚举."""
    ADMIN = "admin"
    DBA = "dba"
    DEVELOPER = "developer"
    READONLY = "readonly"


class OperationType(str, Enum):
    """操作类型枚举."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    INDEX = "index"
    TRUNCATE = "truncate"
    GRANT = "grant"
    REVOKE = "revoke"
    UNKNOWN = "unknown"


class ValidationResult(BaseModel):
    """验证结果模型."""
    is_valid: bool = Field(..., description="是否通过验证")
    risk_level: RiskLevel = Field(..., description="风险等级")
    violations: List[str] = Field(default_factory=list, description="违规项列表")
    warnings: List[str] = Field(default_factory=list, description="警告列表")
    recommendations: List[str] = Field(default_factory=list, description="建议列表")
    audit_info: Dict[str, Any] = Field(default_factory=dict, description="审计信息")


class User(BaseModel):
    """用户模型."""
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    role: UserRole = Field(..., description="用户角色")
    permissions: Set[str] = Field(default_factory=set, description="用户权限集合")
    database_access: Set[str] = Field(default_factory=set, description="可访问的数据库")


class OptimizationPlan(BaseModel):
    """优化计划模型."""
    plan_id: str = Field(default_factory=lambda: str(uuid4()), description="计划ID")
    database: str = Field(..., description="目标数据库")
    operations: List[Dict[str, Any]] = Field(..., description="操作列表")
    estimated_impact: str = Field(..., description="预估影响")
    rollback_plan: Optional[Dict[str, Any]] = Field(None, description="回滚计划")


class AuditLog(BaseModel):
    """审计日志模型."""
    log_id: str = Field(default_factory=lambda: str(uuid4()), description="日志ID")
    user_id: str = Field(..., description="用户ID")
    operation_type: OperationType = Field(..., description="操作类型")
    sql_statement: Optional[str] = Field(None, description="SQL语句")
    database: str = Field(..., description="数据库名")
    risk_level: RiskLevel = Field(..., description="风险等级")
    validation_result: bool = Field(..., description="验证结果")
    violations: List[str] = Field(default_factory=list, description="违规项")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    ip_address: Optional[str] = Field(None, description="IP地址")
    session_id: Optional[str] = Field(None, description="会话ID")


class SafetyValidator:
    """安全验证器 - 提供SQL操作安全性检查和权限验证."""
    
    def __init__(self):
        """初始化安全验证器."""
        self.audit_logs: List[AuditLog] = []
        self._init_security_rules()
    
    def _init_security_rules(self):
        """初始化安全规则."""
        # SQL注入检测模式
        self.sql_injection_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b.*\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(;|\s)(drop|delete|truncate|alter)\s+(table|database|schema)",
            r"(\b(or|and)\b\s*\d+\s*=\s*\d+)",
            r"(\b(or|and)\b\s*['\"].*['\"])",
            r"(--|#|/\*|\*/)",
            r"(\bxp_cmdshell\b|\bsp_executesql\b)",
            r"(\b(script|javascript|vbscript)\b)",
        ]
        
        # 危险操作模式
        self.dangerous_operations = [
            r"\b(drop\s+(table|database|schema|index))\b",
            r"\b(truncate\s+table)\b",
            r"\b(delete\s+from\s+\w+\s*(?:where\s+1\s*=\s*1)?)\b",
            r"\b(update\s+\w+\s+set\s+.*(?:where\s+1\s*=\s*1)?)\b",
            r"\b(grant\s+all)\b",
            r"\b(revoke\s+all)\b",
        ]
        
        # 操作白名单 - 允许的安全操作
        self.operation_whitelist = {
            OperationType.SELECT: ["select", "with", "explain"],
            OperationType.INDEX: ["create index", "drop index"],
            OperationType.UPDATE: ["update statistics", "analyze table"],
        }
        
        # 操作黑名单 - 禁止的危险操作
        self.operation_blacklist = {
            "system_commands": ["xp_cmdshell", "sp_executesql", "exec", "execute"],
            "dangerous_drops": ["drop database", "drop schema", "drop table"],
            "bulk_operations": ["truncate", "delete from", "update.*set.*where 1=1"],
        }
        
        # 角色权限映射
        self.role_permissions = {
            UserRole.ADMIN: {
                "can_execute_ddl": True,
                "can_execute_dml": True,
                "can_drop_objects": True,
                "can_grant_permissions": True,
                "max_risk_level": RiskLevel.CRITICAL,
            },
            UserRole.DBA: {
                "can_execute_ddl": True,
                "can_execute_dml": True,
                "can_drop_objects": False,
                "can_grant_permissions": False,
                "max_risk_level": RiskLevel.HIGH,
            },
            UserRole.DEVELOPER: {
                "can_execute_ddl": False,
                "can_execute_dml": True,
                "can_drop_objects": False,
                "can_grant_permissions": False,
                "max_risk_level": RiskLevel.MEDIUM,
            },
            UserRole.READONLY: {
                "can_execute_ddl": False,
                "can_execute_dml": False,
                "can_drop_objects": False,
                "can_grant_permissions": False,
                "max_risk_level": RiskLevel.LOW,
            },
        }
    
    async def validate_sql_operation(self, sql: str, user: User, database: str = None) -> ValidationResult:
        """
        验证SQL操作的安全性.
        
        Args:
            sql: SQL语句
            user: 用户信息
            database: 数据库名
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"开始验证SQL操作，用户: {user.username}, 数据库: {database}")
        
        violations = []
        warnings = []
        recommendations = []
        risk_level = RiskLevel.LOW
        
        # 1. 检查SQL注入风险
        injection_risk = self._check_sql_injection(sql)
        if injection_risk["detected"]:
            violations.extend(injection_risk["violations"])
            risk_level = RiskLevel.CRITICAL
        
        # 2. 检查危险操作
        dangerous_ops = self._check_dangerous_operations(sql)
        if dangerous_ops["detected"]:
            violations.extend(dangerous_ops["violations"])
            risk_level = RiskLevel.HIGH
        
        # 3. 验证用户权限
        permission_check = self._check_user_permissions(sql, user, database)
        if not permission_check["allowed"]:
            violations.extend(permission_check["violations"])
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # 4. 检查操作白名单和黑名单
        whitelist_check = self._check_operation_lists(sql)
        if whitelist_check["blacklisted"]:
            violations.extend(whitelist_check["violations"])
            risk_level = RiskLevel.CRITICAL
        elif not whitelist_check["whitelisted"]:
            warnings.append("操作未在白名单中，请谨慎执行")
            risk_level = max(risk_level, RiskLevel.MEDIUM)
        
        # 5. 评估整体风险
        final_risk = self._assess_overall_risk(sql, user, risk_level)
        
        # 6. 生成建议
        recommendations = self._generate_recommendations(sql, violations, warnings)
        
        # 7. 记录审计日志
        await self._log_audit_event(user, sql, database, final_risk, len(violations) == 0, violations)
        
        is_valid = len(violations) == 0 and final_risk <= self.role_permissions[user.role]["max_risk_level"]
        
        return ValidationResult(
            is_valid=is_valid,
            risk_level=final_risk,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            audit_info={
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "database": database,
                "sql_hash": hash(sql),
                "validation_timestamp": datetime.now().isoformat(),
            }
        )
    
    async def validate_optimization_plan(self, plan: OptimizationPlan, user: User) -> ValidationResult:
        """
        验证优化计划的安全性.
        
        Args:
            plan: 优化计划
            user: 用户信息
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"开始验证优化计划，计划ID: {plan.plan_id}, 用户: {user.username}")
        
        violations = []
        warnings = []
        recommendations = []
        risk_level = RiskLevel.LOW
        
        # 1. 检查用户是否有执行优化的权限
        if not self._can_execute_optimization(user):
            violations.append(f"用户 {user.username} 没有执行优化操作的权限")
            risk_level = RiskLevel.HIGH
        
        # 2. 检查数据库访问权限
        if plan.database not in user.database_access:
            violations.append(f"用户 {user.username} 没有访问数据库 {plan.database} 的权限")
            risk_level = RiskLevel.HIGH
        
        # 3. 验证每个操作的安全性
        for i, operation in enumerate(plan.operations):
            op_sql = operation.get("sql", "")
            if op_sql:
                sql_validation = await self.validate_sql_operation(op_sql, user, plan.database)
                if not sql_validation.is_valid:
                    violations.extend([f"操作 {i+1}: {v}" for v in sql_validation.violations])
                    risk_level = max(risk_level, sql_validation.risk_level)
                warnings.extend([f"操作 {i+1}: {w}" for w in sql_validation.warnings])
        
        # 4. 检查回滚计划
        if not plan.rollback_plan:
            warnings.append("优化计划缺少回滚方案，建议添加")
            risk_level = max(risk_level, RiskLevel.MEDIUM)
        else:
            rollback_validation = self._validate_rollback_plan(plan.rollback_plan)
            if not rollback_validation["valid"]:
                violations.extend(rollback_validation["issues"])
                risk_level = max(risk_level, RiskLevel.HIGH)
        
        # 5. 评估优化计划的整体风险
        plan_risk = self._assess_plan_risk(plan, user)
        final_risk = max(risk_level, plan_risk)
        
        # 6. 生成建议
        recommendations = self._generate_plan_recommendations(plan, violations, warnings)
        
        # 7. 记录审计日志
        await self._log_optimization_audit(user, plan, final_risk, len(violations) == 0, violations)
        
        is_valid = len(violations) == 0 and final_risk <= self.role_permissions[user.role]["max_risk_level"]
        
        return ValidationResult(
            is_valid=is_valid,
            risk_level=final_risk,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            audit_info={
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "plan_id": plan.plan_id,
                "database": plan.database,
                "operation_count": len(plan.operations),
                "validation_timestamp": datetime.now().isoformat(),
            }
        )
    
    def _check_sql_injection(self, sql: str) -> Dict[str, Any]:
        """检查SQL注入风险."""
        violations = []
        sql_lower = sql.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                violations.append(f"检测到潜在的SQL注入风险: {pattern}")
        
        return {
            "detected": len(violations) > 0,
            "violations": violations
        }
    
    def _check_dangerous_operations(self, sql: str) -> Dict[str, Any]:
        """检查危险操作."""
        violations = []
        sql_lower = sql.lower()
        
        for pattern in self.dangerous_operations:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                violations.append(f"检测到危险操作: {pattern}")
        
        return {
            "detected": len(violations) > 0,
            "violations": violations
        }
    
    def _check_user_permissions(self, sql: str, user: User, database: str) -> Dict[str, Any]:
        """检查用户权限."""
        violations = []
        sql_lower = sql.lower().strip()
        
        # 检查数据库访问权限
        if database and database not in user.database_access:
            violations.append(f"用户 {user.username} 没有访问数据库 {database} 的权限")
        
        # 根据SQL类型检查权限
        operation_type = self._get_operation_type(sql)
        user_permissions = self.role_permissions.get(user.role, {})
        
        if operation_type in [OperationType.CREATE, OperationType.ALTER, OperationType.DROP]:
            if not user_permissions.get("can_execute_ddl", False):
                violations.append(f"用户角色 {user.role.value} 没有执行DDL操作的权限")
        
        if operation_type in [OperationType.INSERT, OperationType.UPDATE, OperationType.DELETE]:
            if not user_permissions.get("can_execute_dml", False):
                violations.append(f"用户角色 {user.role.value} 没有执行DML操作的权限")
        
        if operation_type == OperationType.DROP:
            if not user_permissions.get("can_drop_objects", False):
                violations.append(f"用户角色 {user.role.value} 没有删除数据库对象的权限")
        
        if operation_type in [OperationType.GRANT, OperationType.REVOKE]:
            if not user_permissions.get("can_grant_permissions", False):
                violations.append(f"用户角色 {user.role.value} 没有授权管理的权限")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    def _check_operation_lists(self, sql: str) -> Dict[str, Any]:
        """检查操作白名单和黑名单."""
        violations = []
        sql_lower = sql.lower()
        
        # 检查黑名单
        blacklisted = False
        for category, patterns in self.operation_blacklist.items():
            for pattern in patterns:
                if re.search(pattern, sql_lower, re.IGNORECASE):
                    violations.append(f"操作被黑名单禁止: {category} - {pattern}")
                    blacklisted = True
        
        # 检查白名单
        whitelisted = False
        operation_type = self._get_operation_type(sql)
        if operation_type in self.operation_whitelist:
            for pattern in self.operation_whitelist[operation_type]:
                if pattern.lower() in sql_lower:
                    whitelisted = True
                    break
        
        return {
            "blacklisted": blacklisted,
            "whitelisted": whitelisted,
            "violations": violations
        }
    
    def _get_operation_type(self, sql: str) -> OperationType:
        """获取SQL操作类型."""
        sql_lower = sql.lower().strip()
        
        if sql_lower.startswith("select"):
            return OperationType.SELECT
        elif sql_lower.startswith("insert"):
            return OperationType.INSERT
        elif sql_lower.startswith("update"):
            return OperationType.UPDATE
        elif sql_lower.startswith("delete"):
            return OperationType.DELETE
        elif sql_lower.startswith("create index") or sql_lower.startswith("drop index"):
            return OperationType.INDEX
        elif sql_lower.startswith("create"):
            return OperationType.CREATE
        elif sql_lower.startswith("drop"):
            return OperationType.DROP
        elif sql_lower.startswith("alter"):
            return OperationType.ALTER
        elif sql_lower.startswith("truncate"):
            return OperationType.TRUNCATE
        elif sql_lower.startswith("grant"):
            return OperationType.GRANT
        elif sql_lower.startswith("revoke"):
            return OperationType.REVOKE
        else:
            return OperationType.UNKNOWN
    
    def _assess_overall_risk(self, sql: str, user: User, current_risk: RiskLevel) -> RiskLevel:
        """评估整体风险等级."""
        # 基于用户角色调整风险等级
        if user.role == UserRole.READONLY and current_risk > RiskLevel.LOW:
            return RiskLevel.HIGH
        
        # 基于SQL复杂度调整风险
        if len(sql) > 1000:  # 复杂SQL
            current_risk = max(current_risk, RiskLevel.MEDIUM)
        
        # 基于操作类型调整风险
        operation_type = self._get_operation_type(sql)
        if operation_type in [OperationType.DROP, OperationType.TRUNCATE]:
            current_risk = max(current_risk, RiskLevel.HIGH)
        
        return current_risk
    
    def _generate_recommendations(self, sql: str, violations: List[str], warnings: List[str]) -> List[str]:
        """生成安全建议."""
        recommendations = []
        
        if violations:
            recommendations.append("请修复所有安全违规项后再执行操作")
        
        if warnings:
            recommendations.append("请仔细检查警告项，确保操作安全")
        
        # 基于SQL类型的建议
        operation_type = self._get_operation_type(sql)
        if operation_type in [OperationType.UPDATE, OperationType.DELETE]:
            recommendations.append("建议在执行前备份相关数据")
            recommendations.append("建议使用WHERE子句限制影响范围")
        
        if operation_type == OperationType.DROP:
            recommendations.append("建议在执行前确认对象不再使用")
            recommendations.append("建议创建完整的恢复计划")
        
        return recommendations
    
    def _can_execute_optimization(self, user: User) -> bool:
        """检查用户是否可以执行优化操作."""
        return user.role in [UserRole.ADMIN, UserRole.DBA]
    
    def _validate_rollback_plan(self, rollback_plan: Dict[str, Any]) -> Dict[str, Any]:
        """验证回滚计划."""
        issues = []
        
        if not rollback_plan.get("steps"):
            issues.append("回滚计划缺少具体步骤")
        
        if not rollback_plan.get("validation_queries"):
            issues.append("回滚计划缺少验证查询")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _assess_plan_risk(self, plan: OptimizationPlan, user: User) -> RiskLevel:
        """评估优化计划风险."""
        risk = RiskLevel.LOW
        
        # 基于操作数量评估风险
        if len(plan.operations) > 10:
            risk = max(risk, RiskLevel.MEDIUM)
        if len(plan.operations) > 50:
            risk = max(risk, RiskLevel.HIGH)
        
        # 基于用户角色评估风险
        if user.role == UserRole.DEVELOPER:
            risk = max(risk, RiskLevel.MEDIUM)
        
        return risk
    
    def _generate_plan_recommendations(self, plan: OptimizationPlan, violations: List[str], warnings: List[str]) -> List[str]:
        """生成优化计划建议."""
        recommendations = []
        
        if violations:
            recommendations.append("请修复所有违规项后再执行优化计划")
        
        if not plan.rollback_plan:
            recommendations.append("强烈建议添加详细的回滚计划")
        
        if len(plan.operations) > 20:
            recommendations.append("建议将大型优化计划分解为多个小批次执行")
        
        recommendations.append("建议在非高峰时段执行优化操作")
        recommendations.append("建议在执行前进行充分的测试")
        
        return recommendations
    
    async def _log_audit_event(self, user: User, sql: str, database: str, risk_level: RiskLevel, 
                              validation_result: bool, violations: List[str]):
        """记录审计事件."""
        audit_log = AuditLog(
            user_id=user.user_id,
            operation_type=self._get_operation_type(sql),
            sql_statement=sql,
            database=database or "unknown",
            risk_level=risk_level,
            validation_result=validation_result,
            violations=violations,
        )
        
        self.audit_logs.append(audit_log)
        logger.info(f"记录审计日志: {audit_log.log_id}")
    
    async def _log_optimization_audit(self, user: User, plan: OptimizationPlan, risk_level: RiskLevel,
                                    validation_result: bool, violations: List[str]):
        """记录优化操作审计事件."""
        audit_log = AuditLog(
            user_id=user.user_id,
            operation_type=OperationType.ALTER,  # 优化操作通常是ALTER类型
            sql_statement=f"OPTIMIZATION_PLAN:{plan.plan_id}",
            database=plan.database,
            risk_level=risk_level,
            validation_result=validation_result,
            violations=violations,
        )
        
        self.audit_logs.append(audit_log)
        logger.info(f"记录优化审计日志: {audit_log.log_id}")
    
    def get_audit_logs(self, user_id: str = None, start_time: datetime = None, 
                      end_time: datetime = None) -> List[AuditLog]:
        """获取审计日志."""
        logs = self.audit_logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        return logs
    
    def export_audit_logs(self, format: str = "json") -> str:
        """导出审计日志."""
        if format == "json":
            import json
            return json.dumps([log.dict() for log in self.audit_logs], default=str, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")