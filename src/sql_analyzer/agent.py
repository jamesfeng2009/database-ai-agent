"""基于 OpenAI API 的 SQL 分析智能体"""

import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import httpx
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    httpx = None

from .analyzer_base import BaseSQLAnalyzer, get_default_system_message
from .config import OpenAIConfig, OllamaConfig
from .models import (
    OptimizationSuggestion,
    PerformanceIssue,
    SQLAnalysisRequest,
    SQLAnalysisResponse,
)
from .tools import (
    calculate_performance_score,
    detect_performance_issues,
    format_analysis_request,
    generate_optimization_suggestions,
)

# 配置日志
logger = logging.getLogger(__name__)


class StreamingDisplay:
    """流式显示工具类，用于优化AI流式响应的显示效果."""
    
    def __init__(self, max_line_length: int = 80, show_progress: bool = True) -> None:
        """初始化流式显示工具.
        
        Args:
            max_line_length: 每行最大显示字符数
            show_progress: 是否显示进度信息
        """
        self.max_line_length = max_line_length
        self.show_progress = show_progress
        self.display_buffer = ""
        self.line_length = 0
        self.char_count = 0
        self.start_time = time.time()
        
    def start(self, message: str = "🤖 正在生成SQL分析结果...") -> None:
        """开始流式显示.
        
        Args:
            message: 开始显示的消息
        """
        print(message)
        print("=" * 60)
        self.start_time = time.time()
        
    def add_token(self, token: str) -> None:
        """添加新的token到显示缓冲区.
        
        Args:
            token: 新的token字符串
        """
        self.char_count += len(token)
        self.display_buffer += token
        self.line_length += len(token)
        
        # 如果遇到换行符，立即换行显示
        if '\n' in token:
            self._handle_newline()
        # 如果行太长，智能换行
        elif self.line_length > self.max_line_length:
            self._handle_line_break()
        else:
            # 在同一行更新显示
            self._update_current_line()
            
    def _handle_newline(self) -> None:
        """处理换行符."""
        lines = self.display_buffer.split('\n')
        for i, display_line in enumerate(lines[:-1]):
            # 清理当前行并显示完整内容
            self._clear_line()
            print(display_line.strip())
        
        # 保留最后一行作为新的缓冲区
        self.display_buffer = lines[-1] if lines else ""
        self.line_length = len(self.display_buffer)
        
        # 如果最后一行有内容，显示它
        if self.display_buffer.strip():
            self._update_current_line()
            
    def _handle_line_break(self) -> None:
        """处理行太长的情况，智能换行."""
        # 在合适的位置换行（优先在空格处）
        break_point = self.max_line_length
        for i in range(self.max_line_length - 1, max(0, self.max_line_length - 20), -1):
            if i < len(self.display_buffer) and self.display_buffer[i] == ' ':
                break_point = i + 1
                break
        
        # 显示当前行并换行
        current_line = self.display_buffer[:break_point].strip()
        self._clear_line()
        print(current_line)
        
        # 剩余内容作为新行的开始
        self.display_buffer = self.display_buffer[break_point:]
        self.line_length = len(self.display_buffer)
        
        # 显示新行开始的内容
        if self.display_buffer.strip():
            self._update_current_line()
            
    def _update_current_line(self) -> None:
        """更新当前行显示."""
        if self.show_progress:
            # 计算显示时间
            elapsed = time.time() - self.start_time
            status = f" [{self.char_count}字符 {elapsed:.1f}s]"
            available_width = self.max_line_length - len(status)
            
            if len(self.display_buffer) <= available_width:
                print(f"\r{self.display_buffer}{status}", end="", flush=True)
            else:
                # 截断显示并加省略号
                truncated = self.display_buffer[:available_width-3] + "..."
                print(f"\r{truncated}{status}", end="", flush=True)
        else:
            print(f"\r{self.display_buffer}", end="", flush=True)
            
    def _clear_line(self) -> None:
        """清理当前行."""
        print(f"\r{' ' * self.max_line_length}\r", end="")
        
    def finish(self) -> None:
        """完成流式显示."""
        # 确保最后的内容被完整显示
        if self.display_buffer.strip():
            self._clear_line()
            print(self.display_buffer.strip())
        else:
            print()  # 确保光标在新行
            
        # 显示完成信息
        elapsed = time.time() - self.start_time
        print("=" * 60)
        print(f"✅ 分析完成 (共生成 {self.char_count} 个字符，耗时 {elapsed:.1f}s)")
        print()
        
    def error_cleanup(self) -> None:
        """错误时的清理工作."""
        if self.display_buffer:
            self._clear_line()
            print()


class SQLAnalyzerAgent(BaseSQLAnalyzer):
    """SQL 性能分析智能体.
    
    基于 OpenAI API 的智能体，专门用于分析 MySQL 慢 SQL 查询的性能问题
    并提供优化建议。
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        name: str = "sql_analyzer",
        system_message: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        stream: bool = False,
        show_streaming_progress: bool = True,
    ) -> None:
        """初始化 SQL 分析智能体.
        
        Args:
            api_key: OpenAI API 密钥
            model: 模型名称
            base_url: API 基础 URL（可选）
            name: 智能体名称
            system_message: 系统提示消息，如果为空则使用默认消息
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            stream: 是否使用流式响应
            show_streaming_progress: 是否显示流式响应进度
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("需要安装 openai 包: pip install openai")
        
        super().__init__(model, name, system_message, timeout)
        
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.stream = stream
        self.show_streaming_progress = show_streaming_progress
        
        print(f"🤖️ 初始化 SQL 分析智能体 : 使用的模型为{self.model}")
        if self.stream:
            progress_status = "启用" if self.show_streaming_progress else "禁用"
            print(f"⚡ 流式响应: 启用 (进度显示: {progress_status})")
        
        # 创建 OpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            max_retries=max_retries,
        )
    
    @classmethod
    def from_config(cls, config: OpenAIConfig, name: str = "sql_analyzer") -> "SQLAnalyzerAgent":
        """从配置创建智能体.
        
        Args:
            config: OpenAI 配置
            name: 智能体名称
            
        Returns:
            配置好的 SQL 分析智能体
        """
        return cls(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            name=name,
            timeout=config.timeout,
            max_retries=config.max_retries,
            stream=config.stream,
            show_streaming_progress=config.show_streaming_progress,
        )
    
    def _get_default_system_message(self) -> str:
        """获取默认的系统提示消息.
        
        Returns:
            系统提示消息字符串
        """
        return get_default_system_message()
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试 API 连接.
        
        Returns:
            包含连接测试结果的字典
        """
        result = {
            "success": False,
            "error": None,
            "details": {},
        }
        
        try:
            logger.info(f"开始测试 API 连接: {self.base_url}")
            
            # 发送一个简单的测试请求
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "测试连接"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            result["success"] = True
            result["details"] = {
                "model": self.model,
                "base_url": self.base_url,
                "response_id": response.id if hasattr(response, 'id') else None,
                "usage": response.usage.dict() if hasattr(response, 'usage') and response.usage else None,
            }
            logger.info("API 连接测试成功")
            
        except Exception as e:
            result["error"] = str(e)
            result["details"] = {
                "model": self.model,
                "base_url": self.base_url,
                "error_type": type(e).__name__,
            }
            logger.error(f"API 连接测试失败: {e}")
            
            # 提供具体的错误诊断
            if "Connection" in str(e) or "timeout" in str(e).lower():
                result["diagnosis"] = "网络连接问题，请检查网络连接和防火墙设置"
            elif "401" in str(e) or "Unauthorized" in str(e):
                result["diagnosis"] = "API 密钥无效，请检查 API 密钥是否正确"
            elif "404" in str(e) or "Not Found" in str(e):
                result["diagnosis"] = "API 端点不存在，请检查 base_url 是否正确"
            elif "403" in str(e) or "Forbidden" in str(e):
                result["diagnosis"] = "API 访问被拒绝，请检查 API 密钥权限"
            elif "429" in str(e) or "Rate limit" in str(e):
                result["diagnosis"] = "API 调用频率限制，请稍后重试"
            elif "model" in str(e).lower():
                result["diagnosis"] = f"模型 '{self.model}' 不存在或不可用，请检查模型名称"
            else:
                result["diagnosis"] = "未知错误，请检查 API 配置"
        
        return result

    async def _handle_streaming_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        """处理 OpenAI API 流式响应.
        
        Args:
            model: 模型名称
            messages: 消息列表
            
        Returns:
            完整的响应文本
        """
        full_response = ""
        display = StreamingDisplay(show_progress=self.show_streaming_progress)
        
        try:
            display.start()
            
            # 使用 OpenAI 流式 API
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    display.add_token(token)
            
            display.finish()
            
        except Exception as e:
            display.error_cleanup()
            logger.error(f"流式响应处理失败: {e}")
            raise
        
        return full_response

    async def analyze_sql(self, request: SQLAnalysisRequest) -> SQLAnalysisResponse:
        """智能体分析 SQL 性能的核心方法
        
        Args:
            request: SQL 分析请求
            
        Returns:
            SQL 分析响应
        """
        # 首先使用工具函数进行基础分析
        issues = detect_performance_issues(request)
        suggestions = generate_optimization_suggestions(request, issues)
        score = calculate_performance_score(request, issues)
        
        # 准备给 AI 的输入
        formatted_request = format_analysis_request(request)
        
        # 构造给 AI 的消息
        user_message = f"""请分析以下 SQL 查询的性能：

        {formatted_request}
        
        请提供：
        1. 详细的性能分析
        2. 执行计划解读
        3. 发现的问题总结
        4. 优化建议的详细说明
        
        基础分析结果：
        - 性能评分：{score}/100
        - 发现问题数量：{len(issues)}
        - 优化建议数量：{len(suggestions)}
        """
        
        # 使用 OpenAI API 进行深度分析
        detailed_analysis = ""
        try:
            logger.info(f"开始调用 AI API: model={self.model}")
            
            if self.stream:
                # 流式响应处理
                detailed_analysis = await self._handle_streaming_response(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message}
                    ]
                )
            else:
                # 非流式响应处理
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                detailed_analysis = response.choices[0].message.content or "无法获取详细分析"
            
            logger.info("AI API 调用成功")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"AI API 调用失败: {error_msg}")
            
            # 提供更详细的错误信息和诊断
            diagnosis = ""
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                diagnosis = "网络连接超时，请检查网络连接"
            elif "401" in error_msg:
                diagnosis = "API 密钥认证失败，请检查密钥是否正确"
            elif "404" in error_msg:
                diagnosis = f"API 端点不存在，请检查 base_url: {self.base_url}"
            elif "403" in error_msg:
                diagnosis = "API 访问权限不足"
            elif "429" in error_msg:
                diagnosis = "API 调用频率限制，请稍后重试"
            elif "model" in error_msg.lower():
                diagnosis = f"模型 '{self.model}' 不可用"
            else:
                diagnosis = f"未知 API 错误: {type(e).__name__}"
            
            detailed_analysis = f"""AI 分析失败: {error_msg}

错误诊断: {diagnosis}

配置信息:
- 模型: {self.model}
- API 端点: {self.base_url}
- 超时设置: {self.timeout}秒

建议:
1. 验证 API 密钥和端点配置
2. 检查网络连接和防火墙设置
3. 如果是 DeepSeek API，确保 base_url 为 https://api.deepseek.com

使用基础分析结果。"""
        
        # 生成执行计划分析
        execution_plan_analysis = self._generate_execution_plan_analysis(request)
        
        # 生成总结
        summary = self._generate_summary(issues, suggestions, score)
        
        return SQLAnalysisResponse(
            summary=summary,
            performance_score=score,
            issues=issues,
            suggestions=suggestions,
            detailed_analysis=detailed_analysis,
            execution_plan_analysis=execution_plan_analysis,
            explain_results=request.explain_results
        )
    
    def _generate_execution_plan_analysis(self, request: SQLAnalysisRequest) -> str:
        """生成执行计划分析.
        
        Args:
            request: SQL 分析请求
            
        Returns:
            执行计划分析文本
        """
        if not request.explain_results:
            return "无执行计划信息"
        
        # 获取数据库适配器
        from .database.adapters import DatabaseAdapterFactory
        from .tools import _detect_database_type
        
        database_type = _detect_database_type(request.explain_results)
        adapter = DatabaseAdapterFactory.create_adapter(database_type)
        
        lines = ["执行计划分析："]
        
        total_rows = 0
        has_full_scan = False
        has_index_usage = False
        
        for i, result in enumerate(request.explain_results, 1):
            # 使用适配器获取数据
            table_name = adapter.get_table_name(result)
            connection_type = adapter.get_connection_type(result)
            rows = adapter.get_scan_rows(result)
            index_info = adapter.get_index_info(result)
            extra_info = adapter.get_extra_info(result)
            cost_info = adapter.get_cost_info(result)
            
            lines.append(f"\n步骤 {i}: 访问表 {table_name}")
            
            # 分析连接类型
            if adapter.is_full_table_scan(result):
                lines.append("  ⚠️  全表扫描 - 性能风险高")
                has_full_scan = True
            elif adapter.is_index_scan(result):
                lines.append(f"  ✅ 使用索引 ({connection_type})")
                has_index_usage = True
            elif "Bitmap" in connection_type:
                lines.append(f"  🔄 位图扫描 ({connection_type})")
                has_index_usage = True
            
            # 分析行数信息
            if rows:
                lines.append(f"  📊 预估扫描行数: {rows:,}")
                total_rows += rows
            
            # 分析实际行数（PostgreSQL特有）
            actual_rows = cost_info.get("actual_rows")
            if actual_rows:
                lines.append(f"  📈 实际扫描行数: {actual_rows:,}")
            
            # 分析索引信息
            key = index_info.get("key")
            possible_keys = index_info.get("possible_keys")
            if key:
                lines.append(f"  🔑 使用索引: {key}")
            elif possible_keys:
                lines.append(f"  ⚠️  未使用可用索引: {possible_keys}")
            
            # 分析成本信息（PostgreSQL特有）
            startup_cost = cost_info.get("startup_cost")
            total_cost = cost_info.get("total_cost")
            if startup_cost:
                lines.append(f"  💰 启动成本: {startup_cost}")
            if total_cost:
                lines.append(f"  💰 总成本: {total_cost}")
            
            # 分析额外信息
            if extra_info:
                if "Using temporary" in extra_info:
                    lines.append("  ⚠️  使用临时表")
                if "Using filesort" in extra_info:
                    lines.append("  ⚠️  使用文件排序")
                if "Using index" in extra_info:
                    lines.append("  ✅ 使用覆盖索引")
                if "Parallel" in extra_info:
                    lines.append("  🔄 并行执行")
                if "Buffers" in extra_info:
                    lines.append("  📦 缓存使用")
        
        lines.append(f"\n总计预估扫描行数: {total_rows:,}")
        
        if has_full_scan:
            lines.append("⚠️  查询包含全表扫描，建议优化")
        elif has_index_usage:
            lines.append("✅ 查询有效利用了索引")
        
        return "\n".join(lines)
    
    def _generate_summary(
        self, 
        issues: List[PerformanceIssue], 
        suggestions: List[OptimizationSuggestion], 
        score: int
    ) -> str:
        """生成分析总结.
        
        Args:
            issues: 发现的问题
            suggestions: 优化建议
            score: 性能得分
            
        Returns:
            总结文本
        """
        if score >= 80:
            summary = f"查询性能良好（得分: {score}/100）。"
        elif score >= 60:
            summary = f"查询性能尚可（得分: {score}/100），有改进空间。"
        else:
            summary = f"查询性能较差（得分: {score}/100），急需优化。"
        
        if issues:
            critical_issues = [i for i in issues if i.severity == "critical"]
            high_issues = [i for i in issues if i.severity == "high"]
            
            if critical_issues:
                summary += f" 发现 {len(critical_issues)} 个严重问题。"
            elif high_issues:
                summary += f" 发现 {len(high_issues)} 个高优先级问题。"
            else:
                summary += f" 发现 {len(issues)} 个一般问题。"
        
        if suggestions:
            high_priority = [s for s in suggestions if s.priority == "high"]
            if high_priority:
                summary += f" 提供了 {len(high_priority)} 个高优先级优化建议。"
            else:
                summary += f" 提供了 {len(suggestions)} 个优化建议。"
        
        return summary


class OllamaAgent(BaseSQLAnalyzer):
    """基于 Ollama API 的 SQL 分析智能体.
    
    专门用于与本地部署的 Ollama 模型进行交互，分析 MySQL 慢 SQL 查询的性能问题
    并提供优化建议。
    """
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        name: str = "ollama_sql_analyzer",
        system_message: Optional[str] = None,
        timeout: float = 60.0,
        stream: bool = False,
        show_streaming_progress: bool = True,
    ) -> None:
        """初始化 Ollama SQL 分析智能体.
        
        Args:
            model: Ollama 模型名称
            base_url: Ollama API 基础 URL
            name: 智能体名称
            system_message: 系统提示消息，如果为空则使用默认消息
            timeout: 请求超时时间（秒）
            stream: 是否使用流式响应
            show_streaming_progress: 是否显示流式响应进度
        """
        if httpx is None:
            raise ImportError("需要安装 httpx 包: pip install httpx")
        
        super().__init__(model, name, system_message, timeout)
        
        self.base_url = base_url.rstrip('/')
        self.stream = stream
        self.show_streaming_progress = show_streaming_progress
        
        print(f"🦙 初始化 Ollama SQL 分析智能体: 使用的模型为 {self.model}")
        print(f"📡 API地址: {self.base_url}")
        if self.stream:
            progress_status = "启用" if self.show_streaming_progress else "禁用"
            print(f"⚡ 流式响应: 启用 (进度显示: {progress_status})")
    
    @classmethod
    def from_config(cls, config: OllamaConfig, name: str = "ollama_sql_analyzer") -> "OllamaAgent":
        """从配置创建智能体.
        
        Args:
            config: Ollama 配置
            name: 智能体名称
            
        Returns:
            配置好的 Ollama SQL 分析智能体
        """
        return cls(
            model=config.model,
            base_url=config.base_url,
            name=name,
            timeout=config.timeout,
            stream=config.stream,
            show_streaming_progress=config.show_streaming_progress,
        )
    
    def _get_default_system_message(self) -> str:
        """获取默认的系统提示消息.
        
        Returns:
            系统提示消息字符串
        """
        return get_default_system_message()
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试 Ollama API 连接.
        
        Returns:
            包含连接测试结果的字典
        """
        result = {
            "success": False,
            "error": None,
            "details": {},
        }
        
        try:
            logger.info(f"开始测试 Ollama API 连接: {self.base_url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "测试连接",
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    result["success"] = True
                    result["details"] = {
                        "model": self.model,
                        "base_url": self.base_url,
                        "response_preview": response_data.get("response", "")[:100],
                        "done": response_data.get("done", False),
                    }
                    logger.info("Ollama API 连接测试成功")
                else:
                    result["error"] = f"HTTP {response.status_code}: {response.text}"
                    result["details"] = {
                        "model": self.model,
                        "base_url": self.base_url,
                        "status_code": response.status_code,
                    }
                    logger.error(f"Ollama API 连接测试失败: {result['error']}")
                    
        except Exception as e:
            result["error"] = str(e)
            result["details"] = {
                "model": self.model,
                "base_url": self.base_url,
                "error_type": type(e).__name__,
            }
            logger.error(f"Ollama API 连接测试失败: {e}")
            
            # 提供具体的错误诊断
            if "Connection" in str(e) or "timeout" in str(e).lower():
                result["diagnosis"] = "Ollama 服务连接失败，请检查服务是否运行和网络连接"
            elif "404" in str(e) or "Not Found" in str(e):
                result["diagnosis"] = f"Ollama API 端点不存在，请检查 base_url: {self.base_url}"
            elif "TimeoutException" in str(e):
                result["diagnosis"] = "请求超时，请检查 Ollama 服务响应速度或增加超时时间"
            elif "model" in str(e).lower():
                result["diagnosis"] = f"模型 '{self.model}' 不存在，请检查模型名称或先下载模型"
            else:
                result["diagnosis"] = "未知错误，请检查 Ollama 服务状态和配置"
        
        return result
    
    async def analyze_sql(self, request: SQLAnalysisRequest) -> SQLAnalysisResponse:
        """使用 Ollama 分析 SQL 性能的核心方法.
        
        Args:
            request: SQL 分析请求
            
        Returns:
            SQL 分析响应
        """
        # 首先使用工具函数进行基础分析
        issues = detect_performance_issues(request)
        suggestions = generate_optimization_suggestions(request, issues)
        score = calculate_performance_score(request, issues)
        
        # 准备给 AI 的输入
        formatted_request = format_analysis_request(request)
        
        # 构造给 AI 的消息
        user_message = f"""请分析以下 SQL 查询的性能：

{formatted_request}

请提供：
1. 详细的性能分析
2. 执行计划解读
3. 发现的问题总结
4. 优化建议的详细说明

基础分析结果：
- 性能评分：{score}/100
- 发现问题数量：{len(issues)}
- 优化建议数量：{len(suggestions)}
"""
        
        # 构造完整的提示词
        full_prompt = f"{self.system_message}\n\n{user_message}"
        
        # 使用 Ollama API 进行深度分析
        detailed_analysis = ""
        try:
            logger.info(f"开始调用 Ollama API: model={self.model}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if self.stream:
                    # 流式响应处理
                    detailed_analysis = await self._handle_streaming_response(client, full_prompt)
                else:
                    # 非流式响应处理
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": full_prompt,
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        detailed_analysis = response_data.get("response", "无法获取详细分析")
                        logger.info("Ollama API 调用成功")
                    else:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Ollama API 调用失败: {error_msg}")
            
            # 提供更详细的错误信息和诊断
            diagnosis = ""
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                diagnosis = "Ollama 服务连接失败，请检查服务是否运行"
            elif "404" in error_msg:
                diagnosis = f"Ollama API 端点不存在，请检查 base_url: {self.base_url}"
            elif "TimeoutException" in error_msg:
                diagnosis = "请求超时，建议增加超时时间或检查模型响应速度"
            elif "model" in error_msg.lower():
                diagnosis = f"模型 '{self.model}' 不可用，请检查模型是否已下载"
            else:
                diagnosis = f"未知 Ollama API 错误: {type(e).__name__}"
            
            detailed_analysis = f"""AI 分析失败: {error_msg}

错误诊断: {diagnosis}

配置信息:
- 模型: {self.model}
- API 端点: {self.base_url}
- 超时设置: {self.timeout}秒

建议:
1. 确保 Ollama 服务正在运行 (ollama serve)
2. 检查模型是否已下载 (ollama pull {self.model})
3. 验证 API 端点地址是否正确
4. 检查网络连接和防火墙设置

使用基础分析结果。"""
        
        # 生成执行计划分析
        execution_plan_analysis = self._generate_execution_plan_analysis(request)
        
        # 生成总结
        summary = self._generate_summary(issues, suggestions, score)
        
        return SQLAnalysisResponse(
            summary=summary,
            performance_score=score,
            issues=issues,
            suggestions=suggestions,
            detailed_analysis=detailed_analysis,
            execution_plan_analysis=execution_plan_analysis,
            explain_results=request.explain_results
        )
    
    async def _handle_streaming_response(self, client: httpx.AsyncClient, prompt: str) -> str:
        """处理 Ollama 流式响应.
        
        Args:
            client: httpx 客户端
            prompt: 完整的提示词
            
        Returns:
            完整的响应文本
        """
        full_response = ""
        display = StreamingDisplay(show_progress=self.show_streaming_progress)
        
        try:
            display.start()
            
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    token = chunk["response"]
                                    full_response += token
                                    display.add_token(token)
                                
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    display.finish()
                    
                else:
                    raise Exception(f"HTTP {response.status_code}: {await response.aread()}")
                    
        except Exception as e:
            display.error_cleanup()
            logger.error(f"流式响应处理失败: {e}")
            raise
        
        return full_response
    
    def _generate_execution_plan_analysis(self, request: SQLAnalysisRequest) -> str:
        """生成执行计划分析.
        
        Args:
            request: SQL 分析请求
            
        Returns:
            执行计划分析文本
        """
        if not request.explain_results:
            return "无执行计划信息"
        
        # 获取数据库适配器
        from .database.adapters import DatabaseAdapterFactory
        from .tools import _detect_database_type
        
        database_type = _detect_database_type(request.explain_results)
        adapter = DatabaseAdapterFactory.create_adapter(database_type)
        
        lines = ["执行计划分析："]
        
        total_rows = 0
        has_full_scan = False
        has_index_usage = False
        
        for i, result in enumerate(request.explain_results, 1):
            # 使用适配器获取数据
            table_name = adapter.get_table_name(result)
            connection_type = adapter.get_connection_type(result)
            rows = adapter.get_scan_rows(result)
            index_info = adapter.get_index_info(result)
            extra_info = adapter.get_extra_info(result)
            cost_info = adapter.get_cost_info(result)
            
            lines.append(f"\n步骤 {i}: 访问表 {table_name}")
            
            # 分析连接类型
            if adapter.is_full_table_scan(result):
                lines.append("  ⚠️  全表扫描 - 性能风险高")
                has_full_scan = True
            elif adapter.is_index_scan(result):
                lines.append(f"  ✅ 使用索引 ({connection_type})")
                has_index_usage = True
            elif "Bitmap" in connection_type:
                lines.append(f"  🔄 位图扫描 ({connection_type})")
                has_index_usage = True
            
            # 分析行数信息
            if rows:
                lines.append(f"  📊 预估扫描行数: {rows:,}")
                total_rows += rows
            
            # 分析实际行数（PostgreSQL特有）
            actual_rows = cost_info.get("actual_rows")
            if actual_rows:
                lines.append(f"  📈 实际扫描行数: {actual_rows:,}")
            
            # 分析索引信息
            key = index_info.get("key")
            possible_keys = index_info.get("possible_keys")
            if key:
                lines.append(f"  🔑 使用索引: {key}")
            elif possible_keys:
                lines.append(f"  ⚠️  未使用可用索引: {possible_keys}")
            
            # 分析成本信息（PostgreSQL特有）
            startup_cost = cost_info.get("startup_cost")
            total_cost = cost_info.get("total_cost")
            if startup_cost:
                lines.append(f"  💰 启动成本: {startup_cost}")
            if total_cost:
                lines.append(f"  💰 总成本: {total_cost}")
            
            # 分析额外信息
            if extra_info:
                if "Using temporary" in extra_info:
                    lines.append("  ⚠️  使用临时表")
                if "Using filesort" in extra_info:
                    lines.append("  ⚠️  使用文件排序")
                if "Using index" in extra_info:
                    lines.append("  ✅ 使用覆盖索引")
                if "Parallel" in extra_info:
                    lines.append("  🔄 并行执行")
                if "Buffers" in extra_info:
                    lines.append("  📦 缓存使用")
        
        lines.append(f"\n总计预估扫描行数: {total_rows:,}")
        
        if has_full_scan:
            lines.append("⚠️  查询包含全表扫描，建议优化")
        elif has_index_usage:
            lines.append("✅ 查询有效利用了索引")
        
        return "\n".join(lines)
    
    def _generate_summary(
        self, 
        issues: List[PerformanceIssue], 
        suggestions: List[OptimizationSuggestion], 
        score: int
    ) -> str:
        """生成分析总结.
        
        Args:
            issues: 发现的问题
            suggestions: 优化建议
            score: 性能得分
            
        Returns:
            总结文本
        """
        if score >= 80:
            summary = f"查询性能良好（得分: {score}/100）。"
        elif score >= 60:
            summary = f"查询性能尚可（得分: {score}/100），有改进空间。"
        else:
            summary = f"查询性能较差（得分: {score}/100），急需优化。"
        
        if issues:
            critical_issues = [i for i in issues if i.severity == "critical"]
            high_issues = [i for i in issues if i.severity == "high"]
            
            if critical_issues:
                summary += f" 发现 {len(critical_issues)} 个严重问题。"
            elif high_issues:
                summary += f" 发现 {len(high_issues)} 个高优先级问题。"
            else:
                summary += f" 发现 {len(issues)} 个一般问题。"
        
        if suggestions:
            high_priority = [s for s in suggestions if s.priority == "high"]
            if high_priority:
                summary += f" 提供了 {len(high_priority)} 个高优先级优化建议。"
            else:
                summary += f" 提供了 {len(suggestions)} 个优化建议。"
        
        return summary


# 保持向后兼容的便利函数
def create_sql_analyzer_agent(
    api_key: str,
    model: str = "deepseek-chat",
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    stream: bool = False,
    show_streaming_progress: bool = True,
) -> SQLAnalyzerAgent:
    """创建 SQL 分析智能体的便利函数.
    
    Args:
        api_key: OpenAI API 密钥
        model: 模型名称
        base_url: API 基础 URL（可选）
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        stream: 是否使用流式响应
        show_streaming_progress: 是否显示流式响应进度
        
    Returns:
        配置好的 SQL 分析智能体
    """
    return SQLAnalyzerAgent(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        stream=stream,
        show_streaming_progress=show_streaming_progress,
    )


def create_ollama_agent(
    model: str,
    base_url: str = "http://localhost:11434",
    name: str = "ollama_sql_analyzer",
    timeout: float = 60.0,
    stream: bool = False,
    show_streaming_progress: bool = True,
) -> OllamaAgent:
    """创建 Ollama SQL 分析智能体的便利函数.
    
    Args:
        model: Ollama 模型名称
        base_url: Ollama API 基础 URL
        name: 智能体名称
        timeout: 请求超时时间（秒）
        stream: 是否使用流式响应
        show_streaming_progress: 是否显示流式响应进度
        
    Returns:
        配置好的 Ollama SQL 分析智能体
    """
    return OllamaAgent(
        model=model,
        base_url=base_url,
        name=name,
        timeout=timeout,
        stream=stream,
        show_streaming_progress=show_streaming_progress,
    )


# 新的工厂函数，支持从配置创建智能体
def create_agent_from_config(config_type: str, **kwargs) -> BaseSQLAnalyzer:
    """从配置创建智能体的工厂函数.
    
    Args:
        config_type: 配置类型，"openai" 或 "ollama"
        **kwargs: 传递给具体智能体的额外参数
        
    Returns:
        配置好的智能体实例
        
    Raises:
        ValueError: 如果配置类型不支持或配置无效
    """
    from .config import load_config_from_env
    
    config = load_config_from_env()
    
    if config_type == "openai":
        if config["openai"] is None:
            raise ValueError("OpenAI 配置无效，请检查环境变量")
        return SQLAnalyzerAgent.from_config(config["openai"], **kwargs)
    elif config_type == "ollama":
        if config["ollama"] is None:
            raise ValueError("Ollama 配置无效，请检查环境变量")
        return OllamaAgent.from_config(config["ollama"], **kwargs)
    else:
        raise ValueError(f"不支持的配置类型: {config_type}") 