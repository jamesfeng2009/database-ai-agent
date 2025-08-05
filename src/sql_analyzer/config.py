"""配置管理模块."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OpenAIConfig:
    """OpenAI API 配置."""
    
    api_key: str
    model: str = "deepseek-chat"
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    stream: bool = False
    show_streaming_progress: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "OPENAI") -> "OpenAIConfig":
        """从环境变量创建配置.
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            OpenAI 配置实例
            
        Raises:
            ValueError: 如果必需的环境变量不存在
        """
        api_key = os.getenv(f"{prefix}_API_KEY")
        if not api_key:
            raise ValueError(f"环境变量 {prefix}_API_KEY 不能为空")
        
        return cls(
            api_key=api_key,
            model=os.getenv(f"{prefix}_MODEL", cls.model),
            base_url=os.getenv(f"{prefix}_BASE_URL"),
            timeout=float(os.getenv(f"{prefix}_TIMEOUT", str(cls.timeout))),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", str(cls.max_retries))),
            stream=os.getenv(f"{prefix}_STREAM", "false").lower() == "true",
            show_streaming_progress=os.getenv(f"{prefix}_SHOW_STREAMING_PROGRESS", "true").lower() == "true",
        )


@dataclass
class OllamaConfig:
    """Ollama API 配置."""
    
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0
    stream: bool = False
    show_streaming_progress: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "OLLAMA") -> "OllamaConfig":
        """从环境变量创建配置.
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            Ollama 配置实例
            
        Raises:
            ValueError: 如果必需的环境变量不存在
        """
        model = os.getenv(f"{prefix}_MODEL")
        if not model:
            raise ValueError(f"环境变量 {prefix}_MODEL 不能为空")
        
        return cls(
            model=model,
            base_url=os.getenv(f"{prefix}_BASE_URL", cls.base_url),
            timeout=float(os.getenv(f"{prefix}_TIMEOUT", str(cls.timeout))),
            stream=os.getenv(f"{prefix}_STREAM", "false").lower() == "true",
            show_streaming_progress=os.getenv(f"{prefix}_SHOW_STREAMING_PROGRESS", "true").lower() == "true",
        )


@dataclass
class SQLAnalyzerConfig:
    """SQL 分析器通用配置."""
    
    log_level: str = "INFO"
    output_format: str = "json"  # json, text, markdown
    enable_detailed_analysis: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "SQL_ANALYZER") -> "SQLAnalyzerConfig":
        """从环境变量创建配置.
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            SQL 分析器配置实例
        """
        return cls(
            log_level=os.getenv(f"{prefix}_LOG_LEVEL", cls.log_level),
            output_format=os.getenv(f"{prefix}_OUTPUT_FORMAT", cls.output_format),
            enable_detailed_analysis=os.getenv(
                f"{prefix}_ENABLE_DETAILED_ANALYSIS", 
                "true"
            ).lower() == "true",
        )


def load_config_from_env() -> dict:
    """从环境变量加载所有配置.
    
    Returns:
        包含所有配置的字典
    """
    config = {
        "sql_analyzer": SQLAnalyzerConfig.from_env()
    }
    
    # 尝试加载 OpenAI 配置
    try:
        config["openai"] = OpenAIConfig.from_env()
    except ValueError:
        config["openai"] = None
    
    # 尝试加载 Ollama 配置  
    try:
        config["ollama"] = OllamaConfig.from_env()
    except ValueError:
        config["ollama"] = None
    
    return config


def get_sample_env_config() -> str:
    """获取示例环境变量配置.
    
    Returns:
        示例环境变量配置字符串
    """
    return """# SQL 分析器环境变量配置示例

# === OpenAI API 配置 ===
# OpenAI API 密钥（必填）
OPENAI_API_KEY=your_openai_api_key_here

# 模型名称（可选，默认: deepseek-chat）
OPENAI_MODEL=deepseek-chat

# API 基础 URL（可选，用于自定义端点）
OPENAI_BASE_URL=https://api.deepseek.com

# 请求超时时间，单位秒（可选，默认: 60.0）
OPENAI_TIMEOUT=60.0

# 最大重试次数（可选，默认: 3）
OPENAI_MAX_RETRIES=3

# 是否启用流式响应（可选，默认: false）
OPENAI_STREAM=false

# 是否显示流式响应进度信息（可选，默认: true）
OPENAI_SHOW_STREAMING_PROGRESS=true

# === Ollama API 配置 ===
# Ollama 模型名称（必填）
OLLAMA_MODEL=llama3.2

# Ollama API 基础 URL（可选，默认: http://localhost:11434）
OLLAMA_BASE_URL=http://localhost:11434

# 请求超时时间，单位秒（可选，默认: 60.0）
OLLAMA_TIMEOUT=60.0

# 是否启用流式响应（可选，默认: false）
OLLAMA_STREAM=false

# 是否显示流式响应进度信息（可选，默认: true）
OLLAMA_SHOW_STREAMING_PROGRESS=true

# === SQL 分析器通用配置 ===
# 日志级别（可选，默认: INFO）
SQL_ANALYZER_LOG_LEVEL=INFO

# 输出格式（可选，默认: json）可选值: json, text, markdown
SQL_ANALYZER_OUTPUT_FORMAT=json

# 是否启用详细分析（可选，默认: true）
SQL_ANALYZER_ENABLE_DETAILED_ANALYSIS=true
""" 