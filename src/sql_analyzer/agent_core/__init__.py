"""SQL分析器Agent核心模块 - 重构后的模块化架构."""

# Agent实现
from .agents import (
    BaseAgent,
    CoordinatorAgent,
    KnowledgeAgent,
    MemoryAgent,
    SQLAnalysisAgent,
    NLPAgent
)

# 通信协议
from .communication import (
    A2AMessage,
    A2AMessageType,
    A2AProtocol,
    EventSystem
)

# 服务组件
from .services import (
    AutoOptimizer,
    SafetyValidator,
    RollbackManager,
    SQLIntegration,
    ChunkingManager,
    TextChunk
)

# 管理组件
from .management import (
    MultiAgentSystem,
    TaskOrchestrator,
    ContextManager,
    ConversationManager,
    SessionManager
)

# 数据模型
from .models import *

__all__ = [
    # Agents
    "BaseAgent",
    "CoordinatorAgent",
    "KnowledgeAgent", 
    "MemoryAgent",
    "SQLAnalysisAgent",
    "NLPAgent",
    
    # Communication
    "A2AMessage",
    "A2AMessageType",
    "A2AProtocol",
    "EventSystem",
    
    # Services
    "AutoOptimizer",
    "SafetyValidator",
    "RollbackManager", 
    "SQLIntegration",
    "ChunkingManager",
    "TextChunk",
    
    # Management
    "MultiAgentSystem",
    "TaskOrchestrator",
    "ContextManager",
    "ConversationManager",
    "SessionManager"
]