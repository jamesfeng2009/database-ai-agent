"""多Agent系统管理器 - 负责启动、管理和协调所有Agent."""

import asyncio
import logging
import signal
from typing import Dict, List, Optional, Type

from ..communication.a2a_protocol import MessageBus, get_message_bus
from ..agents.base_agent import BaseAgent
from ..agents.coordinator_agent import CoordinatorAgent
from ..agents.nlp_agent import NLPAgent
from ..agents.sql_analysis_agent import SQLAnalysisAgent

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """多Agent系统管理器."""
    
    def __init__(self):
        """初始化多Agent系统."""
        self.message_bus: Optional[MessageBus] = None
        self.agents: Dict[str, BaseAgent] = {}
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # 注册Agent类型
        self.agent_classes = {
            "coordinator": CoordinatorAgent,
            "nlp": NLPAgent,
            "sql_analysis": SQLAnalysisAgent,
            # 可以继续添加其他Agent类型
        }
        
        # 默认Agent配置
        self.default_agents = [
            "coordinator",
            "nlp", 
            "sql_analysis"
        ]
    
    async def start(self, agents_to_start: Optional[List[str]] = None):
        """启动多Agent系统.
        
        Args:
            agents_to_start: 要启动的Agent列表，None表示启动默认Agent
        """
        if self.running:
            logger.warning("多Agent系统已经在运行")
            return
        
        try:
            logger.info("启动多Agent系统...")
            
            # 启动消息总线
            self.message_bus = get_message_bus()
            await self.message_bus.start()
            
            # 确定要启动的Agent
            agents_to_start = agents_to_start or self.default_agents
            
            # 启动Agent
            for agent_type in agents_to_start:
                await self._start_agent(agent_type)
            
            # 设置信号处理
            self._setup_signal_handlers()
            
            self.running = True
            logger.info(f"多Agent系统启动完成，运行中的Agent: {list(self.agents.keys())}")
            
            # 等待系统运行
            await self._run_system()
            
        except Exception as e:
            logger.error(f"多Agent系统启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止多Agent系统."""
        if not self.running:
            return
        
        logger.info("停止多Agent系统...")
        
        try:
            self.running = False
            self._shutdown_event.set()
            
            # 停止所有Agent
            stop_tasks = []
            for agent in self.agents.values():
                stop_tasks.append(agent.stop())
            
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # 停止消息总线
            if self.message_bus:
                await self.message_bus.stop()
            
            self.agents.clear()
            logger.info("多Agent系统已停止")
            
        except Exception as e:
            logger.error(f"停止多Agent系统时发生错误: {e}")
    
    async def add_agent(self, agent_type: str, agent_instance: Optional[BaseAgent] = None) -> bool:
        """添加Agent到系统.
        
        Args:
            agent_type: Agent类型
            agent_instance: Agent实例，None表示创建新实例
            
        Returns:
            是否成功添加
        """
        try:
            if agent_type in self.agents:
                logger.warning(f"Agent {agent_type} 已存在")
                return False
            
            # 创建或使用提供的Agent实例
            if agent_instance:
                agent = agent_instance
            else:
                agent = await self._create_agent(agent_type)
            
            if not agent:
                logger.error(f"无法创建Agent: {agent_type}")
                return False
            
            # 启动Agent
            await agent.start()
            self.agents[agent.agent_id] = agent
            
            logger.info(f"Agent {agent_type} 已添加到系统")
            return True
            
        except Exception as e:
            logger.error(f"添加Agent失败: {agent_type}, 错误: {e}")
            return False
    
    async def remove_agent(self, agent_id: str) -> bool:
        """从系统中移除Agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功移除
        """
        try:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.warning(f"Agent {agent_id} 不存在")
                return False
            
            # 停止Agent
            await agent.stop()
            del self.agents[agent_id]
            
            logger.info(f"Agent {agent_id} 已从系统中移除")
            return True
            
        except Exception as e:
            logger.error(f"移除Agent失败: {agent_id}, 错误: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取Agent实例.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent实例
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, any]]:
        """列出所有Agent.
        
        Returns:
            Agent信息列表
        """
        agent_list = []
        for agent in self.agents.values():
            agent_list.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "capabilities": agent.capabilities,
                "is_running": agent.is_running
            })
        
        return agent_list
    
    async def get_system_status(self) -> Dict[str, any]:
        """获取系统状态.
        
        Returns:
            系统状态信息
        """
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_stats()
        
        return {
            "system_running": self.running,
            "total_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.is_running),
            "message_bus_running": self.message_bus is not None,
            "agents": agent_stats
        }
    
    async def send_system_broadcast(self, action: str, payload: Dict[str, any]):
        """发送系统广播消息.
        
        Args:
            action: 操作名称
            payload: 消息载荷
        """
        if self.message_bus:
            await self.message_bus.broadcast_message("system", action, payload)
    
    async def _start_agent(self, agent_type: str) -> bool:
        """启动指定类型的Agent.
        
        Args:
            agent_type: Agent类型
            
        Returns:
            是否成功启动
        """
        try:
            agent = await self._create_agent(agent_type)
            if not agent:
                return False
            
            await agent.start()
            self.agents[agent.agent_id] = agent
            
            logger.info(f"Agent {agent_type} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动Agent失败: {agent_type}, 错误: {e}")
            return False
    
    async def _create_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """创建Agent实例.
        
        Args:
            agent_type: Agent类型
            
        Returns:
            Agent实例
        """
        agent_class = self.agent_classes.get(agent_type)
        if not agent_class:
            logger.error(f"未知的Agent类型: {agent_type}")
            return None
        
        try:
            return agent_class()
        except Exception as e:
            logger.error(f"创建Agent实例失败: {agent_type}, 错误: {e}")
            return None
    
    def _setup_signal_handlers(self):
        """设置信号处理器."""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，开始关闭系统...")
            asyncio.create_task(self.stop())
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _run_system(self):
        """运行系统主循环."""
        try:
            # 等待关闭信号
            await self._shutdown_event.wait()
        except Exception as e:
            logger.error(f"系统运行时发生错误: {e}")
        finally:
            await self.stop()
    
    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]):
        """注册新的Agent类型.
        
        Args:
            agent_type: Agent类型名称
            agent_class: Agent类
        """
        self.agent_classes[agent_type] = agent_class
        logger.info(f"注册Agent类型: {agent_type}")


# 全局多Agent系统实例
_global_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """获取全局多Agent系统实例.
    
    Returns:
        多Agent系统实例
    """
    global _global_multi_agent_system
    if _global_multi_agent_system is None:
        _global_multi_agent_system = MultiAgentSystem()
    return _global_multi_agent_system


async def start_multi_agent_system(agents_to_start: Optional[List[str]] = None):
    """启动多Agent系统的便捷函数.
    
    Args:
        agents_to_start: 要启动的Agent列表
    """
    system = get_multi_agent_system()
    await system.start(agents_to_start)


async def stop_multi_agent_system():
    """停止多Agent系统的便捷函数."""
    system = get_multi_agent_system()
    await system.stop()


# 命令行启动脚本
async def main():
    """主函数，用于命令行启动."""
    import argparse
    
    parser = argparse.ArgumentParser(description="多Agent SQL分析系统")
    parser.add_argument(
        "--agents", 
        nargs="+", 
        default=None,
        help="要启动的Agent列表"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # 启动系统
        await start_multi_agent_system(args.agents)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭系统...")
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
    finally:
        await stop_multi_agent_system()


if __name__ == "__main__":
    asyncio.run(main())