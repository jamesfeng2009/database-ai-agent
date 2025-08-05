#!/usr/bin/env python3
"""SQL 分析器依赖安装脚本

这个脚本帮助用户快速安装 SQL 分析器所需的所有依赖包。
"""

import subprocess
import sys
import os
from typing import List, Tuple


def run_command(command: List[str], description: str) -> bool:
    """运行命令并返回是否成功.
    
    Args:
        command: 要执行的命令列表
        description: 命令描述
        
    Returns:
        命令是否成功执行
    """
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        if e.stdout:
            print(f"   输出: {e.stdout}")
        if e.stderr:
            print(f"   错误: {e.stderr}")
        return False


def check_python_version() -> bool:
    """检查 Python 版本是否满足要求.
    
    Returns:
        Python 版本是否满足要求
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 版本过低: {version.major}.{version.minor}")
        print("   需要 Python 3.9 或更高版本")
        return False
    
    print(f"✅ Python 版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies() -> bool:
    """安装项目依赖.
    
    Returns:
        安装是否成功
    """
    print("\n📦 开始安装依赖包...")
    
    # 基础依赖
    base_deps = [
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "sqlparse>=0.4.0",
        "httpx>=0.25.0",
        "openai>=1.0.0",
    ]
    
    # 数据库驱动
    db_deps = [
        "aiomysql>=0.2.0",
        "pymysql>=1.1.0",
        "asyncpg>=0.29.0",
    ]
    
    # AI 相关依赖
    ai_deps = [
        "autogen-agentchat>=0.4.0",
        "autogen-ext[openai]>=0.4.0",
    ]
    
    # 开发依赖（可选）
    dev_deps = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
    ]
    
    all_deps = base_deps + db_deps + ai_deps
    
    # 安装基础依赖
    if not run_command([sys.executable, "-m", "pip", "install"] + all_deps, "安装基础依赖"):
        return False
    
    # 询问是否安装开发依赖
    print("\n🤔 是否安装开发依赖（用于测试和代码质量检查）？")
    print("   输入 'y' 安装，输入其他键跳过")
    
    user_input = input("   请选择 (y/N): ").strip().lower()
    if user_input == 'y':
        if not run_command([sys.executable, "-m", "pip", "install"] + dev_deps, "安装开发依赖"):
            print("⚠️  开发依赖安装失败，但不影响基本功能")
    
    return True


def create_virtual_environment() -> bool:
    """创建虚拟环境.
    
    Returns:
        虚拟环境创建是否成功
    """
    print("\n🌍 检查虚拟环境...")
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 已在虚拟环境中")
        return True
    
    print("⚠️  未检测到虚拟环境")
    print("   建议在虚拟环境中运行此工具")
    print("   创建虚拟环境命令:")
    print("   python -m venv sql_analyzer_env")
    print("   source sql_analyzer_env/bin/activate  # Linux/Mac")
    print("   sql_analyzer_env\\Scripts\\activate     # Windows")
    
    user_input = input("   是否继续安装？(y/N): ").strip().lower()
    return user_input == 'y'


def verify_installation() -> bool:
    """验证安装是否成功.
    
    Returns:
        验证是否成功
    """
    print("\n🔍 验证安装...")
    
    # 测试导入关键模块
    test_imports = [
        ("pydantic", "数据验证库"),
        ("dotenv", "环境变量管理"),
        ("aiomysql", "MySQL 异步驱动"),
        ("asyncpg", "PostgreSQL 异步驱动"),
        ("httpx", "HTTP 客户端"),
        ("openai", "OpenAI API 客户端"),
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description} ({module}) 导入成功")
        except ImportError as e:
            print(f"❌ {description} ({module}) 导入失败: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  以下模块导入失败: {', '.join(failed_imports)}")
        print("   请检查安装是否完整")
        return False
    
    print("✅ 所有关键模块导入成功")
    return True


def show_next_steps() -> None:
    """显示后续步骤."""
    print("\n🎉 安装完成！")
    print("\n📋 后续步骤:")
    print("1. 配置环境变量:")
    print("   cp config/env.example .env")
    print("   # 编辑 .env 文件，设置数据库连接信息")
    print("")
    print("2. 运行 SQL 分析器:")
    print("   python app.py")
    print("")
    print("3. 支持的数据库:")
    print("   🐬 MySQL - 设置 MYSQL_* 环境变量")
    print("   🐘 PostgreSQL - 设置 POSTGRESQL_* 环境变量")
    print("")
    print("4. AI 分析模式:")
    print("   🦙 Ollama 本地模式 - 设置 OLLAMA_* 环境变量")
    print("   🤖 OpenAI 云端模式 - 设置 OPENAI_* 环境变量")
    print("")
    print("📖 更多信息请查看 README.md")


def main() -> None:
    """主函数."""
    print("SQL 分析器依赖安装脚本")
    print("=" * 50)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查虚拟环境
    if not create_virtual_environment():
        print("安装已取消")
        sys.exit(0)
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败")
        sys.exit(1)
    
    # 验证安装
    if not verify_installation():
        print("❌ 安装验证失败")
        sys.exit(1)
    
    # 显示后续步骤
    show_next_steps()


if __name__ == "__main__":
    main() 