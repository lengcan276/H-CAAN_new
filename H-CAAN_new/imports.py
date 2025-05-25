# File: imports.py
"""提供跨模块导入帮助的工具"""

import os
import sys

def add_project_root_to_path():
    """将项目根目录添加到Python路径"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def import_agent(agent_name):
    """导入指定的代理"""
    add_project_root_to_path()
    try:
        module = __import__(f"agents.{agent_name}_agent", fromlist=["*"])
        return module
    except ImportError as e:
        print(f"Error importing agent {agent_name}: {e}")
        return None