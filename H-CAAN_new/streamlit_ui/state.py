# streamlit_ui/state.py
"""
状态管理模块 - 管理应用的全局状态
"""
import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# 状态文件保存路径
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_state.json")

# 初始状态
DEFAULT_STATE = {
    "current_page": "home",
    "data_processed": False,
    "model_configured": False,
    "model_trained": False,
    "results_analyzed": False,
    "dataset_info": None,
    "model_config": None,
    "training_results": None,
    "evaluation_results": None,
    "session_id": None,
    "last_updated": None
}

class StateManager:
    """状态管理器类，提供更高级的状态管理功能"""
    
    def __init__(self):
        self.state_file = STATE_FILE
        self.default_state = DEFAULT_STATE
    
    def initialize(self):
        """初始化应用状态"""
        initialize_state()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置状态值"""
        st.session_state[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """批量更新状态"""
        for key, value in updates.items():
            st.session_state[key] = value
    
    def save(self):
        """保存状态"""
        save_state()
    
    def load(self):
        """加载状态"""
        load_state()
    
    def reset(self):
        """重置状态"""
        reset_state()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return get_state_summary()
    
    def is_step_completed(self, step: str) -> bool:
        """检查某个步骤是否完成"""
        step_mapping = {
            'data_prepared': 'data_processed',
            'model_configured': 'model_configured',
            'training_completed': 'model_trained',
            'results_analyzed': 'results_analyzed'
        }
        return st.session_state.get(step_mapping.get(step, step), False)
    
    def update_workflow_status(self, step: str, completed: bool = True):
        """更新工作流程状态"""
        step_mapping = {
            'data_prepared': 'data_processed',
            'model_configured': 'model_configured',
            'training_completed': 'model_trained',
            'results_analyzed': 'results_analyzed'
        }
        if step in step_mapping:
            st.session_state[step_mapping[step]] = completed
    
    def get_workflow_status(self) -> Dict[str, bool]:
        """获取工作流程状态"""
        return {
            'data_prepared': st.session_state.get('data_processed', False),
            'model_configured': st.session_state.get('model_configured', False),
            'training_completed': st.session_state.get('model_trained', False),
            'results_analyzed': st.session_state.get('results_analyzed', False)
        }

# 创建全局状态管理器实例
state_manager = StateManager()

def initialize_state():
    """初始化应用状态"""
    # 为新会话生成唯一ID
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 初始化默认状态
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 尝试加载保存的状态
    try:
        load_state()
    except Exception as e:
        st.warning(f"无法加载保存的状态: {e}")

def save_state():
    """保存当前状态到文件"""
    state_to_save = {}
    for key in DEFAULT_STATE.keys():
        if key in st.session_state:
            # 只保存可序列化的值
            try:
                json.dumps({key: st.session_state[key]})
                state_to_save[key] = st.session_state[key]
            except (TypeError, OverflowError):
                # 跳过不可序列化的值
                pass
    
    # 添加时间戳
    state_to_save["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state_to_save, f, indent=2)
    except Exception as e:
        st.warning(f"无法保存状态: {e}")

def load_state():
    """从文件加载状态"""
    if not os.path.exists(STATE_FILE):
        return
    
    try:
        with open(STATE_FILE, "r") as f:
            saved_state = json.load(f)
        
        # 更新会话状态
        for key, value in saved_state.items():
            if key in DEFAULT_STATE:
                st.session_state[key] = value
    except Exception as e:
        st.warning(f"加载状态时出错: {e}")

def reset_state():
    """重置状态到默认值"""
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value
    
    # 生成新会话ID
    st.session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 保存重置后的状态
    save_state()

def get_state_summary() -> Dict[str, Any]:
    """获取状态摘要，用于显示"""
    return {
        "workflow_status": {
            "data_processed": st.session_state.get("data_processed", False),
            "model_configured": st.session_state.get("model_configured", False),
            "model_trained": st.session_state.get("model_trained", False),
            "results_analyzed": st.session_state.get("results_analyzed", False)
        },
        "dataset_info": {
            "name": st.session_state.get("dataset_info", {}).get("name", "未选择") if isinstance(st.session_state.get("dataset_info"), dict) else "未选择",
            "size": st.session_state.get("dataset_info", {}).get("size", 0) if isinstance(st.session_state.get("dataset_info"), dict) else 0,
        },
        "current_page": st.session_state.get("current_page", "home")
    }

# 导出
__all__ = [
    'state_manager',
    'StateManager',
    'initialize_state',
    'save_state',
    'load_state',
    'reset_state',
    'get_state_summary'
]