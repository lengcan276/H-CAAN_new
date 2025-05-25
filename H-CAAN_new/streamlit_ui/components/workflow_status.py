"""
工作流程状态组件 - 显示当前工作流程的完成状态
"""
import streamlit as st
from streamlit_ui.workflow import get_workflow_progress, PAGE_NAMES

def render_workflow_status():
    """渲染工作流程状态组件"""
    # 计算工作流程进度
    progress = get_workflow_progress()
    
    # 创建进度条
    st.progress(progress)
    
    # 显示当前页面
    current_page = st.session_state.get("current_page", "home")
    current_page_name = PAGE_NAMES.get(current_page, "未知页面")
    
    # 显示当前状态
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.caption(f"当前页面: {current_page_name}")
    
    with col2:
        st.caption(f"完成进度: {int(progress * 100)}%")
    
    with col3:
        if st.session_state.get("session_id"):
            st.caption(f"会话ID: {st.session_state['session_id'][:8]}...")
    
    # 添加分隔线
    st.divider()