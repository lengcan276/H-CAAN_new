# streamlit_ui/app.py
import streamlit as st
import os
import sys

# 页面配置
st.set_page_config(
    page_title="H-CAAN: Hierarchical Cross-modal Adaptive Attention Network",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入必要的模块
from streamlit_ui.state import initialize_state, save_state
from streamlit_ui.styles import apply_custom_styles
from streamlit_ui.workflow import (
    WorkflowStep, 
    get_current_step, 
    set_current_step,
    initialize_workflow_state,
    mark_step_completed
)
from streamlit_ui.components.sidebar import render_sidebar
from streamlit_ui.components.header import render_header, render_page_header
from streamlit_ui.components.workflow_status import render_workflow_status

# 初始化状态
initialize_state()
initialize_workflow_state()

# 应用自定义样式
apply_custom_styles()

# 自定义侧边栏样式
st.markdown("""
<style>
    /* 自定义侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    
    /* 调整页面链接样式 */
    .st-emotion-cache-1rtdyuf {
        font-size: 14px;
    }
    
    /* 隐藏页面图标 */
    .st-emotion-cache-1avcm0n {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# 渲染侧边栏
render_sidebar()

# 获取当前步骤
current_step = get_current_step()

# 主内容区域
main_container = st.container()

with main_container:
    # 根据当前步骤渲染相应页面
    if current_step == WorkflowStep.HOME:
        # 渲染页面头部
        render_header()
        
        # 渲染主页面内容
        st.title("H-CAAN: Hierarchical Cross-modal Adaptive Attention Network")
        st.markdown("---")
        
        # 显示工作流程状态
        render_workflow_status()
        
        # 欢迎信息
        st.markdown("""
        ## Welcome to H-CAAN System
        
        H-CAAN is a multimodal deep learning model designed for drug property prediction. 
        It integrates chemical language and molecular graph information through hierarchical 
        cross-modal adaptive attention mechanisms to improve prediction accuracy.
        
        ### Key Features:
        - 🔬 **Multimodal Fusion**: Integrates SMILES, ECFP, molecular graphs, and MFBERT features
        - 🧠 **Adaptive Attention**: Hierarchical cross-modal attention mechanism
        - 📊 **Comprehensive Analysis**: Automated evaluation and visualization
        - 📝 **Paper Generation**: Automated research paper writing
        
        ### Getting Started
        
        Please follow these steps to use the system:
        
        1. **Data Preparation** - Upload your molecular dataset
        2. **Model Configuration** - Select appropriate model parameters
        3. **Model Training** - Train your model
        4. **Results Analysis** - View and analyze prediction results
        5. **Paper Generation** - Automatically generate research paper
        
        Click the button below or use the navigation bar to start!
        """)
        
        # Mark home as completed
        mark_step_completed(WorkflowStep.HOME)
        
        # 快速开始按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Data Preparation", type="primary", use_container_width=True):
                # 使用workflow系统导航到下一步
                set_current_step(WorkflowStep.DATA_PREPARATION)
                st.rerun()
    
    elif current_step == WorkflowStep.DATA_PREPARATION:
        # 渲染页面头部
        render_page_header(current_step)
        from streamlit_ui.pages.data import render_data_page
        render_data_page()
    
    elif current_step == WorkflowStep.MODEL_CONFIGURATION:
        render_page_header(current_step)
        from streamlit_ui.pages.model import render_model_page
        render_model_page()
    
    elif current_step == WorkflowStep.TRAINING:
        render_page_header(current_step)
        from streamlit_ui.pages.train import render_train_page
        render_train_page()
    
    elif current_step == WorkflowStep.RESULTS:
        render_page_header(current_step)
        from streamlit_ui.pages.results import render_results_page
        render_results_page()
    
    elif current_step == WorkflowStep.PAPER:
        render_page_header(current_step)
        from streamlit_ui.pages.paper import render_paper_page
        render_paper_page()
    
    else:
        st.error(f"Unknown step: {current_step}")

# 保存状态
save_state()