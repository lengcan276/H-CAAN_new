# File: app_fixed.py
import os
import sys
import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="H-CAAN: Hierarchical Cross-modal Adaptive Attention Network",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 使用绝对路径修复导入问题
project_root = os.path.dirname(os.path.abspath(__file__))
pages_dir = os.path.join(project_root, "pages")
sys.path.insert(0, project_root)

# 直接导入页面模块（不从函数导入）
import pages.data_page
import pages.model_page
import pages.training_page
import pages.results_page

# 检查模块并获取需要的函数
data_page_func = getattr(pages.data_page, "data_page", None)
model_page_func = getattr(pages.model_page, "model_page", None)
training_page_func = getattr(pages.training_page, "training_page", None)
results_page_func = getattr(pages.results_page, "results_page", None)

# 如果函数不可用，创建占位函数
if not data_page_func:
    def data_page_func():
        st.error("Data page functionality not available")
        st.write("Please check if data_page.py contains the data_page function")

if not model_page_func:
    def model_page_func():
        st.error("Model page functionality not available")
        st.write("Please check if model_page.py contains the model_page function")

if not training_page_func:
    def training_page_func():
        st.error("Training page functionality not available")
        st.write("Please check if training_page.py contains the training_page function")
        
if not results_page_func:
    def results_page_func():
        st.error("Results page functionality not available")
        st.write("Please check if results_page.py contains the results_page function")

# 主应用函数
def main():
    # 添加导航栏
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Preparation", "Model Configuration", 
                                     "Training & Evaluation", "Results & Visualization"])
    
    # 设置默认页面
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'data_page'
    
    # 处理页面导航
    if 'current_page' in st.session_state:
        if st.session_state['current_page'] == 'model_page':
            page = "Model Configuration"
        elif st.session_state['current_page'] == 'training_page':
            page = "Training & Evaluation"
        elif st.session_state['current_page'] == 'results_page':
            page = "Results & Visualization"
    
    # 显示选定的页面
    st.title("H-CAAN: Hierarchical Cross-modal Adaptive Attention Network")
    
    try:
        if page == "Data Preparation":
            st.session_state['current_page'] = 'data_page'
            data_page_func()
        elif page == "Model Configuration":
            st.session_state['current_page'] = 'model_page'
            model_page_func()
        elif page == "Training & Evaluation":
            st.session_state['current_page'] = 'training_page'
            training_page_func()
        elif page == "Results & Visualization":
            st.session_state['current_page'] = 'results_page'
            results_page_func()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.code("""
        Traceback:
        {}
        """.format(sys.exc_info()))

if __name__ == "__main__":
    main()