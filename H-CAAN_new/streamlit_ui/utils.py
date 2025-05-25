# streamlit/utils.py
import streamlit as st
import os
import importlib.util

def rerun():
    """安全地重新运行Streamlit应用，兼容不同版本。"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.warning("Could not rerun the app. Please refresh the page manually.")

def load_page_module(page_name, file_path=None):
    """
    从文件路径安全地加载页面模块。
    
    Args:
        page_name: 页面的名称
        file_path: 页面文件的路径，如果为None则使用默认路径
        
    Returns:
        加载的模块，如果加载失败则返回None
    """
    if file_path is None:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "pages", f"{page_name}.py")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            st.error(f"Page file does not exist: {file_path}")
            return None
        
        # 从文件路径导入模块
        spec = importlib.util.spec_from_file_location(page_name, file_path)
        if spec is None:
            st.error(f"Could not create module spec from: {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        st.error(f"Error loading page module {page_name}: {str(e)}")
        return None

def get_page_function(page_name):
    """
    获取页面函数。
    
    Args:
        page_name: 页面的名称
        
    Returns:
        页面函数，如果获取失败则返回None
    """
    module = load_page_module(page_name)
    if module is None:
        return None
        
    # 尝试获取页面函数
    page_func = getattr(module, page_name, None)
    if page_func is None:
        st.error(f"No function named '{page_name}' in module")
    
    return page_func
