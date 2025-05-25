"""
模态对话框组件 - 显示模态框和弹出窗口
"""
import streamlit as st
import time
import uuid

def show_modal(title, content, key=None, has_close_button=True):
    """
    显示模态对话框
    
    Args:
        title (str): 对话框标题
        content (callable): 对话框内容渲染函数，该函数应接受一个key参数
        key (str, optional): 对话框唯一标识符，默认生成随机ID
        has_close_button (bool, optional): 是否显示关闭按钮，默认为True
    
    Returns:
        bool: 对话框是否被关闭
    """
    # 生成唯一key
    if key is None:
        key = f"modal_{uuid.uuid4().hex[:8]}"
    
    # 初始化状态
    if f"{key}_is_open" not in st.session_state:
        st.session_state[f"{key}_is_open"] = True
    
    # 检查对话框是否应该显示
    if not st.session_state[f"{key}_is_open"]:
        return False
    
    # 创建对话框遮罩
    modal_container = st.container()
    
    with modal_container:
        # 对话框样式
        modal_style = """
        <style>
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .modal-content {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .modal-close {
            cursor: pointer;
            font-size: 20px;
            color: #aaa;
        }
        .modal-close:hover {
            color: black;
        }
        </style>
        """
        
        # 渲染模态框HTML
        modal_html = f"""
        <div class="modal-backdrop" id="modal-{key}">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>{title}</h2>
                    {"<span class='modal-close' onclick='closeModal()'>×</span>" if has_close_button else ""}
                </div>
                <div class="modal-body" id="modal-body-{key}">
                    <!-- 内容将由Streamlit注入 -->
                </div>
            </div>
        </div>
        
        <script>
        function closeModal() {{
            // 通过Streamlit API关闭模态框
            if (window.parent.streamlit) {{
                window.parent.streamlit.setComponentValue({{
                    modal_key: "{key}",
                    action: "close"
                }});
            }}
            
            // 隐藏模态框
            document.getElementById("modal-{key}").style.display = "none";
        }}
        </script>
        """
        
        # 显示模态框HTML
        st.markdown(modal_style + modal_html, unsafe_allow_html=True)
        
        # 创建一个占位控件，用于接收模态框的关闭事件
        modal_callback = st.empty()
        
        # 添加监听器
        modal_result = modal_callback.text_input(
            "Modal Callback", 
            value="", 
            key=f"{key}_callback",
            label_visibility="collapsed"
        )
        
        # 处理回调
        if modal_result:
            try:
                result = json.loads(modal_result)
                if result.get("modal_key") == key and result.get("action") == "close":
                    st.session_state[f"{key}_is_open"] = False
                    st.rerun()
                    return False
            except:
                pass
        
        # 渲染内容
        content(key)
    
    return True

def confirmation_dialog(title, message, confirm_text="确认", cancel_text="取消", key=None):
    """
    显示确认对话框
    
    Args:
        title (str): 对话框标题
        message (str): 对话框消息
        confirm_text (str, optional): 确认按钮文本
        cancel_text (str, optional): 取消按钮文本
        key (str, optional): 对话框唯一标识符
    
    Returns:
        bool: 用户是否确认
    """
    # 生成唯一key
    if key is None:
        key = f"confirm_{uuid.uuid4().hex[:8]}"
    
    # 初始化状态
    if f"{key}_result" not in st.session_state:
        st.session_state[f"{key}_result"] = None
    
    # 定义内容渲染函数
    def render_content(modal_key):
        st.markdown(message)
        
        cols = st.columns(2)
        
        with cols[0]:
            if st.button(cancel_text, key=f"{modal_key}_cancel"):
                st.session_state[f"{key}_result"] = False
                st.session_state[f"{key}_is_open"] = False
                st.rerun()
        
        with cols[1]:
            if st.button(confirm_text, key=f"{modal_key}_confirm"):
                st.session_state[f"{key}_result"] = True
                st.session_state[f"{key}_is_open"] = False
                st.rerun()
    
    # 显示对话框
    show_modal(title, render_content, key=key)
    
    # 返回结果
    return st.session_state[f"{key}_result"]

def alert_dialog(title, message, button_text="确定", key=None):
    """
    显示提示对话框
    
    Args:
        title (str): 对话框标题
        message (str): 对话框消息
        button_text (str, optional): 按钮文本
        key (str, optional): 对话框唯一标识符
    """
    # 生成唯一key
    if key is None:
        key = f"alert_{uuid.uuid4().hex[:8]}"
    
    # 初始化状态
    if f"{key}_dismissed" not in st.session_state:
        st.session_state[f"{key}_dismissed"] = False
    
    # 定义内容渲染函数
    def render_content(modal_key):
        st.markdown(message)
        
        if st.button(button_text, key=f"{modal_key}_ok"):
            st.session_state[f"{key}_dismissed"] = True
            st.session_state[f"{key}_is_open"] = False
            st.rerun()
    
    # 显示对话框
    show_modal(title, render_content, key=key)
    
    # 返回是否已关闭
    return st.session_state[f"{key}_dismissed"]

def loading_dialog(title="处理中", message="请稍候...", key=None):
    """
    显示加载对话框
    
    Args:
        title (str, optional): 对话框标题
        message (str, optional): 对话框消息
        key (str, optional): 对话框唯一标识符
    
    Returns:
        callable: 关闭对话框的函数
    """
    # 生成唯一key
    if key is None:
        key = f"loading_{uuid.uuid4().hex[:8]}"
    
    # 定义内容渲染函数
    def render_content(modal_key):
        st.markdown(message)
        st.spinner("加载中...")
        progress_bar = st.progress(0)
        
        # 模拟进度条
        if f"{key}_progress" not in st.session_state:
            st.session_state[f"{key}_progress"] = 0
        
        # 更新进度条
        progress_bar.progress(st.session_state[f"{key}_progress"])
    
    # 显示对话框
    show_modal(title, render_content, key=key, has_close_button=False)
    
    # 定义关闭函数
    def close_dialog():
        st.session_state[f"{key}_is_open"] = False
        st.rerun()
    
    # 定义更新进度函数
    def update_progress(value):
        if 0 <= value <= 1:
            st.session_state[f"{key}_progress"] = value
            st.rerun()
    
    # 返回关闭和更新进度函数
    return close_dialog, update_progress

if __name__ == "__main__":
    # 测试代码
    st.title("Modal Component Test")
    
    if st.button("Show Confirmation Dialog"):
        result = confirmation_dialog(
            "确认操作", 
            "您确定要执行此操作吗？此操作无法撤销。",
            "确认执行",
            "取消操作"
        )
        
        if result is True:
            st.success("用户已确认")
        elif result is False:
            st.error("用户已取消")
    
    if st.button("Show Alert Dialog"):
        alert_dialog(
            "操作成功", 
            "您的操作已成功完成。"
        )
    
    if st.button("Show Loading Dialog"):
        close_fn, update_fn = loading_dialog("处理数据", "正在处理您的数据，请稍候...")
        
        # 模拟进度更新
        for i in range(11):
            update_fn(i / 10)
            time.sleep(0.2)
        
        close_fn()
        st.success("处理完成！")