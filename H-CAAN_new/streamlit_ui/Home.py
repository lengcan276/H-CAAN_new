"""
系统主页 - 介绍与导航
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    st.set_page_config(
        page_title="H-CAAN 多智能体系统",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 创建侧边栏导航
    with st.sidebar:
        st.title("🧬 H-CAAN")
        st.markdown("---")
        
        page = st.radio(
            "导航",
            ["🏠 主页", "📁 数据管理", "🔄 特征融合", "🎯 模型训练", 
             "📊 模型解释", "📝 论文生成"]
        )
    
    # 根据选择显示不同页面
    if page == "🏠 主页":
        show_home_page()
    elif page == "📁 数据管理":
        from DataPage import show_data_page
        show_data_page()
    elif page == "🔄 特征融合":
        from FusionPage import show_fusion_page
        show_fusion_page()
    elif page == "🎯 模型训练":
        from ModelingPage import show_modeling_page
        show_modeling_page()
    elif page == "📊 模型解释":
        from ExplanationPage import show_explanation_page
        show_explanation_page()
    elif page == "📝 论文生成":
        from PaperPage import show_paper_page
        show_paper_page()

def show_home_page():
    """显示主页内容"""
    # 标题和介绍
    st.title("🧬 H-CAAN 多智能体药物属性预测系统")
    st.markdown("### 层次化跨模态自适应注意力网络")
    
    # 系统概述
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🎯 系统特点**
        - 多模态分子表示融合
        - 智能体协同工作
        - 端到端自动化流程
        - 可解释性分析
        """)
        
    with col2:
        st.success("""
        **📊 核心功能**
        - 数据处理与特征提取
        - 多模态特征融合
        - 模型训练与预测
        - 自动论文生成
        """)
        
    with col3:
        st.warning("""
        **🔧 技术架构**
        - 深度学习框架
        - 多智能体系统
        - 注意力机制
        - 集成学习
        """)
    
    st.markdown("---")
    
    # 工作流程图
    st.subheader("📋 系统工作流程")
    
    # 创建流程图
    fig = go.Figure()
    
    # 添加节点
    nodes = {
        'data': {'x': 0, 'y': 2, 'text': '数据加载'},
        'preprocess': {'x': 1, 'y': 2, 'text': '预处理'},
        'fusion': {'x': 2, 'y': 2, 'text': '特征融合'},
        'model': {'x': 3, 'y': 2, 'text': '模型训练'},
        'predict': {'x': 4, 'y': 2, 'text': '预测'},
        'explain': {'x': 3, 'y': 1, 'text': '解释分析'},
        'paper': {'x': 4, 'y': 1, 'text': '论文生成'}
    }
    
    # 添加连接线
    edges = [
        ('data', 'preprocess'),
        ('preprocess', 'fusion'),
        ('fusion', 'model'),
        ('model', 'predict'),
        ('model', 'explain'),
        ('explain', 'paper')
    ]
    
    # 绘制边
    for start, end in edges:
        fig.add_trace(go.Scatter(
            x=[nodes[start]['x'], nodes[end]['x']],
            y=[nodes[start]['y'], nodes[end]['y']],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    # 绘制节点
    for node_id, node in nodes.items():
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(size=50, color='lightblue'),
            text=[node['text']],
            textposition='middle center',
            showlegend=False
        ))
    
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 快速开始
    st.subheader("🚀 快速开始")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **第一步：准备数据**
        1. 准备分子数据文件（支持CSV、SDF、MOL2格式）
        2. 确保包含SMILES字符串和目标属性值
        3. 点击侧边栏"数据管理"上传数据
        """)
            
    with col2:
        st.markdown("""
        **第二步：训练模型**
        1. 选择已上传的数据集
        2. 配置训练参数
        3. 启动模型训练
        """)
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("📈 系统状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总任务数", "156", "+12")
        
    with col2:
        st.metric("已完成", "143", "+10")
        
    with col3:
        st.metric("运行中", "8", "+2")
        
    with col4:
        st.metric("失败", "5", "0")
    
    # 最近活动
    st.subheader("🕐 最近活动")
    
    activities = pd.DataFrame({
        '时间': pd.date_range(end=datetime.now(), periods=5, freq='H'),
        '任务': ['模型训练', '数据预处理', '特征融合', '预测分析', '论文生成'],
        '状态': ['完成', '完成', '运行中', '完成', '完成'],
        '耗时': ['2.5分钟', '30秒', '-', '5秒', '1分钟']
    })
    
    st.dataframe(activities, use_container_width=True)
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        H-CAAN Multi-Agent System v1.0 | 
        <a href='https://github.com/your-repo'>GitHub</a> | 
        <a href='#'>文档</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()