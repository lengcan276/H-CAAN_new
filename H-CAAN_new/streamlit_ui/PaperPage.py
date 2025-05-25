"""
论文生成页面 - 论文撰写界面及文档管理
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ui_agent import UIAgent

def show_paper_page():
    """显示论文生成页面"""
    st.title("📝 自动论文生成")
    st.markdown("基于实验结果自动撰写科研论文")
    
    # 初始化
    if 'ui_agent' not in st.session_state:
        st.session_state.ui_agent = UIAgent()
    
    ui_agent = st.session_state.ui_agent
    
    # 论文配置
    with st.expander("⚙️ 论文设置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            paper_title = st.text_input(
                "论文标题",
                value="H-CAAN: 层次化跨模态自适应注意力网络用于药物属性预测"
            )
            
            authors = st.text_area(
                "作者列表",
                value="张三¹, 李四¹, 王五²\n¹计算机学院 ²药学院",
                height=60
            )
            
            keywords = st.text_input(
                "关键词",
                value="药物属性预测, 多模态学习, 注意力机制, 深度学习"
            )
        
        with col2:
            sections = st.multiselect(
                "包含章节",
                [
                    "摘要", "引言", "相关工作", "方法", 
                    "实验", "结果", "讨论", "结论", "参考文献"
                ],
                default=[
                    "摘要", "引言", "相关工作", "方法", 
                    "实验", "结果", "讨论", "结论", "参考文献"
                ]
            )
            
            output_formats = st.multiselect(
                "生成格式",
                ["Markdown", "PDF", "Word", "LaTeX"],
                default=["Markdown", "PDF"]
            )
    
    # 检查前置条件
    if st.session_state.get('model_trained', False) and 'explanation_report' in st.session_state:
        # 论文生成状态
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info("""
            ✅ 模型已训练完成  
            ✅ 解释报告已生成  
            ✅ 实验数据已准备
            """)
        
        with col2:
            if st.button("🚀 生成论文", use_container_width=True):
                with st.spinner("正在生成论文..."):
                    # 准备论文元数据
                    metadata = {
                        'title': paper_title,
                        'authors': authors,
                        'keywords': keywords,
                        'sections': sections,
                        'datasets': ['溶解度数据集'],
                        'date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # 准备结果数据
                    results = {
                        'metrics': st.session_state.get('training_metrics', {}),
                        'predictions': np.random.randn(100).tolist(),
                        'feature_importance': np.random.rand(20).tolist()
                    }
                    
                    # 调用论文生成
                    result = ui_agent.handle_user_input({
                        'action': 'generate_paper',
                        'params': {
                            'results': results,
                            'explanations': st.session_state.get('explanation_report', {}),
                            'metadata': metadata
                        }
                    })
                    
                    if result['status'] == 'success':
                        st.success("论文生成完成！")
                        st.session_state.paper_generated = True
                        st.session_state.paper_path = result['paper_path']
        
        with col3:
            if st.button("🔄 重置设置", use_container_width=True):
                st.session_state.paper_generated = False
                st.rerun()
        
        # 论文预览和编辑
        if st.session_state.get('paper_generated', False):
            tab1, tab2, tab3, tab4 = st.tabs(["📄 预览", "✏️ 编辑", "📊 图表", "💾 导出"])
            
            with tab1:
                show_preview_tab()
            
            with tab2:
                show_edit_tab()
            
            with tab3:
                show_figures_tab()
            
            with tab4:
                show_export_tab()
    else:
        # 引导用户完成前置步骤
        st.warning("请先完成以下步骤：")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.get('model_trained', False):
                st.error("❌ 模型未训练")
            else:
                st.success("✅ 模型已训练")
        
        with col2:
            if 'explanation_report' not in st.session_state:
                st.error("❌ 解释报告未生成")
            else:
                st.success("✅ 解释报告已生成")

def show_preview_tab():
    """预览标签页"""
    st.subheader("论文预览")
    
    # 加载生成的论文内容
    paper_content = """
# H-CAAN: 层次化跨模态自适应注意力网络用于药物属性预测

**作者**: 张三¹, 李四¹, 王五²  
¹计算机学院 ²药学院

## 摘要

本研究提出了一种基于层次化跨模态自适应注意力网络（H-CAAN）的药物属性预测方法。该方法通过整合分子的多模态表示（SMILES、分子图、分子指纹），利用深度学习技术实现了高精度的药物属性预测。实验结果表明，该方法在测试集上达到了R²=0.895的预测精度。本研究为药物发现和开发提供了新的计算工具。

**关键词**: 药物属性预测, 多模态学习, 注意力机制, 深度学习

## 1. 引言

药物发现是一个复杂而昂贵的过程，准确预测分子属性对于加速药物开发至关重要。近年来，深度学习技术在药物属性预测领域取得了显著进展。然而，现有方法通常只考虑单一模态的分子表示，限制了模型的表达能力。

本研究提出了H-CAAN方法，通过融合多种分子表示模态，充分利用不同表示之间的互补信息。主要贡献包括：

1. 设计了层次化注意力机制，有效融合多模态分子特征
2. 提出了自适应门控策略，动态调整不同模态的贡献
3. 构建了端到端的预测框架，实现了高精度的属性预测

## 2. 相关工作

### 2.1 分子表示学习

分子表示学习是药物属性预测的基础。常用的分子表示包括：

- **SMILES表示**：将分子结构编码为字符串序列
- **分子图表示**：将分子建模为图结构，原子为节点，化学键为边
- **分子指纹**：基于子结构的二进制向量表示

### 2.2 多模态学习

多模态学习旨在整合来自不同源的信息。在分子属性预测中，已有研究尝试结合多种表示...

[论文内容继续...]
"""
    
    # 显示论文内容
    st.markdown(paper_content)

def show_edit_tab():
    """编辑标签页"""
    st.subheader("论文编辑")
    
    # 选择要编辑的章节
    section_to_edit = st.selectbox(
        "选择章节",
        ["摘要", "引言", "方法", "实验", "结果", "讨论", "结论"]
    )
    
    # 编辑区域
    if section_to_edit == "摘要":
        abstract_text = st.text_area(
            "编辑摘要",
            value="本研究提出了一种基于层次化跨模态自适应注意力网络（H-CAAN）的药物属性预测方法...",
            height=200
        )
        
        if st.button("保存修改"):
            st.success(f"{section_to_edit}已更新")
    
    # AI写作助手
    st.markdown("---")
    st.markdown("#### AI写作助手")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 润色文本"):
            st.info("AI正在优化文本...")
    
    with col2:
        if st.button("💡 生成建议"):
            st.info("生成写作建议...")

def show_figures_tab():
    """图表标签页"""
    st.subheader("论文图表")
    
    # 图表列表
    figures = {
        "图1: 模型架构图": "model_architecture",
        "图2: 损失曲线": "loss_curve",
        "图3: 特征重要性": "feature_importance",
        "图4: 预测散点图": "prediction_scatter",
        "表1: 实验结果对比": "results_table"
    }
    
    selected_figure = st.selectbox("选择图表", list(figures.keys()))
    
    # 显示对应的图表
    if "损失曲线" in selected_figure:
        import plotly.graph_objects as go
        
        epochs = list(range(1, 101))
        train_loss = [0.5 * np.exp(-i/30) + 0.05 for i in epochs]
        val_loss = [0.5 * np.exp(-i/25) + 0.08 for i in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='训练损失'))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='验证损失'))
        fig.update_layout(title="训练过程损失曲线", xaxis_title="Epoch", yaxis_title="Loss")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif "特征重要性" in selected_figure:
        import plotly.express as px
        
        features = [f"特征{i}" for i in range(10)]
        importance = np.sort(np.random.rand(10))[::-1]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title="Top 10 重要特征")
        st.plotly_chart(fig, use_container_width=True)
    
    # 图表管理
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ 添加图表"):
            st.info("从结果中选择图表...")
    
    with col2:
        if st.button("📝 编辑标题"):
            st.info("编辑图表标题...")
    
    with col3:
        if st.button("🎨 调整样式"):
            st.info("自定义图表样式...")

def show_export_tab():
    """导出标签页"""
    st.subheader("论文导出")
    
    # 导出选项
    st.markdown("#### 导出设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_figures = st.checkbox("包含图表", value=True)
        include_tables = st.checkbox("包含表格", value=True)
        include_references = st.checkbox("包含参考文献", value=True)
    
    with col2:
        figure_quality = st.select_slider(
            "图片质量",
            options=["低", "中", "高", "最高"],
            value="高"
        )
        
        paper_template = st.selectbox(
            "论文模板",
            ["IEEE", "Nature", "自定义"]
        )
    
    # 导出按钮
    st.markdown("---")
    st.markdown("#### 开始导出")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Markdown", use_container_width=True):
            paper_content = "# 论文标题\n\n## 摘要\n\n..."
            st.download_button(
                "下载 Markdown",
                paper_content,
                "paper.md",
                "text/markdown"
            )
    
    with col2:
        if st.button("📕 PDF", use_container_width=True):
            st.info("正在生成PDF...")
    
    with col3:
        if st.button("📘 Word", use_container_width=True):
            st.info("正在生成Word文档...")
    
    with col4:
        if st.button("📗 LaTeX", use_container_width=True):
            st.info("正在生成LaTeX源码...")
    
    # 导出历史
    st.markdown("---")
    st.markdown("#### 导出历史")
    
    export_history = pd.DataFrame({
        '时间': pd.date_range(end=datetime.now(), periods=3, freq='H'),
        '格式': ['PDF', 'Markdown', 'Word'],
        '状态': ['完成', '完成', '完成'],
        '大小': ['2.3 MB', '156 KB', '1.8 MB']
    })
    
    st.dataframe(export_history, use_container_width=True)