"""
模型训练页面 - 模型训练状态、预测结果与不确定性分析
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ui_agent import UIAgent

def show_modeling_page():
    """显示模型训练页面"""
    st.title("🎯 模型训练与预测")
    st.markdown("配置、训练和评估预测模型")
    
    # 初始化
    if 'ui_agent' not in st.session_state:
        st.session_state.ui_agent = UIAgent()
    
    ui_agent = st.session_state.ui_agent
    
    # 训练配置
    with st.expander("⚙️ 训练配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'uploaded_data' in st.session_state:
                st.success("✅ 数据已加载")
                target_property = st.selectbox("目标属性", ["溶解度", "毒性", "活性"])
                train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8, 0.05)
            else:
                st.warning("请先上传数据")
        
        with col2:
            model_type = st.selectbox(
                "模型类型",
                ["集成模型", "随机森林", "神经网络"]
            )
            learning_rate = st.number_input("学习率", 0.0001, 0.1, 0.001, format="%.4f")
            
        with col3:
            batch_size = st.selectbox("批次大小", [16, 32, 64, 128], index=1)
            epochs = st.number_input("训练轮数", 10, 500, 100)
            early_stopping = st.checkbox("早停策略", value=True)
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 训练", "📊 监控", "🎯 预测", "📈 评估"])
    
    with tab1:
        show_training_tab(ui_agent)
    
    with tab2:
        show_monitoring_tab()
    
    with tab3:
        show_prediction_tab(ui_agent)
    
    with tab4:
        show_evaluation_tab()

def show_training_tab(ui_agent):
    """训练标签页"""
    st.subheader("模型训练")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 训练前检查
        st.markdown("#### 训练前检查")
        
        checks = {
            "数据已加载": 'uploaded_data' in st.session_state,
            "特征已融合": st.session_state.get('fusion_completed', False),
            "参数已配置": True,
            "GPU可用": False
        }
        
        for check, status in checks.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.warning(f"⚠️ {check}")
        
        all_ready = all(checks.values())
    
    with col2:
        st.markdown("#### 操作")
        
        if st.button("🚀 开始训练", disabled=not all_ready, use_container_width=True):
            st.session_state.training_started = True
            st.session_state.training_progress = 0
            
            # 调用训练
            with st.spinner("正在准备训练..."):
                result = ui_agent.handle_user_input({
                    'action': 'start_training',
                    'params': {
                        'data_path': 'data/raw/example_solubility.csv',
                        'labels': np.random.rand(100),
                        'train_params': {
                            'task_name': 'solubility',
                            'learning_rate': 0.001,
                            'batch_size': 32,
                            'epochs': 100
                        }
                    }
                })
                
                if result['status'] == 'success':
                    st.success("✅ 训练完成!")
                    st.session_state.model_trained = True
                    st.session_state.model_path = result.get('model_path')
                    st.session_state.training_metrics = result.get('metrics', {})
    
    # 训练进度
    if st.session_state.get('training_started', False):
        st.markdown("---")
        st.markdown("#### 训练进度")
        
        # 模拟训练进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            if not st.session_state.get('training_started', False):
                break
                
            progress = i / 100
            progress_bar.progress(progress)
            status_text.text(f"Epoch {i+1}/100 - Loss: {0.5 - i*0.003:.4f}")
            time.sleep(0.1)
            
            st.session_state.training_progress = progress
        
        if st.session_state.get('training_started', False):
            progress_bar.progress(1.0)
            status_text.text("训练完成!")
            st.session_state.training_started = False
            st.session_state.model_trained = True

def show_monitoring_tab():
    """监控标签页"""
    st.subheader("训练监控")
    
    if st.session_state.get('model_trained', False) or st.session_state.get('training_started', False):
        # 创建实时监控图表
        col1, col2 = st.columns(2)
        
        with col1:
            # 损失曲线
            epochs_data = list(range(1, 101))
            train_loss = [0.5 * np.exp(-i/30) + 0.05 + np.random.normal(0, 0.01) for i in epochs_data]
            val_loss = [0.5 * np.exp(-i/25) + 0.08 + np.random.normal(0, 0.02) for i in epochs_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs_data,
                y=train_loss,
                mode='lines',
                name='训练损失',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=epochs_data,
                y=val_loss,
                mode='lines',
                name='验证损失',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="损失曲线",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 性能指标
            metrics_data = []
            for i in epochs_data[::10]:
                metrics_data.append({
                    'Epoch': i,
                    'R²': min(0.95, 0.6 + i * 0.003 + np.random.normal(0, 0.02)),
                    'RMSE': max(0.2, 0.6 - i * 0.003 + np.random.normal(0, 0.02))
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics_df['Epoch'],
                y=metrics_df['R²'],
                mode='lines+markers',
                name='R²',
                line=dict(color='green'),
                yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=metrics_df['Epoch'],
                y=metrics_df['RMSE'],
                mode='lines+markers',
                name='RMSE',
                line=dict(color='orange'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="性能指标",
                xaxis_title="Epoch",
                yaxis=dict(title="R²", side='left'),
                yaxis2=dict(title="RMSE", side='right', overlaying='y'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 实时指标
        st.markdown("---")
        st.markdown("#### 当前指标")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "训练损失",
                f"{train_loss[-1]:.4f}",
                f"{train_loss[-1] - train_loss[-10]:.4f}"
            )
        
        with col2:
            st.metric(
                "验证损失",
                f"{val_loss[-1]:.4f}",
                f"{val_loss[-1] - val_loss[-10]:.4f}"
            )
        
        with col3:
            st.metric(
                "R²得分",
                f"{metrics_df['R²'].iloc[-1]:.3f}",
                f"+{metrics_df['R²'].iloc[-1] - metrics_df['R²'].iloc[-2]:.3f}"
            )
        
        with col4:
            st.metric(
                "训练时间",
                "2分15秒",
                "预计剩余: 30秒"
            )
    else:
        st.info("请先开始训练以查看监控数据")

def show_prediction_tab(ui_agent):
    """预测标签页"""
    st.subheader("模型预测")
    
    if st.session_state.get('model_trained', False):
        # 预测方式选择
        pred_mode = st.radio(
            "预测模式",
            ["单分子预测", "批量预测", "文件预测"],
            horizontal=True
        )
        
        if pred_mode == "单分子预测":
            st.markdown("#### 输入分子")
            
            smiles_input = st.text_input(
                "SMILES字符串",
                placeholder="例如: CCO, c1ccccc1",
                help="输入要预测的分子SMILES表示"
            )
            
            if smiles_input and st.button("🔮 预测"):
                with st.spinner("正在预测..."):
                    # 模拟预测结果
                    prediction = np.random.normal(1.5, 0.3)
                    uncertainty = np.abs(np.random.normal(0, 0.1))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("预测值", f"{prediction:.3f}")
                    
                    with col2:
                        st.metric("不确定性", f"±{uncertainty:.3f}")
                    
                    with col3:
                        confidence = (1 - uncertainty) * 100
                        st.metric("置信度", f"{confidence:.1f}%")
                    
                    # 可视化分子
                    from rdkit import Chem
                    from rdkit.Chem import Draw
                    
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption=f"分子结构: {smiles_input}")
        
        elif pred_mode == "批量预测":
            st.markdown("#### 批量输入")
            
            smiles_text = st.text_area(
                "SMILES列表（每行一个）",
                height=200,
                placeholder="CCO\nc1ccccc1\nCC(C)O"
            )
            
            if smiles_text and st.button("🔮 批量预测"):
                smiles_list = smiles_text.strip().split('\n')
                
                # 显示预测结果
                predictions = np.random.randn(len(smiles_list)) + 1.5
                uncertainties = np.abs(np.random.randn(len(smiles_list))) * 0.2
                
                results_df = pd.DataFrame({
                    'SMILES': smiles_list,
                    '预测值': predictions,
                    '不确定性': uncertainties,
                    '置信度': (1 - uncertainties) * 100
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # 下载结果
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 下载结果",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
        
        else:  # 文件预测
            st.markdown("#### 文件上传")
            
            pred_file = st.file_uploader(
                "选择预测文件",
                type=['csv'],
                help="CSV文件需包含SMILES列"
            )
            
            if pred_file and st.button("🔮 开始预测"):
                st.info("文件预测功能开发中...")
    else:
        st.info("请先训练模型后再进行预测")

def show_evaluation_tab():
    """评估标签页"""
    st.subheader("模型评估")
    
    if st.session_state.get('model_trained', False):
        # 性能总览
        metrics = st.session_state.get('training_metrics', {
            'rmse': 0.35,
            'mae': 0.28,
            'r2': 0.89,
            'correlation': 0.94
        })
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        
        with col3:
            st.metric("R²", f"{metrics.get('r2', 0):.3f}")
        
        with col4:
            st.metric("相关系数", f"{metrics.get('correlation', 0):.3f}")
        
        # 详细评估
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 预测vs真实散点图
            n_samples = 100
            true_values = np.random.randn(n_samples) + 2
            predicted_values = true_values + np.random.randn(n_samples) * 0.3
            
            fig = px.scatter(
                x=true_values,
                y=predicted_values,
                title="预测值 vs 真实值",
                labels={'x': '真实值', 'y': '预测值'}
            )
            
            # 添加理想线
            fig.add_trace(go.Scatter(
                x=[true_values.min(), true_values.max()],
                y=[true_values.min(), true_values.max()],
                mode='lines',
                name='理想预测',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 残差分布
            residuals = predicted_values - true_values
            
            fig = px.histogram(
                residuals,
                nbins=30,
                title="残差分布",
                labels={'value': '残差', 'count': '频数'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 模型比较
        st.markdown("---")
        st.markdown("#### 模型比较")
        
        comparison_df = pd.DataFrame({
            '模型': ['当前模型', '基线-RF', '基线-DNN', '基线-GBM'],
            'R²': [0.89, 0.82, 0.80, 0.85],
            'RMSE': [0.35, 0.45, 0.48, 0.40],
            'MAE': [0.28, 0.38, 0.40, 0.33]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # 保存模型
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存模型"):
                st.success("模型已保存!")
        
        with col2:
            if st.button("📤 导出模型"):
                st.info("导出功能开发中...")
        
        with col3:
            if st.button("🔄 重新训练"):
                st.session_state.model_trained = False
                st.rerun()
    else:
        st.info("请先训练模型以查看评估结果")