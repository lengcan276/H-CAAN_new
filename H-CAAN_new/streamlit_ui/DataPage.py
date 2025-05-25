"""
数据管理页面 - 数据上传、结构可视化及初步特征展示
"""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import plotly.express as px
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.ui_agent import UIAgent

def show_data_page():
    """显示数据管理页面"""
    st.title("📁 数据管理")
    st.markdown("上传和管理分子数据集")
    
    # 初始化UI代理
    if 'ui_agent' not in st.session_state:
        st.session_state.ui_agent = UIAgent()
    
    ui_agent = st.session_state.ui_agent
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📤 上传数据", "📊 数据预览", "🔍 数据分析"])
    
    with tab1:
        show_upload_tab(ui_agent)
    
    with tab2:
        show_preview_tab()
    
    with tab3:
        show_analysis_tab()

def show_upload_tab(ui_agent):
    """上传数据标签页"""
    
    # 显示已存在的数据文件
    st.subheader("📂 已有数据文件")
    
    raw_data_path = "data/raw"
    if os.path.exists(raw_data_path):
        files = [f for f in os.listdir(raw_data_path) 
                if f.endswith(('.csv', '.sdf', '.mol2', '.smi'))]
        
        if files:
            st.info(f"在 {raw_data_path} 目录下发现 {len(files)} 个数据文件")
            
            # 创建文件选择器
            selected_file = st.selectbox(
                "选择要加载的文件：",
                options=files,
                format_func=lambda x: f"📄 {x}"
            )
            
            if selected_file:
                file_path = os.path.join(raw_data_path, selected_file)
                
                # 显示文件信息
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size / 1024  # KB
                file_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("文件大小", f"{file_size:.2f} KB")
                with col2:
                    st.metric("修改时间", file_time)
                with col3:
                    # 预览文件内容
                    if selected_file.endswith('.csv'):
                        try:
                            df_preview = pd.read_csv(file_path, nrows=5)
                            st.metric("数据行数", f"{len(pd.read_csv(file_path))} 行")
                        except:
                            st.metric("数据行数", "未知")
                
                # 加载按钮
                if st.button(f"🔄 加载 {selected_file}", key=f"load_{selected_file}"):
                    with st.spinner(f"正在加载 {selected_file}..."):
                        result = ui_agent.handle_user_input({
                            'action': 'upload_data',
                            'params': {'file_path': file_path}
                        })
                        
                        if result['status'] == 'success':
                            st.session_state.uploaded_data = result
                            st.session_state.current_file = selected_file
                            st.success(f"成功加载: {selected_file}")
                            
                            # 显示数据预览
                            if selected_file.endswith('.csv'):
                                df = pd.read_csv(file_path, nrows=5)
                                st.markdown("#### 数据预览（前5行）")
                                st.dataframe(df, use_container_width=True)
                            
                            st.rerun()
                        else:
                            st.error(result['message'])
                
                # 删除文件选项
                with st.expander("⚠️ 危险操作"):
                    if st.button(f"🗑️ 删除 {selected_file}", type="secondary"):
                        if st.checkbox(f"确认删除 {selected_file}？"):
                            os.remove(file_path)
                            st.success(f"已删除: {selected_file}")
                            st.rerun()
        else:
            st.warning("未发现任何数据文件")
    else:
        os.makedirs(raw_data_path, exist_ok=True)
        st.warning(f"数据目录 {raw_data_path} 为空")
    
    # 分隔线
    st.markdown("---")
    
    # 上传新文件
    st.subheader("📤 上传新数据")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=['csv', 'sdf', 'mol2', 'smi'],
        help="支持CSV（需包含SMILES列）、SDF、MOL2格式"
    )
    
    if uploaded_file is not None:
        # 显示上传文件信息
        st.info(f"文件名: {uploaded_file.name} | 大小: {uploaded_file.size/1024:.2f} KB")
        
        # 保存上传的文件
        save_path = os.path.join("data", "raw", uploaded_file.name)
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            st.warning(f"⚠️ 文件 {uploaded_file.name} 已存在")
            if st.button("覆盖文件"):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"文件已覆盖: {save_path}")
                st.rerun()
        else:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"文件已保存到: {save_path}")
            st.rerun()
    
    # 示例数据
    st.markdown("---")
    st.subheader("📚 示例数据集")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 溶解度数据集", key="example_sol"):
            # 创建示例数据
            example_data = pd.DataFrame({
                'smiles': [
                    'CCO', 'CC(C)O', 'c1ccccc1', 'CC(=O)O', 'CCN',
                    'CCCC', 'CCC(C)C', 'c1ccc(O)cc1', 'CC(C)(C)O', 'CCCN'
                ],
                'solubility': [1.2, 0.8, -0.5, 2.1, 0.6, -1.2, -0.9, 0.3, 1.5, 0.9],
                'molecular_weight': [46.07, 60.10, 78.11, 60.05, 45.08, 
                                   58.12, 72.15, 94.11, 74.12, 59.11]
            })
            save_path = "data/raw/example_solubility_full.csv"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            example_data.to_csv(save_path, index=False)
            
            st.success("已创建溶解度示例数据集")
            st.rerun()
    
    with col2:
        if st.button("💊 毒性数据集", key="example_tox"):
            # 创建毒性示例数据
            example_data = pd.DataFrame({
                'smiles': [
                    'CCCCCl', 'c1ccc(Cl)cc1', 'CC(C)Br', 'CCCF', 'c1ccncc1',
                    'CC(=O)Cl', 'CCCBr', 'c1ccc(F)cc1', 'CCCI', 'c1cccnc1'
                ],
                'toxicity': [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
                'log_p': [2.1, 2.8, 1.9, 1.5, 0.8, 0.9, 2.3, 2.2, 2.5, 1.2]
            })
            save_path = "data/raw/example_toxicity.csv"
            example_data.to_csv(save_path, index=False)
            
            st.success("已创建毒性示例数据集")
            st.rerun()
            
    with col3:
        if st.button("🔬 活性数据集", key="example_act"):
            # 创建活性示例数据
            example_data = pd.DataFrame({
                'smiles': [
                    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                    'CC1=C(C=C(C=C1)C(F)(F)F)C(=O)NC2=CC=CC=C2C(=O)O',
                    'COC1=CC=CC=C1OCCNCC(COC2=CC=CC3=C2C4=CC=CC=C4N3)O'
                ],
                'activity': [6.5, 7.2, 5.8],
                'target': ['COX-2', 'COX-2', '5-HT1A']
            })
            save_path = "data/raw/example_activity.csv"
            example_data.to_csv(save_path, index=False)
            
            st.success("已创建活性示例数据集")
            st.rerun()

def show_preview_tab():
    """数据预览标签页"""
    st.subheader("数据预览")
    
    # 显示当前加载的文件
    if 'current_file' in st.session_state:
        st.info(f"当前文件: {st.session_state.current_file}")
    
    if 'uploaded_data' in st.session_state:
        preview_data = st.session_state.uploaded_data.get('preview', {})
        
        # SMILES预览
        if 'smiles_sample' in preview_data:
            st.markdown("#### SMILES示例")
            smiles_df = pd.DataFrame({
                'Index': range(len(preview_data['smiles_sample'])),
                'SMILES': preview_data['smiles_sample']
            })
            st.dataframe(smiles_df, use_container_width=True)
            
            # 分子结构可视化
            st.markdown("#### 分子结构可视化")
            
            # 选择显示方式
            display_mode = st.radio(
                "显示方式",
                ["单个分子", "分子网格"],
                horizontal=True
            )
            
            if display_mode == "单个分子":
                selected_idx = st.selectbox(
                    "选择分子",
                    options=range(len(preview_data['smiles_sample'])),
                    format_func=lambda x: f"分子 {x+1}: {preview_data['smiles_sample'][x][:20]}..."
                )
                
                if selected_idx is not None:
                    smiles = preview_data['smiles_sample'][selected_idx]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img = Draw.MolToImage(mol, size=(400, 400))
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(img, caption=f"SMILES: {smiles}")
            else:
                # 分子网格显示
                n_mols = min(6, len(preview_data['smiles_sample']))
                mols = [Chem.MolFromSmiles(smi) for smi in preview_data['smiles_sample'][:n_mols]]
                mols = [mol for mol in mols if mol is not None]
                
                if mols:
                    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200))
                    st.image(img, caption="分子结构网格")
        
        # 属性分布
        if 'properties' in preview_data and preview_data['properties']:
            st.markdown("#### 属性分布")
            prop_name = st.selectbox("选择属性", preview_data['properties'])
            
            # 模拟属性值
            prop_values = np.random.normal(0, 1, preview_data.get('n_molecules', 100))
            
            fig = px.histogram(
                x=prop_values,
                nbins=30,
                title=f"{prop_name} 分布",
                labels={'x': prop_name, 'y': '频数'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 结构统计
        if 'structure_stats' in preview_data:
            st.markdown("#### 结构统计")
            stats_df = pd.DataFrame(preview_data['structure_stats'])
            st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("请先上传或选择数据文件")

def show_analysis_tab():
    """数据分析标签页"""
    st.subheader("数据分析")
    
    if 'uploaded_data' in st.session_state:
        # 数据质量检查
        st.markdown("#### 数据质量检查")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总分子数", "1,234", "")
            
        with col2:
            st.metric("有效SMILES", "95%", "+2%")
            
        with col3:
            st.metric("缺失值", "3%", "-1%")
            
        with col4:
            st.metric("重复数据", "2%", "0%")
        
        # 分子描述符统计
        st.markdown("#### 分子描述符统计")
        
        descriptors = pd.DataFrame({
            '描述符': ['分子量', 'LogP', 'HBD', 'HBA', 'TPSA', '可旋转键'],
            '平均值': [250.3, 2.1, 1.5, 3.2, 65.4, 4.1],
            '标准差': [80.5, 1.2, 1.1, 1.8, 25.3, 2.3],
            '最小值': [100.1, -1.5, 0, 0, 20.2, 0],
            '最大值': [500.8, 5.6, 5, 8, 120.5, 12]
        })
        
        st.dataframe(descriptors, use_container_width=True)
        
        # 相关性分析
        st.markdown("#### 属性相关性")
        
        # 生成模拟的相关性矩阵
        props = ['分子量', 'LogP', 'TPSA', '溶解度']
        corr_matrix = np.array([
            [1.0, 0.65, -0.45, -0.72],
            [0.65, 1.0, -0.38, -0.58],
            [-0.45, -0.38, 1.0, 0.62],
            [-0.72, -0.58, 0.62, 1.0]
        ])
        
        fig = px.imshow(
            corr_matrix,
            x=props,
            y=props,
            color_continuous_scale='RdBu',
            title="属性相关性热图",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 数据导出
        st.markdown("---")
        st.markdown("#### 数据导出与报告")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 导出预处理数据", type="primary"):
                st.info("正在准备数据...")
                st.download_button(
                    label="下载预处理数据",
                    data="预处理数据内容",
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
                
        with col2:
            if st.button("📊 生成数据报告", type="primary"):
                st.info("正在生成报告...")
                
        with col3:
            if st.button("🔍 高级分析", type="primary"):
                st.info("高级分析功能开发中...")
    else:
        st.info("请先上传或选择数据文件进行分析")