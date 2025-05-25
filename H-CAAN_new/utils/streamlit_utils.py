"""
Streamlit UI辅助函数
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO

def create_download_link(data: Any, filename: str, text: str = "下载") -> str:
    """创建下载链接"""
    if isinstance(data, pd.DataFrame):
        towrite = BytesIO()
        data.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        mime = "text/csv"
    elif isinstance(data, str):
        b64 = base64.b64encode(data.encode()).decode()
        mime = "text/plain"
    else:
        return ""
    
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_metrics_row(metrics: Dict[str, float], delta: bool = True):
    """显示一行指标"""
    cols = st.columns(len(metrics))
    
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            if delta and isinstance(value, (list, tuple)) and len(value) == 2:
                st.metric(name, value[0], value[1])
            else:
                st.metric(name, value)

def create_progress_bar(current: int, total: int, text: str = ""):
    """创建进度条"""
    progress = current / total if total > 0 else 0
    st.progress(progress)
    if text:
        st.text(f"{text}: {current}/{total} ({progress*100:.1f}%)")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None) -> go.Figure:
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    fig = px.imshow(
        cm,
        labels=dict(x="预测", y="实际", color="数量"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(title="混淆矩阵")
    return fig

def create_3d_molecule_view(mol_data: Dict) -> go.Figure:
    """创建3D分子视图"""
    # 这里是简化版本，实际需要3D坐标
    fig = go.Figure(data=[
        go.Scatter3d(
            x=mol_data.get('x', [0, 1, 2]),
            y=mol_data.get('y', [0, 1, 0]),
            z=mol_data.get('z', [0, 0, 1]),
            mode='markers+text',
            marker=dict(
                size=12,
                color=mol_data.get('colors', ['red', 'blue', 'green'])
            ),
            text=mol_data.get('atoms', ['C', 'N', 'O'])
        )
    ])
    
    fig.update_layout(
        title="分子3D结构",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    
    return fig

def format_dataframe_for_display(df: pd.DataFrame, 
                               max_rows: int = 100) -> pd.DataFrame:
    """格式化DataFrame用于显示"""
    if len(df) > max_rows:
        st.warning(f"数据量较大，只显示前{max_rows}行")
        df = df.head(max_rows)
    
    # 格式化数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].dtype == float:
            df[col] = df[col].round(4)
    
    return df

def create_sidebar_filters(data: pd.DataFrame) -> Dict:
    """创建侧边栏过滤器"""
    filters = {}
    
    st.sidebar.header("数据过滤")
    
    # 数值列过滤
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        
        filters[col] = st.sidebar.slider(
            f"{col}范围",
            min_val, max_val, (min_val, max_val)
        )
    
    # 分类列过滤
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_vals = data[col].unique()
        if len(unique_vals) < 20:  # 只对类别数较少的列创建过滤器
            filters[col] = st.sidebar.multiselect(
                f"{col}",
                unique_vals,
                default=unique_vals
            )
    
    return filters

def apply_filters(data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """应用过滤器"""
    filtered_data = data.copy()
    
    for col, filter_value in filters.items():
        if col in data.columns:
            if isinstance(filter_value, tuple) and len(filter_value) == 2:
                # 范围过滤
                filtered_data = filtered_data[
                    (filtered_data[col] >= filter_value[0]) &
                    (filtered_data[col] <= filter_value[1])
                ]
            elif isinstance(filter_value, list):
                # 多选过滤
                filtered_data = filtered_data[filtered_data[col].isin(filter_value)]
    
    return filtered_data

def create_interactive_plot(plot_type: str, data: pd.DataFrame,
                          x_col: str, y_col: str, **kwargs) -> go.Figure:
    """创建交互式图表"""
    if plot_type == 'scatter':
        fig = px.scatter(data, x=x_col, y=y_col, **kwargs)
    elif plot_type == 'line':
        fig = px.line(data, x=x_col, y=y_col, **kwargs)
    elif plot_type == 'bar':
        fig = px.bar(data, x=x_col, y=y_col, **kwargs)
    elif plot_type == 'box':
        fig = px.box(data, x=x_col, y=y_col, **kwargs)
    elif plot_type == 'histogram':
        fig = px.histogram(data, x=x_col, **kwargs)
    else:
        fig = px.scatter(data, x=x_col, y=y_col, **kwargs)
    
    # 更新布局
    fig.update_layout(
        hovermode='closest',
        dragmode='pan',
        showlegend=True
    )
    
    return fig