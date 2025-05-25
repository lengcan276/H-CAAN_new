"""
图表组件 - 提供各种数据可视化图表
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any

def render_line_chart(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    x_axis: Optional[Union[str, List]] = None,
    y_axis: Optional[Union[str, List[str]]] = None,
    title: str = "线图",
    x_label: str = "X轴",
    y_label: str = "Y轴",
    use_plotly: bool = True,
    height: int = 400,
    color_palette: Optional[Union[str, List[str]]] = None
):
    """
    渲染线图
    
    Args:
        data: 数据源（DataFrame或字典）
        x_axis: X轴数据列名或数据
        y_axis: Y轴数据列名或列名列表
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        use_plotly: 是否使用Plotly（否则使用Matplotlib）
        height: 图表高度
        color_palette: 颜色板名称或颜色列表
    """
    if use_plotly:
        # 使用Plotly绘图
        if isinstance(data, dict):
            # 字典数据转换为DataFrame
            if x_axis is None:
                # 如果没有提供x_axis，使用索引作为x轴
                x_values = list(range(len(list(data.values())[0])))
            else:
                x_values = x_axis
            
            # 构建DataFrame
            df = pd.DataFrame({k: v for k, v in data.items() if k != x_axis})
            if x_axis is not None and isinstance(x_axis, str) and x_axis in data:
                df[x_axis] = data[x_axis]
            else:
                df["index"] = x_values
                x_axis = "index"
            
            # 创建图表
            fig = px.line(
                df, 
                x=x_axis, 
                y=df.columns.difference([x_axis]).tolist(),
                title=title,
                labels={x_axis: x_label},
                height=height,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
            )
        else:
            # DataFrame数据
            if y_axis is None:
                # 如果没有提供y_axis，使用所有数值列
                y_axis = data.select_dtypes(include=[np.number]).columns.tolist()
            elif isinstance(y_axis, str):
                y_axis = [y_axis]
            
            if x_axis is None:
                # 如果没有提供x_axis，使用索引
                data = data.reset_index()
                x_axis = "index"
            
            # 创建图表
            fig = px.line(
                data, 
                x=x_axis, 
                y=y_axis,
                title=title,
                labels={x_axis: x_label},
                height=height,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
            )
        
        # 更新布局
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title="数据系列",
            hovermode="x unified"
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 使用Matplotlib绘图
        fig, ax = plt.subplots(figsize=(10, height/100))
        
        if isinstance(data, dict):
            # 字典数据
            if x_axis is None:
                # 如果没有提供x_axis，使用索引作为x轴
                x_values = list(range(len(list(data.values())[0])))
            else:
                if isinstance(x_axis, str) and x_axis in data:
                    x_values = data[x_axis]
                else:
                    x_values = x_axis
            
            # 设置颜色
            if color_palette is not None:
                if isinstance(color_palette, str):
                    colors = sns.color_palette(color_palette, n_colors=len(data) - (1 if x_axis in data else 0))
                else:
                    colors = color_palette
            else:
                colors = None
            
            # 绘制每个系列
            color_index = 0
            for key, values in data.items():
                if key != x_axis:
                    if colors is not None:
                        ax.plot(x_values, values, label=key, color=colors[color_index])
                    else:
                        ax.plot(x_values, values, label=key)
                    color_index += 1
        else:
            # DataFrame数据
            if y_axis is None:
                # 如果没有提供y_axis，使用所有数值列
                y_axis = data.select_dtypes(include=[np.number]).columns.tolist()
            elif isinstance(y_axis, str):
                y_axis = [y_axis]
            
            if x_axis is None:
                # 如果没有提供x_axis，使用索引
                x_values = data.index
            else:
                x_values = data[x_axis]
            
            # 设置颜色
            if color_palette is not None:
                if isinstance(color_palette, str):
                    colors = sns.color_palette(color_palette, n_colors=len(y_axis))
                else:
                    colors = color_palette
            else:
                colors = None
            
            # 绘制每个系列
            for i, y in enumerate(y_axis):
                if colors is not None:
                    ax.plot(x_values, data[y], label=y, color=colors[i % len(colors)])
                else:
                    ax.plot(x_values, data[y], label=y)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图表
        st.pyplot(fig)

def render_bar_chart(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    x_axis: Optional[Union[str, List]] = None,
    y_axis: Optional[Union[str, List[str]]] = None,
    title: str = "柱状图",
    x_label: str = "X轴",
    y_label: str = "Y轴",
    use_plotly: bool = True,
    height: int = 400,
    color_palette: Optional[Union[str, List[str]]] = None,
    orientation: str = "vertical"
):
    """
    渲染柱状图
    
    Args:
        data: 数据源（DataFrame或字典）
        x_axis: X轴数据列名或数据
        y_axis: Y轴数据列名或列名列表
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        use_plotly: 是否使用Plotly（否则使用Matplotlib）
        height: 图表高度
        color_palette: 颜色板名称或颜色列表
        orientation: 柱状图方向，"vertical"或"horizontal"
    """
    if use_plotly:
        # 使用Plotly绘图
        if isinstance(data, dict):
            # 字典数据转换为DataFrame
            if x_axis is None:
                # 如果没有提供x_axis，使用键作为x轴
                x_values = list(data.keys())
                x_axis = "category"
                
                # 构建DataFrame
                df = pd.DataFrame({
                    x_axis: x_values,
                    "value": [data[k] if isinstance(data[k], (int, float)) else data[k][0] for k in x_values]
                })
            else:
                # 如果提供了x_axis，假设data是{series_name: values}格式
                if isinstance(x_axis, str) and x_axis in data:
                    x_values = data[x_axis]
                else:
                    x_values = x_axis
                
                # 构建长格式DataFrame用于分组柱状图
                series_data = []
                for key, values in data.items():
                    if key != x_axis:
                        for i, val in enumerate(values):
                            series_data.append({
                                "x": x_values[i] if i < len(x_values) else i,
                                "y": val,
                                "series": key
                            })
                
                df = pd.DataFrame(series_data)
                
                # 创建分组柱状图
                if orientation == "vertical":
                    fig = px.bar(
                        df,
                        x="x",
                        y="y",
                        color="series",
                        title=title,
                        labels={"x": x_label, "y": y_label, "series": "数据系列"},
                        height=height,
                        barmode="group",
                        color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
                    )
                else:
                    fig = px.bar(
                        df,
                        y="x",
                        x="y",
                        color="series",
                        title=title,
                        labels={"y": x_label, "x": y_label, "series": "数据系列"},
                        height=height,
                        barmode="group",
                        orientation="h",
                        color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
                    )
                
                # 显示图表
                st.plotly_chart(fig, use_container_width=True)
                return
        
        # DataFrame数据或简单字典数据
        if isinstance(data, dict):
            # 简单字典转换为DataFrame
            df = pd.DataFrame({"category": list(data.keys()), "value": list(data.values())})
            x = "category"
            y = "value"
        else:
            # DataFrame数据
            df = data
            
            if y_axis is None:
                # 如果没有提供y_axis，使用第一个数值列
                y = df.select_dtypes(include=[np.number]).columns[0]
            else:
                y = y_axis
                
            if x_axis is None:
                # 如果没有提供x_axis，使用第一个非数值列或索引
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    x = non_numeric_cols[0]
                else:
                    df = df.reset_index()
                    x = "index"
            else:
                x = x_axis
        
        # 创建柱状图
        if orientation == "vertical":
            fig = px.bar(
                df,
                x=x,
                y=y,
                title=title,
                labels={x: x_label, y: y_label},
                height=height,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
            )
        else:
            fig = px.bar(
                df,
                y=x,
                x=y,
                title=title,
                labels={y: x_label, x: y_label},
                height=height,
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Plotly if color_palette is None else color_palette
            )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 使用Matplotlib绘图
        fig, ax = plt.subplots(figsize=(10, height/100))
        
        if isinstance(data, dict):
            # 字典数据
            if x_axis is None:
                # 简单字典，直接绘制
                categories = list(data.keys())
                values = list(data.values())
                
                # 设置颜色
                if color_palette is not None:
                    if isinstance(color_palette, str):
                        colors = sns.color_palette(color_palette, n_colors=len(categories))
                    else:
                        colors = color_palette
                else:
                    colors = None
                
                # 绘制柱状图
                if orientation == "vertical":
                    if colors is not None:
                        ax.bar(categories, values, color=colors)
                    else:
                        ax.bar(categories, values)
                else:
                    if colors is not None:
                        ax.barh(categories, values, color=colors)
                    else:
                        ax.barh(categories, values)
            else:
                # 分组柱状图
                if isinstance(x_axis, str) and x_axis in data:
                    categories = data[x_axis]
                else:
                    categories = x_axis
                
                # 计算柱宽
                n_series = len(data) - (1 if x_axis in data else 0)
                width = 0.8 / n_series
                
                # 设置颜色
                if color_palette is not None:
                    if isinstance(color_palette, str):
                        colors = sns.color_palette(color_palette, n_colors=n_series)
                    else:
                        colors = color_palette
                else:
                    colors = None
                
                # 绘制每个系列
                series_index = 0
                for key, values in data.items():
                    if key != x_axis:
                        x_positions = np.arange(len(categories)) - width * (n_series / 2) + width * (series_index + 0.5)
                        
                        if orientation == "vertical":
                            if colors is not None:
                                ax.bar(x_positions, values, width=width, label=key, color=colors[series_index % len(colors)])
                            else:
                                ax.bar(x_positions, values, width=width, label=key)
                        else:
                            if colors is not None:
                                ax.barh(x_positions, values, height=width, label=key, color=colors[series_index % len(colors)])
                            else:
                                ax.barh(x_positions, values, height=width, label=key)
                        
                        series_index += 1
                
                # 设置x轴刻度
                if orientation == "vertical":
                    ax.set_xticks(np.arange(len(categories)))
                    ax.set_xticklabels(categories)
                else:
                    ax.set_yticks(np.arange(len(categories)))
                    ax.set_yticklabels(categories)
        else:
            # DataFrame数据
            if y_axis is None:
                # 如果没有提供y_axis，使用第一个数值列
                y_axis = data.select_dtypes(include=[np.number]).columns[0]
            
            if x_axis is None:
                # 如果没有提供x_axis，使用第一个非数值列或索引
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    x_axis = non_numeric_cols[0]
                else:
                    x_axis = data.index.name if data.index.name else "index"
                    data = data.reset_index()
            
            # 绘制柱状图
            if orientation == "vertical":
                data.plot(
                    kind="bar",
                    x=x_axis,
                    y=y_axis,
                    ax=ax,
                    title=title,
                    xlabel=x_label,
                    ylabel=y_label,
                    colormap=color_palette
                )
            else:
                data.plot(
                    kind="barh",
                    x=x_axis,
                    y=y_axis,
                    ax=ax,
                    title=title,
                    xlabel=y_label,
                    ylabel=x_label,
                    colormap=color_palette
                )
        
        # 设置标题和标签
        ax.set_title(title)
        if orientation == "vertical":
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        else:
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图表
        st.pyplot(fig)

def render_scatter_plot(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    x_axis: Optional[Union[str, List[float]]] = None,
    y_axis: Optional[Union[str, List[float]]] = None,
    color_by: Optional[Union[str, List]] = None,
    size_by: Optional[Union[str, List[float]]] = None,
    title: str = "散点图",
    x_label: str = "X轴",
    y_label: str = "Y轴",
    use_plotly: bool = True,
    height: int = 400,
    color_palette: Optional[Union[str, List[str]]] = None
):
    """
    渲染散点图
    
    Args:
        data: 数据源（DataFrame或字典）
        x_axis: X轴数据列名或数据
        y_axis: Y轴数据列名或数据
        color_by: 用于着色的列名或数据
        size_by: 用于确定点大小的列名或数据
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        use_plotly: 是否使用Plotly（否则使用Matplotlib）
        height: 图表高度
        color_palette: 颜色板名称或颜色列表
    """
    if use_plotly:
        # 使用Plotly绘图
        if isinstance(data, dict):
            # 转换字典为DataFrame
            df = pd.DataFrame(data)
        else:
            df = data
        
        # 确定x和y
        if x_axis is None:
            x = df.columns[0]
        else:
            x = x_axis
        
        if y_axis is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        else:
            y = y_axis
        
        # 创建散点图
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color_by,
            size=size_by,
            title=title,
            labels={x: x_label, y: y_label},
            height=height,
            color_continuous_scale=color_palette if isinstance(color_palette, str) else None,
            color_discrete_sequence=color_palette if isinstance(color_palette, list) else None
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 使用Matplotlib绘图
        fig, ax = plt.subplots(figsize=(10, height/100))
        
        if isinstance(data, dict):
            # 转换字典为DataFrame
            df = pd.DataFrame(data)
        else:
            df = data
        
        # 确定x和y
        if x_axis is None:
            x = df.columns[0]
        else:
            x = x_axis
        
        if y_axis is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        else:
            y = y_axis
        
        # 获取数据
        x_data = df[x] if isinstance(x, str) else x
        y_data = df[y] if isinstance(y, str) else y
        
        # 设置颜色
        if color_by is not None:
            c = df[color_by] if isinstance(color_by, str) else color_by
            scatter = ax.scatter(x_data, y_data, c=c, cmap=color_palette, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by if isinstance(color_by, str) else "Color")
        else:
            ax.scatter(x_data, y_data, alpha=0.7)
        
        # 设置大小
        if size_by is not None:
            s = df[size_by] if isinstance(size_by, str) else size_by
            ax.scatter(x_data, y_data, s=s, alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图表
        st.pyplot(fig)

def render_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "热图",
    x_label: str = "X轴",
    y_label: str = "Y轴",
    annotation: bool = True,
    use_plotly: bool = True,
    height: int = 400,
    color_palette: Optional[str] = None
):
    """
    渲染热图
    
    Args:
        data: 热图数据（DataFrame或二维数组）
        x_labels: X轴标签列表
        y_labels: Y轴标签列表
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        annotation: 是否显示数值标注
        use_plotly: 是否使用Plotly（否则使用Matplotlib）
        height: 图表高度
        color_palette: 颜色板名称
    """
    # 确保数据是DataFrame
    if isinstance(data, np.ndarray):
        if x_labels is not None and y_labels is not None:
            df = pd.DataFrame(data, index=y_labels, columns=x_labels)
        else:
            df = pd.DataFrame(data)
    else:
        df = data
    
    if use_plotly:
        # 使用Plotly绘图
        fig = px.imshow(
            df,
            title=title,
            labels=dict(x=x_label, y=y_label, color="Value"),
            height=height,
            color_continuous_scale=color_palette or "Viridis"
        )
        
        # 添加数值标注
        if annotation:
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=str(round(df.iloc[i, j], 2)),
                        showarrow=False,
                        font=dict(color="white" if df.iloc[i, j] > df.values.mean() else "black")
                    )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 使用Matplotlib绘图
        fig, ax = plt.subplots(figsize=(10, height/100))
        
        # 绘制热图
        if color_palette:
            cmap = color_palette
        else:
            cmap = "viridis"
        
        sns.heatmap(
            df,
            annot=annotation,
            fmt=".2f",
            cmap=cmap,
            ax=ax
        )
        
        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # 显示图表
        st.pyplot(fig)

def render_radar_chart(
    data: Union[pd.DataFrame, Dict[str, List[float]]],
    categories: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    title: str = "雷达图",
    use_plotly: bool = True,
    height: int = 500,
    color_palette: Optional[Union[str, List[str]]] = None
):
    """
    渲染雷达图
    
    Args:
        data: 雷达图数据（DataFrame或字典）
        categories: 类别标签列表
        group_by: 用于分组的列名
        title: 图表标题
        use_plotly: 是否使用Plotly（否则使用Matplotlib）
        height: 图表高度
        color_palette: 颜色板名称或颜色列表
    """
    if use_plotly:
        # 使用Plotly绘图
        if isinstance(data, dict):
            # 如果是字典，假设格式为 {group_name: [values]}
            categories = categories or [f"Category {i+1}" for i in range(len(list(data.values())[0]))]
            
            fig = go.Figure()
            
            for i, (group, values) in enumerate(data.items()):
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=group
                ))
        else:
            # 如果是DataFrame
            if group_by is not None:
                # 按group_by列分组
                groups = data[group_by].unique()
                
                # 确定类别
                if categories is None:
                    categories = [col for col in data.columns if col != group_by]
                
                fig = go.Figure()
                
                for i, group in enumerate(groups):
                    group_data = data[data[group_by] == group]
                    values = [group_data[cat].mean() for cat in categories]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=group
                    ))
            else:
                # 没有分组，每行是一个数据点
                if categories is None:
                    categories = data.columns
                
                fig = go.Figure()
                
                for i, (_, row) in enumerate(data.iterrows()):
                    fig.add_trace(go.Scatterpolar(
                        r=row.values,
                        theta=categories,
                        fill='toself',
                        name=f"Series {i+1}"
                    ))
        
        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title=title,
            height=height
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 使用Matplotlib绘图
        fig = plt.figure(figsize=(10, height/100))
        ax = fig.add_subplot(111, polar=True)
        
        if isinstance(data, dict):
            # 如果是字典，假设格式为 {group_name: [values]}
            categories = categories or [f"Category {i+1}" for i in range(len(list(data.values())[0]))]
            
            # 计算角度
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合
            
            # 设置极轴标签
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # 设置颜色
            if color_palette is not None:
                if isinstance(color_palette, str):
                    colors = sns.color_palette(color_palette, n_colors=len(data))
                else:
                    colors = color_palette
            else:
                colors = None
            
            # 绘制每个组
            for i, (group, values) in enumerate(data.items()):
                values = np.array(values)
                values = np.append(values, values[0])  # 闭合
                
                if colors is not None:
                    ax.plot(angles, values, linewidth=2, label=group, color=colors[i % len(colors)])
                    ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                else:
                    ax.plot(angles, values, linewidth=2, label=group)
                    ax.fill(angles, values, alpha=0.25)
        else:
            # 如果是DataFrame
            if group_by is not None:
                # 按group_by列分组
                groups = data[group_by].unique()
                
                # 确定类别
                if categories is None:
                    categories = [col for col in data.columns if col != group_by]
                
                # 计算角度
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # 闭合
                
                # 设置极轴标签
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                
                # 设置颜色
                if color_palette is not None:
                    if isinstance(color_palette, str):
                        colors = sns.color_palette(color_palette, n_colors=len(groups))
                    else:
                        colors = color_palette
                else:
                    colors = None
                
                # 绘制每个组
                for i, group in enumerate(groups):
                    group_data = data[data[group_by] == group]
                    values = [group_data[cat].mean() for cat in categories]
                    values += values[:1]  # 闭合
                    
                    if colors is not None:
                        ax.plot(angles, values, linewidth=2, label=group, color=colors[i % len(colors)])
                        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                    else:
                        ax.plot(angles, values, linewidth=2, label=group)
                        ax.fill(angles, values, alpha=0.25)
            else:
                # 没有分组，每行是一个数据点
                if categories is None:
                    categories = data.columns
                
                # 计算角度
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # 闭合
                
                # 设置极轴标签
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                
                # 设置颜色
                if color_palette is not None:
                    if isinstance(color_palette, str):
                        colors = sns.color_palette(color_palette, n_colors=len(data))
                    else:
                        colors = color_palette
                else:
                    colors = None
                
                # 绘制每行
                for i, (_, row) in enumerate(data.iterrows()):
                    values = row.values
                    values = np.append(values, values[0])  # 闭合
                    
                    if colors is not None:
                        ax.plot(angles, values, linewidth=2, label=f"Series {i+1}", color=colors[i % len(colors)])
                        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                    else:
                        ax.plot(angles, values, linewidth=2, label=f"Series {i+1}")
                        ax.fill(angles, values, alpha=0.25)
        
        # 设置标题和图例
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 显示图表
        st.pyplot(fig)

def render_architecture_diagram(model_config):
    """
    渲染模型架构图
    
    Args:
        model_config: 模型配置字典
    """
    try:
        import graphviz
        
        # 创建有向图
        dot = graphviz.Digraph(comment="H-CAAN Model Architecture")
        
        # 设置图形属性
        dot.attr(rankdir="LR", size="12,8")
        
        # 定义节点样式
        dot.attr("node", shape="box", style="filled", color="lightblue", fontname="Arial")
        
        # 添加输入节点
        with dot.subgraph(name="cluster_input") as c:
            c.attr(label="Input Layer", style="filled", color="lightgrey")
            c.node("smiles_input", "SMILES Input")
            c.node("ecfp_input", "ECFP Input")
            c.node("graph_input", "Graph Input")
            if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                c.node("mfbert_input", "MFBERT Input")
        
        # 添加编码器节点
        with dot.subgraph(name="cluster_encoders") as c:
            c.attr(label="Encoder Layer", style="filled", color="lightgrey")
            
            # SMILES编码器
            smiles_config = model_config.get("smiles_encoder", {})
            c.node("smiles_encoder", f"Transformer Encoder\nLayers: {smiles_config.get('num_layers', 3)}\nHeads: {smiles_config.get('num_heads', 8)}\nDim: {smiles_config.get('hidden_dim', 256)}")
            
            # ECFP编码器
            ecfp_config = model_config.get("ecfp_encoder", {})
            c.node("ecfp_encoder", f"BiGRU Encoder\nLayers: {ecfp_config.get('num_layers', 2)}\nDim: {ecfp_config.get('hidden_dim', 256)}")
            
            # 图编码器
            gcn_config = model_config.get("gcn_encoder", {})
            c.node("gcn_encoder", f"GCN Encoder\nLayers: {gcn_config.get('num_layers', 3)}\nDim: {gcn_config.get('hidden_dim', 256)}")
            
            # MFBERT编码器
            if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                mfbert_config = model_config.get("mfbert_encoder", {})
                c.node("mfbert_encoder", f"MFBERT Encoder\nDim: {mfbert_config.get('hidden_dim', 512)}")
        
        # 添加融合层节点
        with dot.subgraph(name="cluster_fusion") as c:
            c.attr(label="Fusion Layer", style="filled", color="lightgrey")
            
            # 融合配置
            fusion_config = model_config.get("fusion", {})
            fusion_levels = fusion_config.get("levels", [])
            
            # 判断是否有低级融合
            if "低级（特征层）" in fusion_levels or "Low-level (Feature)" in fusion_levels:
                c.node("low_fusion", "Low-level Fusion\n(Feature Level)")
            
            # 判断是否有中级融合
            if "中级（语义层）" in fusion_levels or "Mid-level (Semantic)" in fusion_levels:
                c.node("mid_fusion", "Mid-level Fusion\n(Semantic Level)")
            
            # 判断是否有高级融合
            if "高级（决策层）" in fusion_levels or "High-level (Decision)" in fusion_levels:
                c.node("high_fusion", "High-level Fusion\n(Decision Level)")
            
            # GCAU节点
            if fusion_config.get("use_adaptive_gating", False):
                c.node("gcau", "Gated Cross-modal\nAttention Unit (GCAU)")
            
            # 最终融合节点
            c.node("final_fusion", "Hierarchical Fusion")
        
        # 添加输出层节点
        with dot.subgraph(name="cluster_output") as c:
            c.attr(label="Output Layer", style="filled", color="lightgrey")
            
            # 模态重要性
            importance_config = model_config.get("modal_importance", {})
            if importance_config.get("use_task_specific", False):
                c.node("importance", "Task-specific\nWeight Generation")
            
            # 输出节点
            c.node("output", f"Property Prediction\nDim: {model_config.get('general', {}).get('output_dim', 128)}")
        
        # 添加边
        # 输入到编码器
        dot.edge("smiles_input", "smiles_encoder")
        dot.edge("ecfp_input", "ecfp_encoder")
        dot.edge("graph_input", "gcn_encoder")
        if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
            dot.edge("mfbert_input", "mfbert_encoder")
        
        # 编码器到融合层
        fusion_config = model_config.get("fusion", {})
        fusion_levels = fusion_config.get("levels", [])
        
        # 根据融合级别添加边
        if "低级（特征层）" in fusion_levels or "Low-level (Feature)" in fusion_levels:
            dot.edge("smiles_encoder", "low_fusion")
            dot.edge("ecfp_encoder", "low_fusion")
            dot.edge("gcn_encoder", "low_fusion")
            if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                dot.edge("mfbert_encoder", "low_fusion")
            
            if fusion_config.get("use_adaptive_gating", False):
                dot.edge("low_fusion", "gcau")
                next_level = "gcau"
            else:
                next_level = "low_fusion"
        else:
            next_level = None
        
        if "中级（语义层）" in fusion_levels or "Mid-level (Semantic)" in fusion_levels:
            if next_level:
                dot.edge(next_level, "mid_fusion")
            else:
                dot.edge("smiles_encoder", "mid_fusion")
                dot.edge("ecfp_encoder", "mid_fusion")
                dot.edge("gcn_encoder", "mid_fusion")
                if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                    dot.edge("mfbert_encoder", "mid_fusion")
            
            next_level = "mid_fusion"
        
        if "高级（决策层）" in fusion_levels or "High-level (Decision)" in fusion_levels:
            if next_level:
                dot.edge(next_level, "high_fusion")
            else:
                dot.edge("smiles_encoder", "high_fusion")
                dot.edge("ecfp_encoder", "high_fusion")
                dot.edge("gcn_encoder", "high_fusion")
                if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                    dot.edge("mfbert_encoder", "high_fusion")
            
            next_level = "high_fusion"
        
        # 连接到最终融合
        if next_level and next_level != "final_fusion":
            dot.edge(next_level, "final_fusion")
        elif not next_level:
            dot.edge("smiles_encoder", "final_fusion")
            dot.edge("ecfp_encoder", "final_fusion")
            dot.edge("gcn_encoder", "final_fusion")
            if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
                dot.edge("mfbert_encoder", "final_fusion")
        
        # 融合到输出
        importance_config = model_config.get("modal_importance", {})
        if importance_config.get("use_task_specific", False):
            dot.edge("final_fusion", "importance")
            dot.edge("importance", "output")
        else:
            dot.edge("final_fusion", "output")
        
        # 显示图形
        st.graphviz_chart(dot)
    
    except Exception as e:
        st.error(f"无法生成架构图: {str(e)}")
        
        # 使用文本替代
        st.markdown("""
        ## H-CAAN架构描述
        
        **1. 输入层**:
        - SMILES编码向量
        - ECFP指纹
        - 分子图
        """)
        
        if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
            st.markdown("- MFBERT嵌入")
        
        st.markdown("""
        **2. 编码器层**:
        - SMILES编码器 (Transformer): 处理SMILES编码向量
        - ECFP编码器 (BiGRU): 处理ECFP指纹
        - 图编码器 (GCN): 处理分子图
        """)
        
        if model_config.get("mfbert_encoder", {}).get("use_mfbert", False):
            st.markdown("- MFBERT编码器: 处理MFBERT嵌入")
        
        st.markdown("**3. 融合层**:")
        
        fusion_config = model_config.get("fusion", {})
        fusion_levels = fusion_config.get("levels", [])
        
        if "低级（特征层）" in fusion_levels or "Low-level (Feature)" in fusion_levels:
            st.markdown("- 低级融合 (特征层)")
        
        if "中级（语义层）" in fusion_levels or "Mid-level (Semantic)" in fusion_levels:
            st.markdown("- 中级融合 (语义层)")
        
        if "高级（决策层）" in fusion_levels or "High-level (Decision)" in fusion_levels:
            st.markdown("- 高级融合 (决策层)")
        
        if fusion_config.get("use_adaptive_gating", False):
            st.markdown("- 门控交叉模态注意力单元 (GCAU)")
        
        st.markdown("- 层次化融合")
        
        st.markdown("**4. 输出层**:")
        
        importance_config = model_config.get("modal_importance", {})
        if importance_config.get("use_task_specific", False):
            st.markdown("- 任务特定权重生成")
        
        output_dim = model_config.get("general", {}).get("output_dim", 128)
        st.markdown(f"- 性质预测 (维度: {output_dim})")

if __name__ == "__main__":
    # 测试代码
    st.title("分子数据可视化组件测试")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["分子属性趋势图", "分子分类统计图", "分子特征散点图", "相似度热图", "分子特征雷达图"])
    
    with tab1:
        st.subheader("分子属性趋势图示例")
        
        # 生成分子属性示例数据
        molecules = [f"分子{i+1}" for i in range(10)]
        data = {
            'molecule': molecules,
            'logP': [2.3, 1.8, 3.2, 4.1, 2.7, 3.5, 1.9, 2.8, 3.9, 3.0],
            'MW': [250, 280, 310, 340, 270, 320, 290, 305, 330, 275],
            'HBD': [2, 3, 1, 0, 2, 1, 3, 2, 1, 2]
        }
        
        # 使用Plotly渲染
        render_line_chart(
            data=data,
            x_axis='molecule',
            y_axis=['logP', 'MW', 'HBD'],
            title="分子属性比较",
            x_label="分子",
            y_label="属性值",
            use_plotly=True,
            color_palette="Blues"
        )
        
        # 使用Matplotlib渲染
        render_line_chart(
            data=data,
            x_axis='molecule',
            y_axis=['logP', 'HBD'],
            title="药物性质分布 (Matplotlib)",
            x_label="分子",
            y_label="属性值",
            use_plotly=False,
            color_palette="viridis"
        )
    
    with tab2:
        st.subheader("分子分类统计图示例")
        
        # 生成分子分类示例数据
        categories = ['抗菌药物', '抗病毒药物', '抗肿瘤药物', '中枢神经系统药物']
        data = {
            'category': categories,
            '活性分子数': [45, 60, 32, 51],
            '临床分子数': [25, 30, 42, 35]
        }
        
        # 使用Plotly渲染垂直柱状图
        render_bar_chart(
            data=data,
            x_axis='category',
            y_axis=['活性分子数', '临床分子数'],
            title="药物分类统计",
            x_label="药物类别",
            y_label="分子数量",
            use_plotly=True
        )
        
        # 使用Matplotlib渲染水平柱状图
        render_bar_chart(
            data=data,
            x_axis='category',
            y_axis=['活性分子数'],
            title="活性分子分布 (Matplotlib)",
            x_label="药物类别",
            y_label="分子数量",
            use_plotly=False,
            orientation="horizontal"
        )
    
    with tab3:
        st.subheader("分子特征散点图示例")
        
        # 生成分子特征示例数据
        np.random.seed(42)
        n = 50
        data = {
            'logP': np.random.uniform(0, 5, n),
            'TPSA': np.random.uniform(50, 150, n),
            'MW': np.random.uniform(200, 500, n),
            'activity_class': np.random.choice(['高活性', '中等活性', '低活性'], n)
        }
        
        # 使用Plotly渲染
        render_scatter_plot(
            data=data,
            x_axis='logP',
            y_axis='TPSA',
            color_by='activity_class',
            size_by='MW',
            title="分子特性与活性关系",
            x_label="LogP (脂溶性)",
            y_label="TPSA (极性表面积)",
            use_plotly=True
        )
        
        # 使用Matplotlib渲染
        render_scatter_plot(
            data=data,
            x_axis='logP',
            y_axis='TPSA',
            color_by='activity_class',
            title="活性分子聚类分布 (Matplotlib)",
            x_label="LogP (脂溶性)",
            y_label="TPSA (极性表面积)",
            use_plotly=False,
            color_palette="viridis"
        )
    
    with tab4:
        st.subheader("分子相似度热图示例")
        
        # 生成分子相似度示例数据
        molecules = ['化合物A', '化合物B', '化合物C', '化合物D', '化合物E']
        # 生成对称矩阵作为相似度矩阵
        np.random.seed(42)
        sim_matrix = np.random.rand(5, 5)
        # 确保对角线为1(自身相似度)
        np.fill_diagonal(sim_matrix, 1)
        # 确保矩阵对称
        sim_matrix = (sim_matrix + sim_matrix.T) / 2
        
        # 使用Plotly渲染
        render_heatmap(
            data=sim_matrix,
            x_labels=molecules,
            y_labels=molecules,
            title="分子相似度矩阵",
            x_label="分子",
            y_label="分子",
            use_plotly=True,
            color_palette="Viridis"
        )
        
        # 使用Matplotlib渲染
        render_heatmap(
            data=sim_matrix,
            x_labels=molecules,
            y_labels=molecules,
            title="分子相似度矩阵 (Matplotlib)",
            x_label="分子",
            y_label="分子",
            use_plotly=False,
            color_palette="coolwarm"
        )
    
    with tab5:
        st.subheader("分子特征雷达图示例")
        
        # 生成分子特征雷达图示例数据
        features = ['脂溶性', '水溶性', '血脑屏障穿透性', '肝毒性', '蛋白结合率']
        data = {
            '候选药物A': [0.8, 0.3, 0.9, 0.2, 0.7],
            '候选药物B': [0.4, 0.7, 0.5, 0.3, 0.6],
            '候选药物C': [0.6, 0.5, 0.4, 0.1, 0.8]
        }
        
        # 使用Plotly渲染
        render_radar_chart(
            data=data,
            categories=features,
            title="候选药物特征比较",
            use_plotly=True
        )
        
        # 使用Matplotlib渲染
        render_radar_chart(
            data=data,
            categories=features,
            title="候选药物特征比较 (Matplotlib)",
            use_plotly=False,
            color_palette="plasma"
        )
    
    # 模型架构图示例
    st.subheader("H-CAAN模型架构图示例")
    
    # 示例模型配置
    example_model_config = {
        "general": {
            "output_dim": 128
        },
        "smiles_encoder": {
            "num_layers": 3,
            "num_heads": 8,
            "hidden_dim": 256
        },
        "ecfp_encoder": {
            "num_layers": 2,
            "hidden_dim": 256
        },
        "gcn_encoder": {
            "num_layers": 3,
            "hidden_dim": 256
        },
        "mfbert_encoder": {
            "use_mfbert": True,
            "hidden_dim": 512
        },
        "fusion": {
            "levels": ["低级（特征层）", "中级（语义层）", "高级（决策层）"],
            "use_adaptive_gating": True
        },
        "modal_importance": {
            "use_task_specific": True
        }
    }
    
    render_architecture_diagram(example_model_config)