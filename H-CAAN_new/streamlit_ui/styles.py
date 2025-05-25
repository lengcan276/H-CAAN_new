"""
CSS样式和主题定制模块，为H-CAAN项目提供统一的视觉风格。
包含全局样式、颜色方案、字体设置、组件样式和主题定制功能。
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Any

# 基本颜色方案
PRIMARY_COLOR = "#4A90E2"  # 主色调：蓝色
SECONDARY_COLOR = "#50E3C2"  # 辅助色：青绿色
ACCENT_COLOR = "#FF6B6B"  # 强调色：珊瑚色
BACKGROUND_COLOR = "#F5F7FA"  # 背景色：浅灰色
TEXT_COLOR = "#333333"  # 文本色：深灰色
DARK_TEXT_COLOR = "#1A1A1A"  # 深色文本：近黑色
LIGHT_TEXT_COLOR = "#777777"  # 浅色文本：中灰色

# 功能性颜色
SUCCESS_COLOR = "#4CAF50"  # 成功色：绿色
WARNING_COLOR = "#FFC107"  # 警告色：琥珀色
ERROR_COLOR = "#F44336"  # 错误色：红色
INFO_COLOR = "#2196F3"  # 信息色：蓝色

# 分子属性表示颜色
MOLECULE_COLORS = {
    "Hydrophobic": "#FFD700",  # 疏水区域：金色
    "Hydrophilic": "#1E90FF",  # 亲水区域：道奇蓝
    "Aromatic": "#FF8C00",  # 芳香区域：深橙色
    "H-Bond Donor": "#FF69B4",  # 氢键供体：粉红色
    "H-Bond Acceptor": "#00CED1",  # 氢键受体：深青色
    "Positive Charge": "#DC143C",  # 正电荷：猩红色
    "Negative Charge": "#6495ED"  # 负电荷：矢车菊蓝
}

# 图表颜色方案
CHART_COLORS = [
    "#4A90E2",  # 蓝色
    "#50E3C2",  # 青绿色
    "#FF6B6B",  # 珊瑚色
    "#FFD700",  # 金色
    "#9D65C9",  # 紫色
    "#FF8C00",  # 深橙色
    "#1ABC9C",  # 绿松石色
    "#F39C12",  # 橙色
    "#E74C3C",  # 红色
    "#3498DB"   # 天蓝色
]

# 工作流程状态颜色
WORKFLOW_STATUS_COLORS = {
    "pending": "#CCCCCC",      # 灰色
    "in_progress": "#FFC107",  # 琥珀色
    "completed": "#4CAF50",    # 绿色
    "error": "#F44336"         # 红色
}

class StyleManager:
    """管理和应用应用程序样式的类"""
    
    def __init__(self):
        """初始化风格管理器"""
        self.dark_mode = False
        self.custom_theme = None
    
    def apply_base_styles(self):
        """应用基本样式到Streamlit应用程序"""
        # 基本CSS样式
        st.markdown(self._get_base_css(), unsafe_allow_html=True)
        
        # 应用自定义主题（如果有）
        if self.custom_theme:
            st.markdown(self._generate_theme_css(self.custom_theme), unsafe_allow_html=True)
    
    def apply_component_styles(self, component_type: str):
        """
        应用特定组件样式
        
        Args:
            component_type: 组件类型名称（'header', 'sidebar', 'workflow', 'molecule_viewer', 等）
        """
        # 获取组件特定的CSS
        css_method = getattr(self, f"_get_{component_type}_css", None)
        if css_method:
            st.markdown(css_method(), unsafe_allow_html=True)
    
    def apply_chart_theme(self, chart_type: str) -> Dict[str, Any]:
        """
        获取图表的样式配置
        
        Args:
            chart_type: 图表类型（'plotly', 'matplotlib', 'altair', 等）
            
        Returns:
            图表样式配置字典
        """
        if chart_type == 'plotly':
            return self._get_plotly_theme()
        elif chart_type == 'matplotlib':
            return self._get_matplotlib_theme()
        elif chart_type == 'altair':
            return self._get_altair_theme()
        return {}
    
    def toggle_dark_mode(self, enable: bool = None):
        """
        切换深色模式
        
        Args:
            enable: 若提供，设置深色模式状态；否则切换当前状态
        """
        if enable is not None:
            self.dark_mode = enable
        else:
            self.dark_mode = not self.dark_mode
            
        if self.dark_mode:
            st.markdown(self._get_dark_mode_css(), unsafe_allow_html=True)
        else:
            # 重新应用基础样式
            self.apply_base_styles()
    
    def set_custom_theme(self, theme: Dict[str, str]):
        """
        设置自定义主题
        
        Args:
            theme: 主题颜色字典
        """
        self.custom_theme = theme
        st.markdown(self._generate_theme_css(theme), unsafe_allow_html=True)
    
    def get_color_scheme(self) -> Dict[str, str]:
        """
        获取当前颜色方案
        
        Returns:
            颜色方案字典
        """
        if self.dark_mode:
            return {
                "primary": PRIMARY_COLOR,
                "secondary": SECONDARY_COLOR,
                "accent": ACCENT_COLOR,
                "background": "#1E1E1E",  # 深色背景
                "text": "#FFFFFF",  # 白色文本
                "light_text": "#CCCCCC",  # 浅灰色文本
                "success": SUCCESS_COLOR,
                "warning": WARNING_COLOR,
                "error": ERROR_COLOR,
                "info": INFO_COLOR
            }
        else:
            return {
                "primary": PRIMARY_COLOR,
                "secondary": SECONDARY_COLOR,
                "accent": ACCENT_COLOR,
                "background": BACKGROUND_COLOR,
                "text": TEXT_COLOR,
                "light_text": LIGHT_TEXT_COLOR,
                "success": SUCCESS_COLOR,
                "warning": WARNING_COLOR,
                "error": ERROR_COLOR,
                "info": INFO_COLOR
            }
    
    def get_status_color(self, status: str) -> str:
        """
        获取状态颜色
        
        Args:
            status: 状态名称
            
        Returns:
            颜色代码
        """
        return WORKFLOW_STATUS_COLORS.get(status, "#CCCCCC")  # 默认灰色
    
    def get_chart_color_sequence(self) -> List[str]:
        """
        获取图表颜色序列
        
        Returns:
            颜色代码列表
        """
        return CHART_COLORS
    
    def get_molecule_color(self, property_type: str) -> str:
        """
        获取分子属性颜色
        
        Args:
            property_type: 属性类型
            
        Returns:
            颜色代码
        """
        return MOLECULE_COLORS.get(property_type, "#777777")  # 默认中灰色
    
    def _get_base_css(self) -> str:
        """获取基本CSS样式"""
        return f"""
        <style>
            /* 全局样式 */
            * {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            /* 标题样式 */
            h1, h2, h3, h4, h5, h6 {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 600;
                color: {TEXT_COLOR};
            }}
            
            h1 {{
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                color: {PRIMARY_COLOR};
            }}
            
            h2 {{
                font-size: 1.8rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #EEEEEE;
            }}
            
            h3 {{
                font-size: 1.5rem;
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
            }}
            
            /* 段落和文本样式 */
            p {{
                font-size: 1rem;
                line-height: 1.6;
                color: {TEXT_COLOR};
                margin-bottom: 1rem;
            }}
            
            a {{
                color: {PRIMARY_COLOR};
                text-decoration: none;
            }}
            
            a:hover {{
                text-decoration: underline;
            }}
            
            code {{
                font-family: 'Fira Code', monospace;
                padding: 0.2rem 0.4rem;
                font-size: 0.85rem;
                background-color: #F0F2F5;
                border-radius: 4px;
            }}
            
            pre {{
                background-color: #F0F2F5;
                padding: 1rem;
                border-radius: 6px;
                overflow-x: auto;
            }}
            
            /* 按钮样式覆盖 */
            .stButton>button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            
            .stButton>button:hover {{
                background-color: {PRIMARY_COLOR}DD;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            
            /* 卡片样式 */
            .card {{
                background-color: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 1.5rem;
            }}
            
            /* 数据表格样式 */
            .dataframe {{
                border-collapse: collapse;
                width: 100%;
                font-size: 0.9rem;
            }}
            
            .dataframe th {{
                background-color: #F0F2F5;
                padding: 0.75rem;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #DDDDDD;
            }}
            
            .dataframe td {{
                padding: 0.75rem;
                border-bottom: 1px solid #EEEEEE;
            }}
            
            .dataframe tr:hover {{
                background-color: #F5F7FA;
            }}
            
            /* 工作流程步骤样式 */
            .workflow-step {{
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
                padding: 0.75rem 1rem;
                border-radius: 6px;
                background-color: #F0F2F5;
            }}
            
            .workflow-step.active {{
                background-color: {PRIMARY_COLOR}22;
                border-left: 4px solid {PRIMARY_COLOR};
            }}
            
            .workflow-step .step-number {{
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background-color: #DDDDDD;
                color: #555555;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                margin-right: 1rem;
            }}
            
            .workflow-step.active .step-number {{
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            
            .workflow-step .step-content {{
                flex: 1;
            }}
            
            .workflow-step .step-title {{
                font-weight: 600;
                margin-bottom: 0.25rem;
            }}
            
            .workflow-step .step-description {{
                font-size: 0.85rem;
                color: {LIGHT_TEXT_COLOR};
            }}
            
            /* 状态标签样式 */
            .status-label {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }}
            
            .status-label.pending {{
                background-color: #EEEEEE;
                color: #555555;
            }}
            
            .status-label.in-progress {{
                background-color: {WARNING_COLOR}33;
                color: {WARNING_COLOR};
            }}
            
            .status-label.completed {{
                background-color: {SUCCESS_COLOR}33;
                color: {SUCCESS_COLOR};
            }}
            
            .status-label.error {{
                background-color: {ERROR_COLOR}33;
                color: {ERROR_COLOR};
            }}
            
            /* 分割线样式 */
            hr {{
                border: 0;
                height: 1px;
                background-color: #EEEEEE;
                margin: 2rem 0;
            }}
            
            /* 可折叠面板样式 */
            details {{
                margin-bottom: 1rem;
                padding: 0.75rem;
                border-radius: 6px;
                background-color: #F5F7FA;
            }}
            
            details summary {{
                font-weight: 600;
                cursor: pointer;
                padding: 0.5rem;
            }}
            
            details[open] summary {{
                margin-bottom: 0.75rem;
                border-bottom: 1px solid #EEEEEE;
            }}
            
            /* 自定义滚动条 */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: #F5F7FA;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: #CCCCCC;
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: #AAAAAA;
            }}
            
            /* 工具提示样式 */
            .tooltip {{
                position: relative;
                display: inline-block;
            }}
            
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 200px;
                background-color: #333333;
                color: white;
                text-align: center;
                border-radius: 6px;
                padding: 0.5rem;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 0.85rem;
            }}
            
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
            
            /* 标签样式 */
            .tag {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                background-color: #F0F2F5;
                color: {TEXT_COLOR};
            }}
            
            /* 分子可视化容器样式 */
            .molecule-viewer {{
                background-color: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 1.5rem;
            }}
            
            .molecule-viewer-controls {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #EEEEEE;
            }}
            
            /* 分子属性标签样式 */
            .property-tag {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            
            .property-tag.hydrophobic {{
                background-color: {MOLECULE_COLORS["Hydrophobic"]}33;
                color: {MOLECULE_COLORS["Hydrophobic"]};
            }}
            
            .property-tag.hydrophilic {{
                background-color: {MOLECULE_COLORS["Hydrophilic"]}33;
                color: {MOLECULE_COLORS["Hydrophilic"]};
            }}
            
            .property-tag.aromatic {{
                background-color: {MOLECULE_COLORS["Aromatic"]}33;
                color: {MOLECULE_COLORS["Aromatic"]};
            }}
            
            .property-tag.h-donor {{
                background-color: {MOLECULE_COLORS["H-Bond Donor"]}33;
                color: {MOLECULE_COLORS["H-Bond Donor"]};
            }}
            
            .property-tag.h-acceptor {{
                background-color: {MOLECULE_COLORS["H-Bond Acceptor"]}33;
                color: {MOLECULE_COLORS["H-Bond Acceptor"]};
            }}
            
            .property-tag.pos-charge {{
                background-color: {MOLECULE_COLORS["Positive Charge"]}33;
                color: {MOLECULE_COLORS["Positive Charge"]};
            }}
            
            .property-tag.neg-charge {{
                background-color: {MOLECULE_COLORS["Negative Charge"]}33;
                color: {MOLECULE_COLORS["Negative Charge"]};
            }}
            
            /* 自定义Streamlit UI样式覆盖 */
            .stRadio > div {{
                flex-direction: row;
                gap: 1rem;
            }}
            
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                padding: 0.5rem 1rem;
                border-radius: 4px 4px 0 0;
            }}
            
            /* 页眉和页脚样式 */
            .header {{
                padding: 1rem 0;
                border-bottom: 1px solid #EEEEEE;
                margin-bottom: 2rem;
            }}
            
            .footer {{
                margin-top: 3rem;
                padding-top: 1.5rem;
                border-top: 1px solid #EEEEEE;
                text-align: center;
                font-size: 0.85rem;
                color: {LIGHT_TEXT_COLOR};
            }}
        </style>
        """
    
    def _get_dark_mode_css(self) -> str:
        """获取深色模式CSS样式"""
        return f"""
        <style>
            /* 深色模式全局样式覆盖 */
            * {{
                color: #FFFFFF;
            }}
            
            body {{
                background-color: #1E1E1E;
            }}
            
            .stApp {{
                background-color: #1E1E1E;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                color: #FFFFFF;
            }}
            
            h1 {{
                color: {PRIMARY_COLOR};
            }}
            
            h2 {{
                border-bottom: 1px solid #444444;
            }}
            
            p {{
                color: #EEEEEE;
            }}
            
            a {{
                color: {SECONDARY_COLOR};
            }}
            
            code {{
                background-color: #333333;
                color: #EEEEEE;
            }}
            
            pre {{
                background-color: #333333;
            }}
            
            /* 卡片深色模式样式 */
            .card {{
                background-color: #2A2A2A;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }}
            
            /* 数据表格深色模式样式 */
            .dataframe th {{
                background-color: #333333;
                border-bottom: 2px solid #444444;
            }}
            
            .dataframe td {{
                border-bottom: 1px solid #444444;
            }}
            
            .dataframe tr:hover {{
                background-color: #333333;
            }}
            
            /* 工作流程步骤深色模式样式 */
            .workflow-step {{
                background-color: #333333;
            }}
            
            .workflow-step.active {{
                background-color: {PRIMARY_COLOR}33;
            }}
            
            .workflow-step .step-number {{
                background-color: #555555;
                color: #EEEEEE;
            }}
            
            .workflow-step .step-description {{
                color: #AAAAAA;
            }}
            
            /* 状态标签深色模式样式 */
            .status-label.pending {{
                background-color: #444444;
                color: #AAAAAA;
            }}
            
            /* 分割线深色模式样式 */
            hr {{
                background-color: #444444;
            }}
            
            /* 可折叠面板深色模式样式 */
            details {{
                background-color: #2A2A2A;
            }}
            
            details[open] summary {{
                border-bottom: 1px solid #444444;
            }}
            
            /* 自定义滚动条深色模式样式 */
            ::-webkit-scrollbar-track {{
                background: #1E1E1E;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: #555555;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: #777777;
            }}
            
            /* 标签深色模式样式 */
            .tag {{
                background-color: #333333;
                color: #EEEEEE;
            }}
            
            /* 分子可视化容器深色模式样式 */
            .molecule-viewer {{
                background-color: #2A2A2A;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }}
            
            .molecule-viewer-controls {{
                border-bottom: 1px solid #444444;
            }}
            
            /* 自定义Streamlit UI深色模式样式覆盖 */
            .stTabs [data-baseweb="tab-list"] {{
                background-color: #1E1E1E;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: #2A2A2A;
            }}
            
            /* 页眉和页脚深色模式样式 */
            .header {{
                border-bottom: 1px solid #444444;
            }}
            
            .footer {{
                border-top: 1px solid #444444;
                color: #AAAAAA;
            }}
        </style>
        """
    
    def _get_header_css(self) -> str:
        """获取页眉组件CSS样式"""
        return """
        <style>
            /* 页眉组件样式 */
            .h-caan-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1rem 0;
                margin-bottom: 2rem;
                border-bottom: 1px solid #EEEEEE;
            }
            
            .h-caan-header .logo-section {
                display: flex;
                align-items: center;
            }
            
            .h-caan-header .logo {
                height: 40px;
                margin-right: 1rem;
            }
            
            .h-caan-header .title {
                font-size: 1.5rem;
                font-weight: 700;
                margin: 0;
            }
            
            .h-caan-header .navigation {
                display: flex;
                gap: 1.5rem;
            }
            
            .h-caan-header .nav-item {
                font-size: 0.9rem;
                font-weight: 500;
                color: #555555;
                text-decoration: none;
                padding: 0.5rem 0;
                position: relative;
            }
            
            .h-caan-header .nav-item:hover {
                color: #333333;
            }
            
            .h-caan-header .nav-item.active {
                color: #4A90E2;
            }
            
            .h-caan-header .nav-item.active::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 2px;
                background-color: #4A90E2;
            }
            
            /* 响应式设计 */
            @media (max-width: 768px) {
                .h-caan-header {
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .h-caan-header .navigation {
                    width: 100%;
                    justify-content: space-between;
                }
            }
        </style>
        """
    
    def _get_sidebar_css(self) -> str:
        """获取侧边栏组件CSS样式"""
        return """
        <style>
            /* 侧边栏组件样式 */
            [data-testid="stSidebar"] {
                background-color: #F5F7FA;
                border-right: 1px solid #EEEEEE;
            }
            
            .sidebar-header {
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #EEEEEE;
            }
            
            .sidebar-section {
                margin-bottom: 2rem;
            }
            
            .sidebar-section-title {
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #777777;
                margin-bottom: 0.75rem;
            }
            
            .sidebar-menu-item {
                display: flex;
                align-items: center;
                padding: 0.5rem 0.75rem;
                margin-bottom: 0.25rem;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }
            
            .sidebar-menu-item:hover {
                background-color: #EEEEEE;
            }
            
            .sidebar-menu-item.active {
                background-color: #4A90E233;
                color: #4A90E2;
                font-weight: 500;
            }
            
            .sidebar-menu-item-icon {
                margin-right: 0.75rem;
                width: 20px;
                text-align: center;
            }
            
            .sidebar-meta {
                font-size: 0.75rem;
                color: #999999;
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid #EEEEEE;
            }
            
            .sidebar-bottom {
                position: absolute;
                bottom: 1rem;
                left: 1rem;
                right: 1rem;
                font-size: 0.75rem;
                color: #999999;
                padding-top: 1rem;
                border-top: 1px solid #EEEEEE;
            }
            
            /* 深色模式侧边栏覆盖 */
            .dark-mode [data-testid="stSidebar"] {
                background-color: #1A1A1A;
                border-right: 1px solid #333333;
            }
            
            .dark-mode .sidebar-header {
                border-bottom: 1px solid #333333;
            }
            
            .dark-mode .sidebar-section-title {
                color: #AAAAAA;
            }
            
            .dark-mode .sidebar-menu-item:hover {
                background-color: #333333;
            }
            
            .dark-mode .sidebar-meta,
            .dark-mode .sidebar-bottom {
                color: #777777;
                border-top: 1px solid #333333;
            }
        </style>
        """
    
    def _get_workflow_status_css(self) -> str:
        """获取工作流程状态组件CSS样式"""
        return f"""
        <style>
            /* 工作流程状态组件样式 */
            .workflow-status-container {{
                margin-bottom: 2rem;
                background-color: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }}
            
            .workflow-status-title {{
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
            }}
            
            .workflow-steps {{
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}
            
            .workflow-step-item {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .workflow-step-indicator {{
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 0.85rem;
                flex-shrink: 0;
            }}
            
            /* 步骤指示器状态样式 */
            .workflow-step-indicator.pending {{
                background-color: #EEEEEE;
                color: #777777;
            }}
            
            .workflow-step-indicator.in_progress {{
                background-color: {WARNING_COLOR};
                color: white;
            }}
            
            .workflow-step-indicator.completed {{
                background-color: {SUCCESS_COLOR};
                color: white;
            }}
            
            .workflow-step-indicator.error {{
                background-color: {ERROR_COLOR};
                color: white;
            }}
            
            .workflow-step-content {{
                flex: 1;
            }}
            
            .workflow-step-title {{
                font-weight: 500;
                margin-bottom: 0.25rem;
            }}
            
            .workflow-step-info {{
                font-size: 0.85rem;
                color: #777777;
            }}
            
            .workflow-progress {{
                height: 6px;
                background-color: #EEEEEE;
                border-radius: 3px;
                margin: 1.5rem 0;
                overflow: hidden;
            }}
            
            .workflow-progress-bar {{
                height: 100%;
                background-color: {PRIMARY_COLOR};
                transition: width 0.3s ease;
            }}
            
            .workflow-status-actions {{
                display: flex;
                justify-content: flex-end;
                margin-top: 1.5rem;
                gap: 1rem;
            }}
            
            /* 连接线样式 */
            .workflow-step-connector {{
                width: 2px;
                height: 30px;
                background-color: #EEEEEE;
                margin-left: 14px;
            }}
            
            .workflow-step-connector.completed {{
                background-color: {SUCCESS_COLOR};
            }}
            
            .workflow-step-connector.in_progress {{
                background-color: {WARNING_COLOR};
            }}
            
            /* 深色模式覆盖 */
            .dark-mode .workflow-status-container {{
                background-color: #2A2A2A;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }}
            
            .dark-mode .workflow-step-indicator.pending {{
                background-color: #444444;
                color: #AAAAAA;
            }}
            
            .dark-mode .workflow-step-info {{
                color: #AAAAAA;
            }}
            
            .dark-mode .workflow-progress {{
                background-color: #444444;
            }}
            
            .dark-mode .workflow-step-connector {{
                background-color: #444444;
            }}
        </style>
        """
    
    def _get_molecule_viewer_css(self) -> str:
        """获取分子查看器组件CSS样式"""
        return """
        <style>
            /* 分子查看器组件样式 */
            .molecule-container {
                width: 100%;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 2rem;
                overflow: hidden;
            }
            
            .molecule-header {
                padding: 1rem;
                border-bottom: 1px solid #EEEEEE;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .molecule-title {
                font-weight: 600;
                font-size: 1.1rem;
                margin: 0;
            }
            
            .molecule-controls {
                display: flex;
                gap: 0.5rem;
            }
            
            .molecule-content {
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .molecule-viewer-window {
                width: 100%;
                height: 400px;
                position: relative;
                border: 1px solid #EEEEEE;
                border-radius: 6px;
                overflow: hidden;
            }
            
            .molecule-info {
                padding: 1rem;
                background-color: #F5F7FA;
                border-radius: 6px;
            }
            
            .molecule-properties {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .molecule-property {
                display: flex;
                flex-direction: column;
                background-color: white;
                padding: 0.75rem;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                min-width: 120px;
            }
            
            .property-label {
                font-size: 0.75rem;
                color: #777777;
                margin-bottom: 0.25rem;
            }
            
            .property-value {
                font-weight: 500;
                font-size: 1rem;
            }
            
            .molecule-tabs {
                margin-top: 1rem;
            }
            
            .molecule-footer {
                padding: 1rem;
                border-top: 1px solid #EEEEEE;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.85rem;
                color: #777777;
            }
            
            /* 分子比较视图样式 */
            .molecule-comparison {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            .molecule-comparison-item {
                flex: 1;
                min-width: 300px;
                border: 1px solid #EEEEEE;
                border-radius: 6px;
                overflow: hidden;
            }
            
            .comparison-header {
                padding: 0.75rem;
                background-color: #F5F7FA;
                border-bottom: 1px solid #EEEEEE;
                font-weight: 500;
            }
            
            /* 深色模式覆盖 */
            .dark-mode .molecule-container {
                background-color: #2A2A2A;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }
            
            .dark-mode .molecule-header,
            .dark-mode .molecule-footer {
                border-color: #444444;
            }
            
            .dark-mode .molecule-viewer-window {
                border-color: #444444;
            }
            
            .dark-mode .molecule-info {
                background-color: #333333;
            }
            
            .dark-mode .molecule-property {
                background-color: #2A2A2A;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
            }
            
            .dark-mode .property-label {
                color: #AAAAAA;
            }
            
            .dark-mode .molecule-comparison-item {
                border-color: #444444;
            }
            
            .dark-mode .comparison-header {
                background-color: #333333;
                border-color: #444444;
            }
        </style>
        """
    
    def _get_file_uploader_css(self) -> str:
        """获取文件上传器组件CSS样式"""
        return f"""
        <style>
            /* 文件上传器组件样式 */
            .file-upload-container {{
                background-color: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 2rem;
            }}
            
            .file-upload-header {{
                margin-bottom: 1.5rem;
            }}
            
            .file-upload-title {{
                font-weight: 600;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            }}
            
            .file-upload-description {{
                font-size: 0.9rem;
                color: #777777;
            }}
            
            .file-upload-dropzone {{
                border: 2px dashed #DDDDDD;
                border-radius: 6px;
                padding: 2rem;
                text-align: center;
                background-color: #F9FAFC;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .file-upload-dropzone:hover {{
                border-color: {PRIMARY_COLOR};
                background-color: {PRIMARY_COLOR}10;
            }}
            
            .file-upload-icon {{
                font-size: 2rem;
                color: #AAAAAA;
                margin-bottom: 1rem;
            }}
            
            .file-upload-text {{
                font-size: 1rem;
                color: #555555;
                margin-bottom: 0.5rem;
            }}
            
            .file-upload-hint {{
                font-size: 0.85rem;
                color: #777777;
            }}
            
            .file-preview {{
                margin-top: 1.5rem;
                padding: 1rem;
                background-color: #F5F7FA;
                border-radius: 6px;
            }}
            
            .file-preview-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #EEEEEE;
            }}
            
            .file-preview-title {{
                font-weight: 600;
            }}
            
            .file-preview-info {{
                display: flex;
                gap: 1rem;
                color: #777777;
                font-size: 0.85rem;
            }}
            
            .file-preview-content {{
                max-height: 300px;
                overflow-y: auto;
                font-family: 'Fira Code', monospace;
                font-size: 0.85rem;
                padding: 1rem;
                background-color: white;
                border-radius: 4px;
                border: 1px solid #EEEEEE;
            }}
            
            .file-batch-upload {{
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid #EEEEEE;
            }}
            
            .file-batch-title {{
                font-weight: 600;
                margin-bottom: 1rem;
            }}
            
            .file-format-tag {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                background-color: #F0F2F5;
                color: #555555;
            }}
            
            /* 深色模式覆盖 */
            .dark-mode .file-upload-container {{
                background-color: #2A2A2A;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }}
            
            .dark-mode .file-upload-description {{
                color: #AAAAAA;
            }}
            
            .dark-mode .file-upload-dropzone {{
                border-color: #444444;
                background-color: #333333;
            }}
            
            .dark-mode .file-upload-dropzone:hover {{
                border-color: {PRIMARY_COLOR};
                background-color: {PRIMARY_COLOR}33;
            }}
            
            .dark-mode .file-upload-icon {{
                color: #777777;
            }}
            
            .dark-mode .file-upload-text {{
                color: #EEEEEE;
            }}
            
            .dark-mode .file-upload-hint {{
                color: #AAAAAA;
            }}
            
            .dark-mode .file-preview {{
                background-color: #333333;
            }}
            
            .dark-mode .file-preview-header {{
                border-color: #444444;
            }}
            
            .dark-mode .file-preview-info {{
                color: #AAAAAA;
            }}
            
            .dark-mode .file-preview-content {{
                background-color: #222222;
                border-color: #444444;
            }}
            
            .dark-mode .file-batch-upload {{
                border-color: #444444;
            }}
            
            .dark-mode .file-format-tag {{
                background-color: #333333;
                color: #EEEEEE;
            }}
        </style>
        """
    
    def _get_plotly_theme(self) -> Dict[str, Any]:
        """获取Plotly图表主题配置"""
        if self.dark_mode:
            return {
                "template": "plotly_dark",
                "layout": {
                    "paper_bgcolor": "#1E1E1E",
                    "plot_bgcolor": "#2A2A2A",
                    "font": {
                        "color": "#EEEEEE"
                    },
                    "title": {
                        "font": {
                            "color": "#FFFFFF"
                        }
                    },
                    "legend": {
                        "font": {
                            "color": "#DDDDDD"
                        }
                    },
                    "colorway": CHART_COLORS
                }
            }
        else:
            return {
                "template": "plotly_white",
                "layout": {
                    "paper_bgcolor": "#FFFFFF",
                    "plot_bgcolor": "#F9FAFC",
                    "font": {
                        "color": "#333333"
                    },
                    "colorway": CHART_COLORS
                }
            }
    
    def _get_matplotlib_theme(self) -> Dict[str, Any]:
        """获取Matplotlib图表主题配置"""
        import matplotlib.pyplot as plt
        
        if self.dark_mode:
            plt.style.use("dark_background")
            return {
                "figure.facecolor": "#1E1E1E",
                "axes.facecolor": "#2A2A2A",
                "text.color": "#EEEEEE",
                "axes.labelcolor": "#EEEEEE",
                "axes.edgecolor": "#444444",
                "xtick.color": "#CCCCCC",
                "ytick.color": "#CCCCCC",
                "grid.color": "#444444",
                "grid.alpha": 0.3,
                "savefig.facecolor": "#1E1E1E",
                "savefig.edgecolor": "#1E1E1E",
                "lines.color": CHART_COLORS[0],
                "patch.edgecolor": "#2A2A2A"
            }
        else:
            plt.style.use("default")
            return {
                "figure.facecolor": "#FFFFFF",
                "axes.facecolor": "#F9FAFC",
                "grid.alpha": 0.3,
                "grid.color": "#CCCCCC"
            }
    
    def _get_altair_theme(self) -> Dict[str, Any]:
        """获取Altair图表主题配置"""
        if self.dark_mode:
            return {
                "config": {
                    "background": "#1E1E1E",
                    "title": {
                        "color": "#FFFFFF"
                    },
                    "style": {
                        "guide-label": {
                            "fill": "#EEEEEE"
                        },
                        "guide-title": {
                            "fill": "#FFFFFF"
                        }
                    },
                    "axis": {
                        "domainColor": "#444444",
                        "gridColor": "#444444",
                        "tickColor": "#444444"
                    },
                    "range": {
                        "category": CHART_COLORS
                    }
                }
            }
        else:
            return {
                "config": {
                    "background": "#FFFFFF",
                    "range": {
                        "category": CHART_COLORS
                    }
                }
            }
    
    def _generate_theme_css(self, theme: Dict[str, str]) -> str:
        """
        从主题字典生成CSS
        
        Args:
            theme: 主题颜色字典
            
        Returns:
            主题CSS字符串
        """
        primary_color = theme.get("primary", PRIMARY_COLOR)
        secondary_color = theme.get("secondary", SECONDARY_COLOR)
        accent_color = theme.get("accent", ACCENT_COLOR)
        
        return f"""
        <style>
            /* 自定义主题 */
            
            /* 标题颜色 */
            h1 {{
                color: {primary_color};
            }}
            
            /* 链接颜色 */
            a {{
                color: {primary_color};
            }}
            
            /* 按钮样式 */
            .stButton>button {{
                background-color: {primary_color};
            }}
            
            .stButton>button:hover {{
                background-color: {primary_color}DD;
            }}
            
            /* 工作流程步骤样式 */
            .workflow-step.active {{
                background-color: {primary_color}22;
                border-left: 4px solid {primary_color};
            }}
            
            .workflow-step.active .step-number {{
                background-color: {primary_color};
            }}
            
            /* 工作流程进度条 */
            .workflow-progress-bar {{
                background-color: {primary_color};
            }}
            
            /* 文件上传区域悬停 */
            .file-upload-dropzone:hover {{
                border-color: {primary_color};
                background-color: {primary_color}10;
            }}
            
            /* 导航激活项 */
            .h-caan-header .nav-item.active {{
                color: {primary_color};
            }}
            
            .h-caan-header .nav-item.active::after {{
                background-color: {primary_color};
            }}
            
            /* 侧边栏激活项 */
            .sidebar-menu-item.active {{
                background-color: {primary_color}33;
                color: {primary_color};
            }}
        </style>
        """

# 创建全局样式管理器实例
style_manager = StyleManager()
def apply_custom_styles():
    """应用自定义样式的便捷函数"""
    style_manager.apply_base_styles()