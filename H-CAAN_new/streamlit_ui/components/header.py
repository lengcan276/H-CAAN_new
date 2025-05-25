# streamlit_ui/components/header.py
"""
Page header and navigation component - displays application title and top navigation bar
"""
import streamlit as st
from datetime import datetime
from streamlit_ui.workflow import (
    get_current_step, 
    PAGE_NAMES, 
    WorkflowStep,
    set_current_step,
    get_workflow_progress
)

def render_header():
    """
    Render page header with title and status indicators
    """
    # Create container for styling control
    header_container = st.container()
    
    with header_container:
        # Application title and logo area
        cols = st.columns([1, 10, 2])
        
        with cols[0]:
            # Display application icon
            st.markdown("# 🧬", unsafe_allow_html=True)
        
        with cols[1]:
            # Application title
            st.markdown("""
            <h1 style='margin-bottom: 0px; margin-top: 0px; font-size: 2.2em; color: #1f77b4;'>H-CAAN</h1>
            <p style='margin-top: 0px; color: gray; font-size: 1.1em;'>
                Hierarchical Cross-modal Adaptive Attention Network
            </p>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            # User info/time/status indicator
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Get overall progress
            progress_dict = get_workflow_progress()
            completed_steps = sum(1 for p in progress_dict.values() if p == 100)
            total_steps = len(progress_dict)
            progress_percentage = int((completed_steps / total_steps * 100) if total_steps > 0 else 0)
            
            st.markdown(f"""
            <div style='text-align: right;'>
                <p style='margin-bottom: 0px; color: gray; font-size: 0.9em;'>{current_time}</p>
                <p style='margin-top: 0px; font-size: 0.9em;'>
                    Progress: <strong>{progress_percentage}%</strong> ({completed_steps}/{total_steps})
                </p>
                <p style='margin-top: -10px; font-size: 0.9em;'>
                    <span style='color: {"green" if st.session_state.get("is_connected", True) else "red"};'>
                        {"●" if st.session_state.get("is_connected", True) else "○"}
                    </span>
                    System {"Connected" if st.session_state.get("is_connected", True) else "Disconnected"}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add separator
    st.divider()
    
    # Return header container for additional content
    return header_container

def render_subheader(title, description=None, icon=None):
    """
    Render page subheader with title and description
    
    Args:
        title (str): Page title
        description (str, optional): Page description
        icon (str, optional): Icon emoji
    """
    # Create container
    subheader_container = st.container()
    
    with subheader_container:
        # Create row layout
        cols = st.columns([10, 1])
        
        with cols[0]:
            # Add icon and title
            if icon:
                st.markdown(f"# {icon} {title}")
            else:
                st.markdown(f"# {title}")
                
            # Add description
            if description:
                st.markdown(
                    f"<p style='color: gray; font-size: 1.1em; margin-top: -20px;'>{description}</p>", 
                    unsafe_allow_html=True
                )
        
        # Add home button
        with cols[1]:
            if st.button("🏠 Home", key="return_home", help="Return to home page"):
                set_current_step(WorkflowStep.HOME)
                st.rerun()
    
    # Add separator
    st.divider()
    
    return subheader_container

def render_breadcrumbs():
    """
    Render breadcrumb navigation based on workflow steps
    """
    current_step = get_current_step()
    
    # Build breadcrumb path
    breadcrumb_path = []
    
    # Define step order
    step_order = [
        WorkflowStep.HOME,
        WorkflowStep.DATA_PREPARATION,
        WorkflowStep.MODEL_CONFIGURATION,
        WorkflowStep.TRAINING,
        WorkflowStep.RESULTS,
        WorkflowStep.PAPER
    ]
    
    # Find current position and build path
    for step in step_order:
        breadcrumb_path.append((step, PAGE_NAMES[step]))
        if step == current_step:
            break
    
    # Create breadcrumb container
    container = st.container()
    
    with container:
        # Build breadcrumb HTML
        breadcrumb_html = """
        <div style='
            margin-bottom: 20px; 
            padding: 10px 20px; 
            background-color: #f8f9fa; 
            border-radius: 5px;
            font-size: 0.9em;
        '>
        """
        
        for i, (step, name) in enumerate(breadcrumb_path):
            if i > 0:
                breadcrumb_html += " <span style='color: #ccc; margin: 0 8px;'>›</span> "
            
            # Make all but last item clickable
            if i < len(breadcrumb_path) - 1:
                breadcrumb_html += f"""
                <a href='#' 
                   onclick='navigateToStep("{step.value}"); return false;' 
                   style='color: #1f77b4; text-decoration: none; hover: underline;'>
                   {name}
                </a>
                """
            else:
                # Current page (not clickable)
                breadcrumb_html += f"<span style='color: #333; font-weight: bold;'>{name}</span>"
        
        breadcrumb_html += "</div>"
        
        # Display breadcrumbs
        st.markdown(breadcrumb_html, unsafe_allow_html=True)
        
        # Add JavaScript for navigation (placeholder - actual navigation handled by Streamlit)
        st.markdown("""
        <script>
        function navigateToStep(stepValue) {
            // This is a placeholder - actual navigation is handled by Streamlit buttons
            console.log('Navigate to:', stepValue);
        }
        </script>
        """, unsafe_allow_html=True)
    
    # Add navigation buttons for actual functionality
    if len(breadcrumb_path) > 1:
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            # Back button
            if st.button("← Back", key="breadcrumb_back", use_container_width=True):
                prev_step = breadcrumb_path[-2][0]  # Second to last step
                set_current_step(prev_step)
                st.rerun()
    
    return container

def render_page_header(step: WorkflowStep):
    """
    Render complete page header with title, breadcrumbs, and status
    
    Args:
        step (WorkflowStep): Current workflow step
    """
    # Render main header
    render_header()
    
    # Render breadcrumbs
    render_breadcrumbs()
    
    # Get page info
    page_name = PAGE_NAMES.get(step, "Unknown Page")
    page_emoji = get_page_emoji(step)
    page_description = get_page_description(step)
    
    # Render subheader
    render_subheader(
        title=page_name.replace(page_emoji, "").strip(),
        description=page_description,
        icon=page_emoji
    )

def get_page_emoji(step: WorkflowStep) -> str:
    """Get emoji for workflow step"""
    emoji_map = {
        WorkflowStep.HOME: "🏠",
        WorkflowStep.DATA_PREPARATION: "📊",
        WorkflowStep.MODEL_CONFIGURATION: "🔧",
        WorkflowStep.TRAINING: "🚀",
        WorkflowStep.RESULTS: "📈",
        WorkflowStep.PAPER: "📝"
    }
    return emoji_map.get(step, "📌")

def get_page_description(step: WorkflowStep) -> str:
    """Get description for workflow step"""
    description_map = {
        WorkflowStep.HOME: "Welcome to H-CAAN - Start your molecular property prediction journey",
        WorkflowStep.DATA_PREPARATION: "Upload and prepare your molecular datasets for training",
        WorkflowStep.MODEL_CONFIGURATION: "Configure the H-CAAN model architecture and hyperparameters",
        WorkflowStep.TRAINING: "Train your model with the prepared data",
        WorkflowStep.RESULTS: "Analyze model performance and visualize predictions",
        WorkflowStep.PAPER: "Generate automated research paper based on your results"
    }
    return description_map.get(step, "")

def render_quick_stats():
    """
    Render quick statistics in header area
    """
    stats_container = st.container()
    
    with stats_container:
        cols = st.columns(4)
        
        # Get workflow progress
        progress_dict = get_workflow_progress()
        completed_steps = sum(1 for p in progress_dict.values() if p == 100)
        
        # Display metrics
        with cols[0]:
            st.metric(
                label="📊 Dataset",
                value=st.session_state.get("dataset_name", "Not loaded"),
                delta=st.session_state.get("dataset_size", "")
            )
        
        with cols[1]:
            st.metric(
                label="🧬 Model",
                value=st.session_state.get("model_type", "Not configured"),
                delta=st.session_state.get("model_params", "")
            )
        
        with cols[2]:
            st.metric(
                label="📈 Best Performance",
                value=st.session_state.get("best_metric", "N/A"),
                delta=st.session_state.get("improvement", "")
            )
        
        with cols[3]:
            st.metric(
                label="✅ Progress",
                value=f"{completed_steps}/{len(progress_dict)}",
                delta=f"{int(completed_steps/len(progress_dict)*100)}%"
            )
    
    return stats_container

# Test code
if __name__ == "__main__":
    st.set_page_config(page_title="Header Component Test", layout="wide")
    
    # Initialize test session state
    if 'test_step' not in st.session_state:
        st.session_state.test_step = WorkflowStep.DATA_PREPARATION
    
    # Render page header
    render_page_header(st.session_state.test_step)
    
    # Test quick stats
    st.session_state.dataset_name = "QM9"
    st.session_state.dataset_size = "133,885 molecules"
    st.session_state.model_type = "H-CAAN"
    st.session_state.model_params = "88M parameters"
    st.session_state.best_metric = "0.95 R²"
    st.session_state.improvement = "+15%"
    
    render_quick_stats()