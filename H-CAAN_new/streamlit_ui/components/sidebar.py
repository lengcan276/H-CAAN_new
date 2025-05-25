# streamlit_ui/components/sidebar.py
import streamlit as st
from streamlit_ui.state import get_state, set_state
from streamlit_ui.workflow import (
    WorkflowStep, 
    PAGE_NAMES, 
    get_current_step, 
    set_current_step,
    can_navigate_to_step,
    check_workflow_dependencies
)

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.markdown("# 🧪 H-CAAN")
        st.markdown("---")
        
        # Navigation
        st.markdown("## 📍 Navigation")
        
        current_step = get_current_step()
        
        # Define correct page order
        page_order = [
            WorkflowStep.HOME,
            WorkflowStep.DATA_PREPARATION,
            WorkflowStep.MODEL_CONFIGURATION,
            WorkflowStep.TRAINING,
            WorkflowStep.RESULTS,
            WorkflowStep.PAPER  # Paper is last
        ]
        
        # Render navigation buttons in correct order
        for step in page_order:
            # Check if step can be accessed
            can_access, missing_deps, message = check_workflow_dependencies(step)
            
            # Style button based on status
            if step == current_step:
                button_type = "primary"
            elif can_access:
                button_type = "secondary"
            else:
                button_type = "secondary"
            
            # Create button
            if st.button(
                PAGE_NAMES[step],
                key=f"nav_{step.value}",
                disabled=not can_access,
                use_container_width=True,
                type=button_type if step == current_step else "secondary"
            ):
                set_current_step(step)
                st.rerun()
            
            # Show dependency message if needed
            if not can_access and message:
                st.caption(f"🔒 {message}")
        
        # Workflow controls
        st.markdown("---")
        st.markdown("## 🎮 Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                if st.confirm("Are you sure you want to reset the workflow?"):
                    from streamlit_ui.workflow import reset_workflow
                    reset_workflow()
                    st.rerun()
        
        with col2:
            if st.button("💾 Save", use_container_width=True):
                from streamlit_ui.workflow import export_workflow_state
                state_json = export_workflow_state()
                st.download_button(
                    label="Download State",
                    data=state_json,
                    file_name=f"h_caan_workflow_{current_step.value}.json",
                    mime="application/json"
                )
        
        # Workflow status summary
        st.markdown("---")
        st.markdown("## 📊 Status")
        render_sidebar_status()

def render_sidebar_status():
    """Render workflow status in sidebar"""
    from streamlit_ui.workflow import get_workflow_progress, is_step_completed
    
    progress_dict = get_workflow_progress()
    
    # Calculate overall progress
    completed = sum(1 for p in progress_dict.values() if p == 100)
    total = len(progress_dict)
    
    # Progress metric
    st.metric(
        label="Overall Progress",
        value=f"{completed}/{total}",
        delta=f"{(completed/total*100):.0f}%"
    )
    
    # Mini progress bar
    st.progress(completed / total if total > 0 else 0)
    
    # Quick stats
    stats_container = st.container()
    with stats_container:
        # Count completed steps
        if completed > 0:
            st.success(f"✅ {completed} steps completed")
        
        # Show current step
        current = get_current_step()
        if current != WorkflowStep.HOME:
            st.info(f"🔄 Working on: {PAGE_NAMES[current]}")
        
        # Show next step
        remaining = total - completed
        if remaining > 0 and current != WorkflowStep.PAPER:
            st.caption(f"⏳ {remaining} steps remaining")