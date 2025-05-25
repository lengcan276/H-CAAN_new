# streamlit_ui/pages/1_Home.py
import streamlit as st
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import necessary modules
from streamlit_ui.state import state_manager
from streamlit_ui.workflow import get_workflow_status

# Page configuration
st.set_page_config(
    page_title="Home - H-CAAN",
    page_icon="🏠",
    layout="wide"
)

def render():
    """Render home page"""
    st.title("H-CAAN: Hierarchical Cross-modal Adaptive Attention Network")
    
    # Welcome section
    st.header("Welcome to H-CAAN System")
    
    st.write("""
    H-CAAN is a multimodal fusion deep learning framework designed specifically for drug property prediction. 
    It integrates chemical language and molecular graph information through hierarchical cross-modal adaptive 
    attention mechanisms to improve prediction accuracy.
    """)
    
    # Workflow status
    st.subheader("Workflow Status")
    
    # Calculate progress
    steps = ["data_processed", "model_configured", "model_trained", "results_analyzed"]
    completed = sum(1 for step in steps if st.session_state.get(step, False))
    progress = completed / len(steps)
    
    # Display progress bar
    st.progress(progress)
    st.caption(f"{completed} out of {len(steps)} steps completed")
    
    # Display specific status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_status = "Completed" if st.session_state.get("data_processed", False) else "Pending"
        status_color = "success" if st.session_state.get("data_processed", False) else "warning"
        st.metric("Data Preparation", data_status)
    
    with col2:
        model_status = "Completed" if st.session_state.get("model_configured", False) else "Pending"
        st.metric("Model Configuration", model_status)
    
    with col3:
        train_status = "Completed" if st.session_state.get("model_trained", False) else "Pending"
        st.metric("Model Training", train_status)
    
    with col4:
        results_status = "Completed" if st.session_state.get("results_analyzed", False) else "Pending"
        st.metric("Results Analysis", results_status)
    
    # Main function modules
    st.header("Main Function Modules")
    
    # Use three column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Preparation")
        st.write("Upload your molecular dataset and preprocess it for model training.")
        
        if st.button("Go to Data Preparation", key="goto_data", use_container_width=True):
            st.switch_page("pages/2_Data_Preparation.py")
    
    with col2:
        st.subheader("Model Configuration & Training")
        st.write("Configure model architecture, select training parameters and execute training process.")
        
        # Check if data is ready
        button_disabled = not st.session_state.get("data_processed", False)
        
        if st.button("Go to Model Configuration", disabled=button_disabled, key="goto_model", use_container_width=True):
            st.switch_page("pages/3_Model_Configuration.py")
        
        if button_disabled:
            st.caption("⚠️ Please complete data preparation first")
    
    with col3:
        st.subheader("Results Visualization")
        st.write("Analyze and visualize model prediction results, explore modal contributions and performance.")
        
        # Check if model is trained
        button_disabled = not st.session_state.get("model_trained", False)
        
        if st.button("Go to Results", disabled=button_disabled, key="goto_results", use_container_width=True):
            st.switch_page("pages/5_Results_Analysis.py")
        
        if button_disabled:
            st.caption("⚠️ Please complete model training first")
    
    # System Architecture
    st.header("System Architecture")
    
    # Display architecture as columns
    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
    
    with arch_col1:
        st.info("""
        **Input Layer**
        - SMILES vectors
        - ECFP fingerprints
        - Molecular graphs
        """)
    
    with arch_col2:
        st.info("""
        **Encoding Layer**
        - Transformer-Encoder
        - BiGRU
        - GCN
        """)
    
    with arch_col3:
        st.info("""
        **Fusion Layer**
        - Hierarchical attention
        - Cross-modal fusion
        - GCAU module
        """)
    
    with arch_col4:
        st.info("""
        **Output Layer**
        - Property prediction
        - Confidence scores
        - Feature importance
        """)
    
    # Recommended workflow
    st.header("Recommended Workflow")
    
    workflow_steps = """
    1. **Data Preparation** → Upload and preprocess molecular data
    2. **Model Configuration** → Select model architecture and parameters
    3. **Training & Evaluation** → Train model and evaluate performance
    4. **Results Visualization** → Analyze predictions and model interpretations
    5. **Paper Generation** → Generate comprehensive research report
    """
    
    st.info(workflow_steps)
    
    # Session information
    st.header("Session Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display basic session info
        session_id = st.session_state.get("session_id", "Unknown")
        st.write(f"**Session ID:** {session_id}")
        
        # Display current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**Current Time:** {current_time}")
        
        # Display Python version
        import sys
        st.write(f"**Python Version:** {sys.version.split()[0]}")
    
    with col2:
        # Display dataset info (if available)
        if st.session_state.get("dataset_info"):
            dataset_info = st.session_state["dataset_info"]
            st.write(f"**Current Dataset:** {dataset_info.get('name', 'Unknown')}")
            st.write(f"**Dataset Size:** {dataset_info.get('size', 0)} records")
        else:
            st.write("**Current Dataset:** Not selected")
        
        # Display model info (if available)
        if st.session_state.get("model_config"):
            model_config = st.session_state["model_config"]
            st.write(f"**Model Architecture:** {model_config.get('architecture', 'Not configured')}")
        else:
            st.write("**Model:** Not configured")
    
    # Quick actions
    st.header("Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📚 View Documentation", use_container_width=True):
            st.info("Documentation feature coming soon...")
    
    with action_col2:
        if st.button("💾 Save Session", use_container_width=True):
            from streamlit_ui.state import save_state
            save_state()
            st.success("Session saved successfully!")
    
    with action_col3:
        if st.button("🔄 Reset Workflow", use_container_width=True):
            # Reset all workflow states
            for key in ["data_processed", "model_configured", "model_trained", 
                       "results_analyzed", "paper_generated"]:
                st.session_state[key] = False
            st.rerun()
    
    # Footer
    st.divider()
    st.caption("H-CAAN is a multimodal fusion deep learning framework for drug property prediction. © 2023-2024 Project Team")

# Main render
render()