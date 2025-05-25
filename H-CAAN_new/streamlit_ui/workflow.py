# streamlit_ui/workflow.py
import streamlit as st
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
from datetime import datetime

class WorkflowStep(Enum):
    """Define workflow steps in correct order"""
    HOME = "home"
    DATA_PREPARATION = "data_preparation"
    MODEL_CONFIGURATION = "model_configuration"
    TRAINING = "training"
    RESULTS = "results"
    PAPER = "paper"  # Paper is the last step

# Page display names
PAGE_NAMES = {
    WorkflowStep.HOME: "🏠 Home",
    WorkflowStep.DATA_PREPARATION: "📊 Data Preparation",
    WorkflowStep.MODEL_CONFIGURATION: "🔧 Model Configuration",
    WorkflowStep.TRAINING: "🚀 Training",
    WorkflowStep.RESULTS: "📈 Results",
    WorkflowStep.PAPER: "📝 Paper Generation"  # Last step
}

# Workflow dependencies
WORKFLOW_DEPENDENCIES = {
    WorkflowStep.HOME: [],
    WorkflowStep.DATA_PREPARATION: [WorkflowStep.HOME],
    WorkflowStep.MODEL_CONFIGURATION: [WorkflowStep.DATA_PREPARATION],
    WorkflowStep.TRAINING: [WorkflowStep.MODEL_CONFIGURATION],
    WorkflowStep.RESULTS: [WorkflowStep.TRAINING],
    WorkflowStep.PAPER: [WorkflowStep.RESULTS]  # Paper depends on Results
}

def initialize_workflow_state():
    """Initialize workflow state in session state"""
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = {
            'current_step': WorkflowStep.HOME,
            'completed_steps': set(),
            'step_data': {},
            'workflow_history': [],
            'start_time': datetime.now().isoformat()
        }
    
    # Initialize workflow progress tracking
    if 'workflow_progress' not in st.session_state:
        st.session_state.workflow_progress = {
            WorkflowStep.HOME: 100,  # Home is always complete
            WorkflowStep.DATA_PREPARATION: 0,
            WorkflowStep.MODEL_CONFIGURATION: 0,
            WorkflowStep.TRAINING: 0,
            WorkflowStep.RESULTS: 0,
            WorkflowStep.PAPER: 0
        }

def get_current_step() -> WorkflowStep:
    """Get current workflow step"""
    initialize_workflow_state()
    return st.session_state.workflow_state['current_step']

def set_current_step(step: WorkflowStep):
    """Set current workflow step"""
    initialize_workflow_state()
    st.session_state.workflow_state['current_step'] = step
    
    # Add to history
    history_entry = {
        'step': step.value,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.workflow_state['workflow_history'].append(history_entry)

def mark_step_completed(step: WorkflowStep):
    """Mark a step as completed"""
    initialize_workflow_state()
    st.session_state.workflow_state['completed_steps'].add(step)
    st.session_state.workflow_progress[step] = 100

def is_step_completed(step: WorkflowStep) -> bool:
    """Check if a step is completed"""
    initialize_workflow_state()
    return step in st.session_state.workflow_state['completed_steps']

def save_step_data(step: WorkflowStep, data: Dict):
    """Save data for a specific step"""
    initialize_workflow_state()
    st.session_state.workflow_state['step_data'][step.value] = data

def get_step_data(step: WorkflowStep) -> Dict:
    """Get data for a specific step"""
    initialize_workflow_state()
    return st.session_state.workflow_state['step_data'].get(step.value, {})

def check_workflow_dependencies(step: WorkflowStep) -> Tuple[bool, List[WorkflowStep], str]:
    """
    Check if all dependencies for a step are satisfied
    
    Returns:
        tuple: (can_access, missing_deps, message)
    """
    dependencies = WORKFLOW_DEPENDENCIES.get(step, [])
    missing_deps = []
    
    for dep in dependencies:
        if not is_step_completed(dep):
            missing_deps.append(dep)
    
    can_access = len(missing_deps) == 0
    
    if not can_access:
        dep_names = [PAGE_NAMES[dep] for dep in missing_deps]
        message = f"Please complete the following steps first: {', '.join(dep_names)}"
    else:
        message = ""
    
    return can_access, missing_deps, message

def can_navigate_to_step(step: WorkflowStep) -> bool:
    """Check if user can navigate to a specific step"""
    can_access, _, _ = check_workflow_dependencies(step)
    return can_access

def get_next_step(current_step: WorkflowStep) -> Optional[WorkflowStep]:
    """Get the next step in the workflow"""
    steps = list(WorkflowStep)
    current_index = steps.index(current_step)
    
    if current_index < len(steps) - 1:
        return steps[current_index + 1]
    return None

def get_previous_step(current_step: WorkflowStep) -> Optional[WorkflowStep]:
    """Get the previous step in the workflow"""
    steps = list(WorkflowStep)
    current_index = steps.index(current_step)
    
    if current_index > 0:
        return steps[current_index - 1]
    return None

def navigate_to_next_step():
    """Navigate to the next step in the workflow"""
    current = get_current_step()
    next_step = get_next_step(current)
    
    if next_step and can_navigate_to_step(next_step):
        set_current_step(next_step)
        st.rerun()

def navigate_to_previous_step():
    """Navigate to the previous step in the workflow"""
    current = get_current_step()
    prev_step = get_previous_step(current)
    
    if prev_step:
        set_current_step(prev_step)
        st.rerun()

def get_workflow_progress() -> Dict[WorkflowStep, int]:
    """Get progress for all workflow steps"""
    initialize_workflow_state()
    return st.session_state.workflow_progress

def update_step_progress(step: WorkflowStep, progress: int):
    """Update progress for a specific step"""
    initialize_workflow_state()
    st.session_state.workflow_progress[step] = min(100, max(0, progress))

def reset_workflow():
    """Reset the entire workflow"""
    if 'workflow_state' in st.session_state:
        del st.session_state.workflow_state
    if 'workflow_progress' in st.session_state:
        del st.session_state.workflow_progress
    initialize_workflow_state()

def export_workflow_state() -> str:
    """Export workflow state as JSON"""
    initialize_workflow_state()
    
    # Convert sets to lists for JSON serialization
    completed_steps = list(st.session_state.workflow_state['completed_steps'])
    completed_steps_values = [step.value for step in completed_steps]
    
    state_data = {
        'workflow_state': {
            'current_step': st.session_state.workflow_state['current_step'].value,
            'completed_steps': completed_steps_values,
            'step_data': st.session_state.workflow_state['step_data'],
            'workflow_history': st.session_state.workflow_state['workflow_history'],
            'start_time': st.session_state.workflow_state['start_time']
        },
        'workflow_progress': {
            step.value: progress 
            for step, progress in st.session_state.workflow_progress.items()
        },
        'export_time': datetime.now().isoformat()
    }
    return json.dumps(state_data, indent=2)

def import_workflow_state(json_data: str) -> bool:
    """Import workflow state from JSON"""
    try:
        data = json.loads(json_data)
        
        # Restore workflow state
        if 'workflow_state' in data:
            st.session_state.workflow_state = {
                'current_step': WorkflowStep(data['workflow_state']['current_step']),
                'completed_steps': {WorkflowStep(step) for step in data['workflow_state']['completed_steps']},
                'step_data': data['workflow_state']['step_data'],
                'workflow_history': data['workflow_state']['workflow_history'],
                'start_time': data['workflow_state']['start_time']
            }
        
        # Restore workflow progress
        if 'workflow_progress' in data:
            st.session_state.workflow_progress = {
                WorkflowStep(step): progress 
                for step, progress in data['workflow_progress'].items()
            }
        
        return True
    except Exception as e:
        st.error(f"Failed to import workflow state: {str(e)}")
        return False

def get_workflow_summary() -> Dict:
    """Get a summary of the current workflow state"""
    initialize_workflow_state()
    
    progress_dict = get_workflow_progress()
    completed_count = sum(1 for p in progress_dict.values() if p == 100)
    total_count = len(progress_dict)
    
    # Calculate time elapsed
    start_time = datetime.fromisoformat(st.session_state.workflow_state['start_time'])
    elapsed_time = datetime.now() - start_time
    
    return {
        'current_step': get_current_step(),
        'completed_steps': list(st.session_state.workflow_state['completed_steps']),
        'progress_percentage': int((completed_count / total_count) * 100) if total_count > 0 else 0,
        'completed_count': completed_count,
        'total_count': total_count,
        'elapsed_time': str(elapsed_time).split('.')[0],  # Remove microseconds
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S')
    }