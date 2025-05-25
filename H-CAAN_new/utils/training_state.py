# utils/training_state.py
import json
import os
from datetime import datetime

class TrainingStateManager:
    def __init__(self, state_dir='data/states'):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        
    def save_state(self, task_name: str, state: dict):
        """保存训练状态"""
        state_file = os.path.join(self.state_dir, f"{task_name}_state.json")
        state['timestamp'] = datetime.now().isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, task_name: str) -> dict:
        """加载训练状态"""
        state_file = os.path.join(self.state_dir, f"{task_name}_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                return json.load(f)
        return {}