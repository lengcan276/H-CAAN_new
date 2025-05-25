# 新建文件: utils/model_manager.py
import json
import os
from datetime import datetime

class ModelManager:
    """模型信息持久化管理"""
    
    def __init__(self, storage_path='data/models/model_registry.json'):
        self.storage_path = storage_path
        self._ensure_file()
    
    def _ensure_file(self):
        """确保存储文件存在"""
        if not os.path.exists(self.storage_path):
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({}, f)
    
    def save_model_info(self, task_name: str, model_path: str, metrics: dict):
        """保存模型信息"""
        with open(self.storage_path, 'r') as f:
            registry = json.load(f)
        
        registry[task_name] = {
            'model_path': model_path,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def get_latest_model(self, task_name: str = None):
        """获取最新的模型信息"""
        with open(self.storage_path, 'r') as f:
            registry = json.load(f)
        
        if not registry:
            return None
            
        if task_name:
            return registry.get(task_name)
        
        # 返回最新的模型
        latest = max(registry.items(), 
                    key=lambda x: x[1].get('timestamp', ''))
        return latest[1]