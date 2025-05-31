# utils/model_manager.py

import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器 - 负责模型的保存、加载和管理"""
    
    def __init__(self, models_dir: str = 'data/models'):
        self.models_dir = models_dir
        self.models_index_file = os.path.join(models_dir, 'models_index.json')
        self._ensure_directory()
        self._load_index()
        
    def _ensure_directory(self):
        """确保模型目录存在"""
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _load_index(self):
        """加载模型索引"""
        if os.path.exists(self.models_index_file):
            try:
                with open(self.models_index_file, 'r') as f:
                    self.models_index = json.load(f)
            except:
                self.models_index = {}
        else:
            self.models_index = {}
            
    def _save_index(self):
        """保存模型索引"""
        with open(self.models_index_file, 'w') as f:
            json.dump(self.models_index, f, indent=2)
            
    def register_model(self, model_path: str, task_name: str, 
                      metrics: Dict, metadata: Dict = None) -> str:
        """注册新模型"""
        model_id = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取文件大小
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        model_info = {
            'model_id': model_id,
            'model_path': model_path,
            'task_name': task_name,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {},
            'file_size_mb': file_size_mb
        }
        
        self.models_index[model_id] = model_info
        self._save_index()
        
        logger.info(f"注册模型: {model_id}")
        return model_id
        
    def list_models(self) -> List[Dict]:
        """列出所有已注册的模型"""
        # 返回模型列表，按创建时间降序排序
        models = list(self.models_index.values())
        
        # 按创建时间排序
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
        
    def get_model(self, model_id: str) -> Optional[Dict]:
        """获取指定模型信息"""
        return self.models_index.get(model_id)
        
    def get_latest_model(self) -> Optional[Dict]:
        """获取最新的模型"""
        models = self.list_models()
        return models[0] if models else None
        
    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        if model_id in self.models_index:
            model_info = self.models_index[model_id]
            
            # 删除模型文件
            if os.path.exists(model_info['model_path']):
                os.remove(model_info['model_path'])
                
            # 从索引中删除
            del self.models_index[model_id]
            self._save_index()
            
            logger.info(f"删除模型: {model_id}")
            return True
            
        return False
        
    def auto_discover_models(self):
        """自动发现models目录中的模型文件"""
        # 扫描models目录
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(self.models_dir, filename)
                
                # 检查是否已经注册
                already_registered = any(
                    info['model_path'] == model_path 
                    for info in self.models_index.values()
                )
                
                if not already_registered:
                    # 尝试加载模型获取信息
                    try:
                        model_data = joblib.load(model_path)
                        
                        # 从文件名推断任务名
                        task_name = filename.replace('ensemble_model_', '').replace('.pkl', '')
                        
                        # 提取metrics
                        metrics = {}
                        if isinstance(model_data, dict):
                            metrics = model_data.get('test_metrics', {})
                            
                        # 注册模型
                        self.register_model(
                            model_path=model_path,
                            task_name=task_name,
                            metrics=metrics,
                            metadata={'auto_discovered': True}
                        )
                        
                        logger.info(f"自动发现并注册模型: {filename}")
                        
                    except Exception as e:
                        logger.warning(f"无法加载模型 {filename}: {str(e)}")
                        
    def get_model_by_path(self, model_path: str) -> Optional[Dict]:
        """根据模型路径获取模型信息"""
        for model_info in self.models_index.values():
            if model_info['model_path'] == model_path:
                return model_info
        return None
        
    def update_model_metrics(self, model_id: str, new_metrics: Dict):
        """更新模型的性能指标"""
        if model_id in self.models_index:
            self.models_index[model_id]['metrics'].update(new_metrics)
            self._save_index()
            logger.info(f"更新模型 {model_id} 的指标")