"""
智能体调用顺序与依赖定义
定义各种任务链和执行顺序
"""
from typing import List, Dict, Callable, Any
import logging

logger = logging.getLogger(__name__)

class TaskChain:
    """任务链定义"""
    
    def __init__(self):
        self.chains = {
            'data_processing': self._create_data_processing_chain(),
            'model_training': self._create_model_training_chain(),
            'full_analysis': self._create_full_analysis_chain(),
            'quick_prediction': self._create_quick_prediction_chain()
        }
        
        self.dependencies = self._define_dependencies()
        
    def _create_data_processing_chain(self) -> List[Dict]:
        """创建数据处理链"""
        return [
            {
                'name': 'load_data',
                'agent': 'data',
                'method': 'load_raw_data',
                'required_inputs': ['data_path'],
                'outputs': ['raw_data']
            },
            {
                'name': 'preprocess_data',
                'agent': 'data',
                'method': 'preprocess_data',
                'required_inputs': ['raw_data'],
                'outputs': ['processed_data']
            }
        ]
        
    def _create_model_training_chain(self) -> List[Dict]:
        """创建模型训练链"""
        return [
            {
                'name': 'fuse_features',
                'agent': 'fusion',
                'method': 'fuse_features',
                'required_inputs': ['processed_data'],
                'outputs': ['fused_features']
            },
            {
                'name': 'train_model',
                'agent': 'model',
                'method': 'train_model',
                'required_inputs': ['fused_features', 'labels', 'train_params'],
                'outputs': ['model_path']
            }
        ]
        
    def _create_full_analysis_chain(self) -> List[Dict]:
        """创建完整分析链"""
        chain = []
        chain.extend(self._create_data_processing_chain())
        chain.extend(self._create_model_training_chain())
        chain.extend([
            {
                'name': 'generate_explanations',
                'agent': 'explain',
                'method': 'generate_explanations',
                'required_inputs': ['model_path', 'fused_features'],
                'outputs': ['explanations']
            },
            {
                'name': 'generate_paper',
                'agent': 'paper',
                'method': 'generate_paper',
                'required_inputs': ['results', 'explanations', 'metadata'],
                'outputs': ['paper_path']
            }
        ])
        return chain
        
    def _create_quick_prediction_chain(self) -> List[Dict]:
        """创建快速预测链"""
        chain = self._create_data_processing_chain()
        chain.extend([
            {
                'name': 'fuse_features',
                'agent': 'fusion',
                'method': 'fuse_features',
                'required_inputs': ['processed_data'],
                'outputs': ['fused_features']
            },
            {
                'name': 'predict',
                'agent': 'model',
                'method': 'predict',
                'required_inputs': ['model_path', 'fused_features'],
                'outputs': ['predictions', 'uncertainties']
            }
        ])
        return chain
        
    def _define_dependencies(self) -> Dict[str, List[str]]:
        """定义任务依赖关系"""
        return {
            'preprocess_data': ['load_data'],
            'fuse_features': ['preprocess_data'],
            'train_model': ['fuse_features'],
            'predict': ['fuse_features'],
            'generate_explanations': ['train_model', 'fuse_features'],
            'generate_paper': ['generate_explanations']
        }
        
    def get_chain(self, chain_name: str) -> List[Dict]:
        """获取任务链"""
        if chain_name not in self.chains:
            raise ValueError(f"未知任务链: {chain_name}")
        return self.chains[chain_name]
        
    def validate_chain(self, chain: List[Dict]) -> bool:
        """验证任务链的合法性"""
        seen_outputs = set()
        
        for task in chain:
            # 检查依赖是否满足
            for required_input in task['required_inputs']:
                if required_input not in seen_outputs and required_input not in ['data_path', 'labels', 'train_params', 'model_path', 'metadata', 'results']:
                    logger.error(f"任务 {task['name']} 缺少输入: {required_input}")
                    return False
                    
            # 添加输出
            seen_outputs.update(task['outputs'])
            
        return True
        
    def optimize_chain(self, chain: List[Dict]) -> List[Dict]:
        """优化任务链，识别可并行的任务"""
        optimized = []
        current_batch = []
        
        for task in chain:
            # 检查是否可以与当前批次并行
            can_parallel = True
            for batch_task in current_batch:
                if self._has_dependency(task, batch_task):
                    can_parallel = False
                    break
                    
            if can_parallel:
                current_batch.append(task)
            else:
                if current_batch:
                    optimized.append(current_batch)
                current_batch = [task]
                
        if current_batch:
            optimized.append(current_batch)
            
        return optimized
        
    def _has_dependency(self, task1: Dict, task2: Dict) -> bool:
        """检查两个任务之间是否有依赖关系"""
        # 检查task1是否依赖task2的输出
        task2_outputs = set(task2['outputs'])
        task1_inputs = set(task1['required_inputs'])
        
        return bool(task2_outputs & task1_inputs)
        
    def create_custom_chain(self, tasks: List[str]) -> List[Dict]:
        """创建自定义任务链"""
        # 这里可以根据任务名称动态创建任务链
        custom_chain = []
        
        # 任务模板
        task_templates = {
            'load': self.chains['data_processing'][0],
            'preprocess': self.chains['data_processing'][1],
            'fuse': self.chains['model_training'][0],
            'train': self.chains['model_training'][1],
            'predict': self.chains['quick_prediction'][-1],
            'explain': self.chains['full_analysis'][-2],
            'paper': self.chains['full_analysis'][-1]
        }
        
        for task_name in tasks:
            if task_name in task_templates:
                custom_chain.append(task_templates[task_name])
            else:
                logger.warning(f"未知任务: {task_name}")
                
        # 验证自定义链
        if self.validate_chain(custom_chain):
            return custom_chain
        else:
            raise ValueError("自定义任务链验证失败")