"""
多智能体统一调度与任务管理
协调各智能体的执行和状态管理
"""
import asyncio
#from typing import Dict, Any, List, Optional, Callable
from typing import Dict, Any, List, Optional, Tuple,Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from datetime import datetime
import json
import os
import numpy as np

from .data_agent import DataAgent
from .fusion_agent import FusionAgent
from .model_agent import ModelAgent
from .explain_agent import ExplainAgent
from .paper_agent import PaperAgent

logger = logging.getLogger(__name__)

class MultiAgentManager:
    """多智能体管理器"""
    
    def __init__(self):
        # 初始化所有智能体
        self.agents = {
            'data': DataAgent(),
            'fusion': FusionAgent(),
            'model': ModelAgent(),
            'explain': ExplainAgent(),
            'paper': PaperAgent()
        }
        
        # 任务状态管理
        self.task_status = {}
        self.task_results = {}
        
        # 线程池用于并行执行
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 任务映射
        self.task_mapping = {
            'load_data': self._handle_load_data,
            'preprocess_data': self._handle_preprocess_data,
            'split_data': self._handle_split_data,  
            'fuse_features': self._handle_fuse_features,
            'train_model': self._handle_train_model,
            'predict': self._handle_predict,
            'explain': self._handle_explain,
            'generate_paper': self._handle_generate_paper
        }
        
        # 工作流定义
        self.workflows = {
            'full_pipeline': [
                'load_data', 'preprocess_data', 'split_data', 'fuse_features',
                'train_model', 'explain', 'generate_paper'
            ],
            'prediction_only': [
                'load_data', 'preprocess_data', 'split_data', 'fuse_features', 'predict'
            ],
            'analysis_only': [
                'load_data', 'preprocess_data', 'explain'
            ]
        }
                
    def dispatch_task(self, task_name: str, **kwargs) -> Any:
        """
        分发单个任务
        
        Args:
            task_name: 任务名称
            **kwargs: 任务参数
            
        Returns:
            任务执行结果
        """
        logger.info(f"分发任务: {task_name}")
        
        if task_name not in self.task_mapping:
            raise ValueError(f"未知任务: {task_name}")
            
        # 创建任务ID
        task_id = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.task_status[task_id] = 'running'
        
        try:
            # 执行任务
            handler = self.task_mapping[task_name]
            result = handler(**kwargs)
            
            # 保存结果
            self.task_results[task_id] = result
            self.task_status[task_id] = 'completed'
            
            logger.info(f"任务 {task_id} 完成")
            return result
            
        except Exception as e:
            logger.error(f"任务 {task_id} 失败: {str(e)}")
            self.task_status[task_id] = 'failed'
            raise
    def _handle_split_data(self, processed_data: Dict, train_ratio: float, 
                        val_ratio: float, test_ratio: float) -> Dict:
        """处理数据集划分任务"""
        return self.agents['data'].split_data(
            processed_data, train_ratio, val_ratio, test_ratio
        )        
    def manage_workflow(self, workflow_name: str, input_data: Dict) -> Any:
        """
        管理工作流执行
        
        Args:
            workflow_name: 工作流名称
            input_data: 初始输入数据
            
        Returns:
            工作流最终结果
        """
        logger.info(f"执行工作流: {workflow_name}")
        
        if workflow_name not in self.workflows:
            raise ValueError(f"未知工作流: {workflow_name}")
            
        workflow_tasks = self.workflows[workflow_name]
        workflow_id = f"workflow_{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化工作流状态
        self.task_status[workflow_id] = {
            'status': 'running',
            'current_task': None,
            'completed_tasks': [],
            'results': {}
        }
        
        # 执行工作流任务
        current_data = input_data
        try:
            for task in workflow_tasks:
                self.task_status[workflow_id]['current_task'] = task
                
                # 准备任务参数
                task_params = self._prepare_task_params(task, current_data, 
                                                       self.task_status[workflow_id]['results'])
                
                # 执行任务
                result = self.dispatch_task(task, **task_params)
                
                # 更新结果
                self.task_status[workflow_id]['results'][task] = result             
                # 特别处理train_model任务
                if task == 'train_model':
                    logger.info(f"模型训练完成，路径: {result}")
                self.task_status[workflow_id]['completed_tasks'].append(task)
                
                # 更新当前数据
                current_data = self._update_workflow_data(current_data, task, result)
                
            self.task_status[workflow_id]['status'] = 'completed'
            logger.info(f"工作流 {workflow_id} 完成")
            
            return self.task_status[workflow_id]['results']
            
        except Exception as e:
            logger.error(f"工作流 {workflow_id} 失败: {str(e)}")
            self.task_status[workflow_id]['status'] = 'failed'
            raise
            
    def get_task_status(self, task_id: str) -> Dict:
        """获取任务状态"""
        return self.task_status.get(task_id, {'status': 'not_found'})
        
    def get_task_result(self, task_id: str) -> Any:
        """获取任务结果"""
        return self.task_results.get(task_id)
        
    # 任务处理函数
    def _handle_load_data(self, data_path: str) -> Dict:
        """处理数据加载任务"""
        return self.agents['data'].load_raw_data(data_path)
        
    def _handle_preprocess_data(self, raw_data: Dict) -> Dict:
        """处理数据预处理任务"""
        return self.agents['data'].preprocess_data(raw_data)
        
    def _handle_fuse_features(self, processed_data: Dict) -> np.ndarray:
        """处理特征融合任务"""
        return self.agents['fusion'].fuse_features(processed_data)
        
    
    def _handle_train_model(self, split_data: Dict, train_params: Dict) -> str:
        """处理模型训练任务"""
        return self.agents['model'].train_model(split_data, train_params)
        
    def _handle_predict(self, model_path: str, fused_features: np.ndarray) -> Tuple:
        """处理预测任务"""
        return self.agents['model'].predict(model_path, fused_features)
        
    def _handle_explain(self, model_path: str, fused_features: np.ndarray,
                       predictions: Optional[np.ndarray] = None) -> Dict:
        """处理解释任务"""
        return self.agents['explain'].generate_explanations(
            model_path, fused_features, predictions
        )
        
    def _handle_generate_paper(self, results: Dict, explanations: Dict,
                              metadata: Dict) -> str:
        """处理论文生成任务"""
        return self.agents['paper'].generate_paper(results, explanations, metadata)
        
  
    def _prepare_task_params(self, task_name: str, current_data: Dict,
                            previous_results: Dict) -> Dict:
        """准备任务参数"""
        params = {}
        
        if task_name == 'load_data':
            params['data_path'] = current_data.get('data_path')
            
        elif task_name == 'preprocess_data':
            params['raw_data'] = previous_results.get('load_data', current_data.get('raw_data'))
            
        elif task_name == 'fuse_features':
            # 优先使用split_data，然后是preprocess_data
            params['processed_data'] = previous_results.get('split_data') or \
                                    previous_results.get('preprocess_data') or \
                                    current_data.get('processed_data')
        
        elif task_name == 'split_data':
            params['processed_data'] = previous_results.get('preprocess_data', 
                                                        current_data.get('processed_data'))
            params['train_ratio'] = current_data.get('train_ratio', 0.8)
            params['val_ratio'] = current_data.get('val_ratio', 0.1)
            params['test_ratio'] = current_data.get('test_ratio', 0.1)
            
        elif task_name == 'train_model':
            # 使用划分后的数据
            params['split_data'] = previous_results.get('split_data') or \
                                current_data.get('split_data')
            params['train_params'] = current_data.get('train_params', {})
            
            # 确保传递target_property
            if 'target_property' in current_data:
                params['train_params']['target_property'] = current_data['target_property']
            
        elif task_name == 'predict':
            params['model_path'] = current_data.get('model_path')
            params['fused_features'] = previous_results.get('fuse_features',
                                                        current_data.get('fused_features'))
            
        elif task_name == 'explain':
            params['model_path'] = previous_results.get('train_model',
                                                    current_data.get('model_path'))
            params['fused_features'] = previous_results.get('fuse_features',
                                                        current_data.get('fused_features'))
            # 修复predictions的获取逻辑
            if 'predict' in previous_results:
                pred_result = previous_results['predict']
                if isinstance(pred_result, tuple) and len(pred_result) > 0:
                    params['predictions'] = pred_result[0]
                else:
                    params['predictions'] = None
            else:
                params['predictions'] = None
            
        elif task_name == 'generate_paper':
            params['results'] = current_data.get('results', {})
            params['explanations'] = previous_results.get('explain', {})
            params['metadata'] = current_data.get('metadata', {})
            
        return params
        
    def _update_workflow_data(self, current_data: Dict, task_name: str,
                             result: Any) -> Dict:
        """更新工作流数据"""
        updated_data = current_data.copy()
        
        # 根据任务类型更新数据
        if task_name == 'train_model':
            updated_data['model_path'] = result
        elif task_name == 'predict':
            updated_data['predictions'], updated_data['uncertainties'] = result
            
        return updated_data
        
    async def dispatch_task_async(self, task_name: str, **kwargs) -> Any:
        """异步分发任务"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, 
                                         self.dispatch_task, 
                                         task_name, 
                                         kwargs)
        
    def save_state(self, filepath: str):
        """保存管理器状态"""
        state = {
            'task_status': self.task_status,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, filepath: str):
        """加载管理器状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.task_status = state.get('task_status', {})
        logger.info(f"加载状态: {filepath}")

    def _handle_preprocess_data(self, raw_data: Dict) -> Dict:
        """处理数据预处理任务"""
        return self.agents['data'].preprocess_data(raw_data)

  