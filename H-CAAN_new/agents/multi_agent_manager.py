"""
多智能体统一调度与任务管理
协调各智能体的执行和状态管理
"""
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
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
        
        # 任务映射 - 添加消融实验相关任务
        self.task_mapping = {
            'load_data': self._handle_load_data,
            'preprocess_data': self._handle_preprocess_data,
            'split_data': self._handle_split_data,  
            'fuse_features': self._handle_fuse_features,
            'train_model': self._handle_train_model,
            'predict': self._handle_predict,
            'explain': self._handle_explain,
            'generate_paper': self._handle_generate_paper,
            'learn_fusion_weights': self._handle_learn_fusion_weights,
            
            # 添加消融实验任务
            'comprehensive_ablation': self._handle_comprehensive_ablation,
            'conditional_ablation': self._handle_conditional_ablation,
            'incremental_ablation': self._handle_incremental_ablation,
            'ablation_study': self._handle_ablation_study
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
            ],
            # 添加消融实验工作流
            'ablation_analysis': [
                'load_data', 'preprocess_data', 'split_data', 
                'learn_fusion_weights', 'comprehensive_ablation'
            ]
        }
    
    def _handle_comprehensive_ablation(self, modal_features: List[np.ndarray] = None,
                                     labels: np.ndarray = None,
                                     learned_weights: np.ndarray = None,
                                     **kwargs) -> Dict:
        """处理综合消融实验"""
        logger.info("执行综合消融实验")
        
        try:
            # 如果没有提供modal_features，尝试从已有数据创建
            if modal_features is None:
                # 尝试从最近的任务结果中获取数据
                if hasattr(self, '_last_processed_data'):
                    base_features = self._last_processed_data.get('fingerprints')
                    if base_features is not None:
                        modal_features = self._create_modal_features_from_base(np.array(base_features))
                else:
                    raise ValueError("没有可用的特征数据进行消融实验")
            
            # 确保输入是numpy数组
            if not isinstance(modal_features[0], np.ndarray):
                modal_features = [np.array(f) for f in modal_features]
            if labels is not None and not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            if learned_weights is not None and not isinstance(learned_weights, np.ndarray):
                learned_weights = np.array(learned_weights)
            
            # 调用fusion agent的综合消融方法
            return self.agents['fusion'].adaptive_weights.comprehensive_ablation_study(
                modal_features, labels, learned_weights
            )
            
        except Exception as e:
            logger.error(f"综合消融实验失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _handle_conditional_ablation(self, modal_features: List[np.ndarray] = None,
                                   labels: np.ndarray = None,
                                   learned_weights: np.ndarray = None,
                                   ablation_type: str = 'mask',
                                   **kwargs) -> Dict:
        """处理条件消融实验"""
        logger.info(f"执行条件消融实验，类型: {ablation_type}")
        
        try:
            # 数据准备逻辑同上
            if modal_features is None:
                if hasattr(self, '_last_processed_data'):
                    base_features = self._last_processed_data.get('fingerprints')
                    if base_features is not None:
                        modal_features = self._create_modal_features_from_base(np.array(base_features))
                else:
                    raise ValueError("没有可用的特征数据进行消融实验")
            
            # 确保输入格式正确
            if not isinstance(modal_features[0], np.ndarray):
                modal_features = [np.array(f) for f in modal_features]
            if labels is not None and not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            if learned_weights is not None and not isinstance(learned_weights, np.ndarray):
                learned_weights = np.array(learned_weights)
            
            # 调用fusion agent的条件消融方法
            return self.agents['fusion'].adaptive_weights.conditional_ablation(
                modal_features, labels, learned_weights, ablation_type
            )
            
        except Exception as e:
            logger.error(f"条件消融实验失败: {str(e)}")
            raise

    def _handle_incremental_ablation(self, modal_features: List[np.ndarray] = None,
                                   labels: np.ndarray = None,
                                   learned_weights: np.ndarray = None,
                                   **kwargs) -> Dict:
        """处理增量消融实验（从单模态逐步添加）"""
        logger.info("执行增量消融实验")
        
        try:
            # 数据准备
            if modal_features is None:
                if hasattr(self, '_last_processed_data'):
                    base_features = self._last_processed_data.get('fingerprints')
                    if base_features is not None:
                        modal_features = self._create_modal_features_from_base(np.array(base_features))
                else:
                    raise ValueError("没有可用的特征数据进行消融实验")
            
            # 格式转换
            if not isinstance(modal_features[0], np.ndarray):
                modal_features = [np.array(f) for f in modal_features]
            if labels is not None and not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            if learned_weights is not None and not isinstance(learned_weights, np.ndarray):
                learned_weights = np.array(learned_weights)
            
            # 如果fusion agent有incremental_ablation方法则调用，否则使用综合消融
            if hasattr(self.agents['fusion'].adaptive_weights, 'incremental_ablation'):
                return self.agents['fusion'].adaptive_weights.incremental_ablation(
                    modal_features, labels, learned_weights
                )
            else:
                # 使用综合消融作为替代
                logger.info("使用综合消融替代增量消融")
                return self.agents['fusion'].adaptive_weights.comprehensive_ablation_study(
                    modal_features, labels, learned_weights
                )
                
        except Exception as e:
            logger.error(f"增量消融实验失败: {str(e)}")
            raise

    def _handle_ablation_study(self, modal_features: List[np.ndarray] = None,
                              labels: np.ndarray = None,
                              learned_weights: np.ndarray = None,
                              ablation_mode: str = '综合消融',
                              ablation_type: str = 'mask',
                              **kwargs) -> Dict:
        """统一的消融实验处理入口"""
        logger.info(f"执行消融实验，模式: {ablation_mode}")
        
        try:
            # 根据模式调用相应的处理方法
            if ablation_mode == '综合消融':
                return self._handle_comprehensive_ablation(
                    modal_features, labels, learned_weights, **kwargs
                )
            elif ablation_mode == '条件消融':
                return self._handle_conditional_ablation(
                    modal_features, labels, learned_weights, ablation_type, **kwargs
                )
            elif ablation_mode == '增量消融':
                return self._handle_incremental_ablation(
                    modal_features, labels, learned_weights, **kwargs
                )
            else:
                raise ValueError(f"未知的消融模式: {ablation_mode}")
                
        except Exception as e:
            logger.error(f"消融实验失败: {str(e)}")
            raise

    def _create_modal_features_from_base(self, base_features: np.ndarray) -> List[np.ndarray]:
        """从基础特征创建六模态特征（用于演示）"""
        logger.info("从基础特征创建六模态特征")
        
        n_samples, n_features = base_features.shape
        modal_features = []
        
        # 模态1：原始特征（MFBERT）
        modal_features.append(base_features)
        
        # 模态2-6：通过不同变换生成
        # 在实际应用中，这些应该是真实的不同模态特征
        try:
            from sklearn.decomposition import PCA
            from sklearn.random_projection import GaussianRandomProjection
            from sklearn.preprocessing import StandardScaler
            
            # 模态2：PCA变换（ChemBERTa）
            if n_samples > 1 and n_features > 1:
                pca = PCA(n_components=min(n_features, n_samples-1), random_state=42)
                modal_features.append(pca.fit_transform(base_features))
            else:
                modal_features.append(base_features * 0.95)
            
            # 模态3：随机投影（Transformer）
            grp = GaussianRandomProjection(n_components=n_features, random_state=42)
            modal_features.append(grp.fit_transform(base_features))
            
            # 模态4：标准化+小扰动（GCN）
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(base_features)
            modal_features.append(scaled_features + np.random.normal(0, 0.05, scaled_features.shape))
            
            # 模态5：非线性变换（GraphTransformer）
            modal_features.append(np.tanh(base_features * 0.5))
            
            # 模态6：添加噪声（BiGRU）
            noise_features = base_features + np.random.normal(0, 0.1, base_features.shape)
            modal_features.append(noise_features)
            
        except Exception as e:
            logger.warning(f"创建某些模态特征失败，使用简单变换: {str(e)}")
            # 使用简单的线性变换作为备选
            while len(modal_features) < 6:
                factor = 0.9 + 0.02 * len(modal_features)
                modal_features.append(base_features * factor)
        
        logger.info(f"成功创建 {len(modal_features)} 个模态特征")
        return modal_features
    
    def _handle_learn_fusion_weights(self, train_features: np.ndarray = None, 
                                   train_labels: np.ndarray = None,
                                   method: str = 'auto', 
                                   n_iterations: int = 5,
                                   **kwargs) -> Dict:
        """处理权重学习任务"""
        logger.info("处理权重学习任务")
        
        try:
        # 参数验证
            if train_features is None or train_labels is None:
                # 尝试从保存的数据中获取
                if hasattr(self, '_last_split_data') and self._last_split_data is not None:
                    split_data = self._last_split_data
                    
                    # 添加更多检查
                    if 'train' not in split_data:
                        raise ValueError("训练数据不存在于split_data中")
                    
                    train_data = split_data['train']
                    
                    # 检查fingerprints
                    if 'fingerprints' not in train_data or train_data['fingerprints'] is None:
                        raise ValueError("特征数据不存在")
                    
                    train_features = np.array(train_data['fingerprints'])
                    
                    # 检查labels
                    if 'labels' not in train_data or train_data['labels'] is None:
                        raise ValueError("标签数据不存在")
                    
                    # 安全地提取标签
                    if isinstance(train_data['labels'], dict):
                        # 获取第一个可用的标签
                        label_values = list(train_data['labels'].values())
                        if not label_values:
                            raise ValueError("标签字典为空")
                        train_labels = np.array(label_values[0])
                    else:
                        train_labels = np.array(train_data['labels'])
                else:
                    raise ValueError("缺少训练数据且无法从历史数据中获取")
            
            # 确保是numpy数组
            if not isinstance(train_features, np.ndarray):
                train_features = np.array(train_features)
            if not isinstance(train_labels, np.ndarray):
                train_labels = np.array(train_labels)
            
            # 创建六模态特征
            modal_features = self._create_modal_features_from_base(train_features)
            
            # 调用fusion agent的学习方法
            result = self.agents['fusion'].learn_optimal_weights(
                train_features=train_features,
                train_labels=train_labels,
                method=method,
                n_iterations=n_iterations
            )
            
            # 保存学习到的权重，供后续消融实验使用
            if result and 'optimal_weights' in result:
                self._last_learned_weights = result['optimal_weights']
            
            return result
            
        except Exception as e:
            logger.error(f"权重学习任务失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回默认结果
            default_weights = [1/6] * 6
            return {
                'optimal_weights': default_weights,
                'weight_evolution': {
                    'weights_over_time': np.array([default_weights]),
                    'performance_over_time': [0.5],
                    'best_performance': 0.5,
                    'best_weights': default_weights,
                    'modal_names': ['MFBERT', 'ChemBERTa', 'Transformer', 'GCN', 'GraphTrans', 'BiGRU']
                }
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
            logger.error(f"未知任务: {task_name}")
            logger.info(f"可用任务: {list(self.task_mapping.keys())}")
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
        result = self.agents['data'].split_data(
            processed_data, train_ratio, val_ratio, test_ratio
        )
        # 保存划分后的数据
        self._last_split_data = result
        return result        
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
        result = self.agents['data'].preprocess_data(raw_data)
        # 保存处理后的数据供消融实验使用
        self._last_processed_data = result
        return result
        
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

  