# streamlit_ui/api.py
"""
统一API调用层，连接UI和agents。
提供与后端智能体通信的标准化接口。
"""

import streamlit as st
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
import asyncio
import traceback
from functools import wraps
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 将项目根目录添加到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 导入状态管理器
try:
    from .state import state_manager
except ImportError:
    from state import state_manager

# 创建模拟Agent类
class MockAgent:
    """模拟Agent，在真实Agent不可用时使用"""
    def __init__(self, name="MockAgent"):
        self.name = name
        logger.warning(f"{name} is using mock implementation")
    
    def __getattr__(self, method_name):
        def mock_method(*args, **kwargs):
            logger.warning(f"Mock {self.name}.{method_name} called")
            return {
                "success": False,
                "message": f"{self.name} is not available",
                "data": None
            }
        return mock_method

# 尝试导入真实的Agent类
agents_available = False
try:
    from agents.agent_manager import AgentManager
    from agents.research_agent import ResearchAgent
    from agents.data_agent import DataAgent
    from agents.model_agent import ModelAgent
    from agents.evaluation_agent import EvaluationAgent
    from agents.writing_agent import WritingAgent
    agents_available = True
    logger.info("Successfully imported agent modules")
except Exception as e:
    logger.error(f"Failed to import agent modules: {str(e)}")
    # 使用模拟类
    AgentManager = lambda: MockAgent("AgentManager")
    ResearchAgent = lambda: MockAgent("ResearchAgent")
    DataAgent = lambda: MockAgent("DataAgent")
    ModelAgent = lambda: MockAgent("ModelAgent")
    EvaluationAgent = lambda: MockAgent("EvaluationAgent")
    WritingAgent = lambda: MockAgent("WritingAgent")

class APIClient:
    """
    统一API客户端，封装与后端智能体的通信。
    提供线程安全且支持异步的API。
    """
    
    def __init__(self):
        """初始化API客户端"""
        self._agent_manager = None
        self._agents = {}
        self._agents_available = agents_available
        self._initialize_agents()
        
        # 记录API调用历史
        self._api_call_history = []
        
        # 异步任务状态
        self._async_tasks = {}
        
        # 锁定以进行线程安全操作
        self._lock = threading.RLock()
    
    def _initialize_agents(self):
        """初始化所有代理"""
        logger.info("Initializing agents...")
        
        try:
            # 初始化AgentManager
            self._agent_manager = AgentManager()
            
            # 初始化各个Agent
            # 先尝试无参数初始化，如果失败则尝试传入agent_manager
            agents_config = [
                ('research', ResearchAgent),
                ('data', DataAgent),
                ('model', ModelAgent),
                ('evaluation', EvaluationAgent),
                ('writing', WritingAgent)
            ]
            
            for agent_name, AgentClass in agents_config:
                try:
                    # 尝试无参数初始化
                    self._agents[agent_name] = AgentClass()
                    logger.info(f"Initialized {agent_name} without parameters")
                except TypeError:
                    try:
                        # 尝试传入agent_manager
                        self._agents[agent_name] = AgentClass(self._agent_manager)
                        logger.info(f"Initialized {agent_name} with agent_manager")
                    except Exception as e:
                        logger.error(f"Failed to initialize {agent_name}: {e}")
                        self._agents[agent_name] = MockAgent(agent_name)
            
            # 尝试注册agents（如果agent_manager支持）
            if hasattr(self._agent_manager, 'register_agents'):
                try:
                    self._agent_manager.register_agents(self._agents)
                    logger.info("Successfully registered agents with manager")
                except Exception as e:
                    logger.warning(f"Failed to register agents: {e}")
                    
        except Exception as e:
            logger.error(f"Critical error during agent initialization: {e}")
            # 降级到全部使用模拟agents
            self._agents = {
                'research': MockAgent('research'),
                'data': MockAgent('data'),
                'model': MockAgent('model'),
                'evaluation': MockAgent('evaluation'),
                'writing': MockAgent('writing')
            }
            
        logger.info(f"Agent initialization complete. Available agents: {list(self._agents.keys())}")
    
    def _safe_agent_call(self, agent_name: str, method_name: str, *args, **kwargs):
        """安全地调用agent方法，包含错误处理"""
        try:
            agent = self._agents.get(agent_name)
            if not agent:
                return {
                    "success": False,
                    "message": f"Agent '{agent_name}' not found",
                    "data": None
                }
            
            method = getattr(agent, method_name, None)
            if not method:
                return {
                    "success": False,
                    "message": f"Method '{method_name}' not found in {agent_name}",
                    "data": None
                }
            
            result = method(*args, **kwargs)
            
            # 确保返回格式一致
            if not isinstance(result, dict):
                result = {"success": True, "data": result}
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling {agent_name}.{method_name}: {e}")
            return {
                "success": False,
                "message": str(e),
                "data": None,
                "error": traceback.format_exc()
            }
    
    def _execute_in_thread(self, func, *args, **kwargs):
        """
        在单独的线程中执行函数
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 传递给函数的参数
            
        Returns:
            函数的结果
        """
        result = [None]
        error = [None]
        
        def thread_func():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()
        
        if error[0]:
            raise error[0]
        
        return result[0]
    
    def _log_api_call(self, function_name, args, kwargs, result=None, error=None, execution_time=None):
        """
        记录API调用
        
        Args:
            function_name: 调用的函数名
            args, kwargs: 传递的参数
            result: 调用结果
            error: 发生的错误
            execution_time: 执行时间（秒）
        """
        with self._lock:
            self._api_call_history.append({
                'timestamp': time.time(),
                'function': function_name,
                'args': str(args)[:100],  # 限制长度
                'kwargs': str(kwargs)[:100],
                'result': str(result)[:100] if result is not None else None,
                'error': str(error) if error is not None else None,
                'execution_time': execution_time
            })
            
            # 保持历史记录在合理范围内
            if len(self._api_call_history) > 1000:
                self._api_call_history = self._api_call_history[-500:]
    
    def execute_async(self, func_name, *args, **kwargs):
        """
        异步执行API调用
        
        Args:
            func_name: 要调用的API函数名称
            *args, **kwargs: 传递给函数的参数
            
        Returns:
            任务ID
        """
        if not hasattr(self, func_name):
            raise ValueError(f"Unknown API function: {func_name}")
        
        func = getattr(self, func_name)
        task_id = f"task_{time.time()}_{func_name}"
        
        def async_wrapper():
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                with self._lock:
                    self._async_tasks[task_id] = {
                        'status': 'completed',
                        'result': result,
                        'error': None,
                        'execution_time': execution_time
                    }
                    
                self._log_api_call(func_name, args, kwargs, result, None, execution_time)
                return result
            except Exception as e:
                with self._lock:
                    self._async_tasks[task_id] = {
                        'status': 'error',
                        'result': None,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                self._log_api_call(func_name, args, kwargs, None, e, None)
        
        with self._lock:
            self._async_tasks[task_id] = {'status': 'running'}
            
        thread = threading.Thread(target=async_wrapper)
        thread.start()
        
        return task_id
    
    def get_task_status(self, task_id):
        """
        获取异步任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典
        """
        with self._lock:
            return self._async_tasks.get(task_id, {'status': 'not_found'})
    
    def get_api_call_history(self):
        """
        获取API调用历史
        
        Returns:
            API调用历史列表
        """
        with self._lock:
            return self._api_call_history.copy()
    
    # ==================== 模型配置API ====================
    
    def get_model_templates(self):
        """
        获取预定义的模型模板
        
        Returns:
            模型模板列表
        """
        templates = [
            {
                "name": "H-CAAN Base",
                "description": "基础H-CAAN模型，适用于一般分子属性预测任务",
                "config": {
                    "architecture": "h_caan_base",
                    "modal_configs": {
                        "smiles": {
                            "encoder": "transformer",
                            "embedding_dim": 768,
                            "num_layers": 6,
                            "num_heads": 8
                        },
                        "ecfp": {
                            "encoder": "bigru",
                            "hidden_dim": 512,
                            "num_layers": 2,
                            "bidirectional": True
                        },
                        "graph": {
                            "encoder": "gcn",
                            "hidden_dims": [256, 512, 768],
                            "num_layers": 3
                        }
                    },
                    "fusion": {
                        "method": "hierarchical_attention",
                        "hidden_dim": 512,
                        "num_heads": 8
                    },
                    "training": {
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "epochs": 100,
                        "early_stopping_patience": 10
                    }
                }
            },
            {
                "name": "H-CAAN Large",
                "description": "大型H-CAAN模型，提供更高的准确性但需要更多计算资源",
                "config": {
                    "architecture": "h_caan_large",
                    "modal_configs": {
                        "smiles": {
                            "encoder": "transformer",
                            "embedding_dim": 1024,
                            "num_layers": 12,
                            "num_heads": 16
                        },
                        "ecfp": {
                            "encoder": "bigru",
                            "hidden_dim": 768,
                            "num_layers": 3,
                            "bidirectional": True
                        },
                        "graph": {
                            "encoder": "gcn",
                            "hidden_dims": [512, 768, 1024],
                            "num_layers": 4
                        }
                    },
                    "fusion": {
                        "method": "hierarchical_attention",
                        "hidden_dim": 768,
                        "num_heads": 16
                    },
                    "training": {
                        "batch_size": 16,
                        "learning_rate": 0.0005,
                        "epochs": 150,
                        "early_stopping_patience": 15
                    }
                }
            },
            {
                "name": "H-CAAN Fast",
                "description": "轻量级H-CAAN模型，适用于快速原型设计和资源受限环境",
                "config": {
                    "architecture": "h_caan_fast",
                    "modal_configs": {
                        "smiles": {
                            "encoder": "transformer",
                            "embedding_dim": 256,
                            "num_layers": 3,
                            "num_heads": 4
                        },
                        "ecfp": {
                            "encoder": "bigru",
                            "hidden_dim": 256,
                            "num_layers": 1,
                            "bidirectional": True
                        },
                        "graph": {
                            "encoder": "gcn",
                            "hidden_dims": [128, 256],
                            "num_layers": 2
                        }
                    },
                    "fusion": {
                        "method": "simple_attention",
                        "hidden_dim": 256,
                        "num_heads": 4
                    },
                    "training": {
                        "batch_size": 64,
                        "learning_rate": 0.002,
                        "epochs": 50,
                        "early_stopping_patience": 5
                    }
                }
            },
            {
                "name": "Custom",
                "description": "自定义模型配置，允许完全控制所有参数",
                "config": {}
            }
        ]
        
        return templates
    
    def configure_model(self, config: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        配置模型
        
        Args:
            config: 模型配置字典
            
        Returns:
            (success, message, validated_config)
        """
        try:
            # 验证配置
            if not config:
                return False, "模型配置不能为空", None
            
            # 检查必要的配置项
            required_fields = ['architecture', 'modal_configs', 'fusion', 'training']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                return False, f"缺少必要的配置项: {', '.join(missing_fields)}", None
            
            # 验证模态配置
            modal_configs = config.get('modal_configs', {})
            required_modals = ['smiles', 'ecfp', 'graph']
            missing_modals = [modal for modal in required_modals if modal not in modal_configs]
            
            if missing_modals:
                return False, f"缺少必要的模态配置: {', '.join(missing_modals)}", None
            
            # 保存配置到状态
            state_manager.set('model_config', config)
            state_manager.set('model_configured', True)
            
            # 记录API调用
            self._log_api_call('configure_model', (config,), {}, config, None, 0)
            
            return True, "模型配置成功", config
            
        except Exception as e:
            self._log_api_call('configure_model', (config,), {}, None, e, 0)
            return False, f"配置模型时出错: {str(e)}", None
    
    # 数据处理API
    
    def process_molecular_data(self, data_source, source_type='file', options=None):
        """
        处理分子数据
        
        Args:
            data_source: 数据源（文件路径、DataFrame或字典）
            source_type: 源类型（'file', 'dataframe', 'dict'）
            options: 处理选项字典
            
        Returns:
            处理结果字典
        """
        try:
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('data', 'process_molecular_data', 
                                         data_source, source_type, **options)
            execution_time = time.time() - start_time
            
            # 更新状态
            if result.get('success') and result.get('data'):
                data = result['data']
                if 'molecules' in data:
                    state_manager.set('molecular_data', data)
                    
                if 'features' in data:
                    processed_features = state_manager.get('processed_features', {})
                    for feature_type, feature_data in data['features'].items():
                        processed_features[feature_type] = feature_data
                    state_manager.set('processed_features', processed_features)
            
            self._log_api_call('process_molecular_data', 
                             (data_source, source_type), options, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('process_molecular_data', 
                             (data_source, source_type), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    # ... 其余方法继续使用 _safe_agent_call 包装 ...
    
    def generate_fingerprints(self, molecules, fingerprint_type='ecfp', options=None):
        """
        为分子生成指纹
        
        Args:
            molecules: 分子列表
            fingerprint_type: 指纹类型
            options: 生成选项
            
        Returns:
            指纹数组
        """
        try:
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('data', 'generate_fingerprints',
                                         molecules, fingerprint_type, **options)
            execution_time = time.time() - start_time
            
            # 更新状态
            if result.get('success') and result.get('data'):
                processed_features = state_manager.get('processed_features', {})
                processed_features[fingerprint_type] = result['data']
                state_manager.set('processed_features', processed_features)
            
            self._log_api_call('generate_fingerprints', 
                             (molecules, fingerprint_type), options, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('generate_fingerprints', 
                             (molecules, fingerprint_type), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def split_dataset(self, data, split_method='random', test_size=0.2, val_size=0.1, options=None):
        """
        拆分数据集
        
        Args:
            data: 要拆分的数据
            split_method: 拆分方法
            test_size: 测试集比例
            val_size: 验证集比例
            options: 其他选项
            
        Returns:
            拆分的数据集字典
        """
        try:
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('data', 'split_dataset',
                                         data, split_method, test_size, val_size, **options)
            execution_time = time.time() - start_time
            
            # 更新状态
            if result.get('success') and result.get('data'):
                dataset_name = options.get('name', 'current_dataset')
                state_manager.set(f'dataset_{dataset_name}', result['data'])
                state_manager.set('dataset_split_info', {
                    'split_method': split_method,
                    'test_size': test_size,
                    'val_size': val_size
                })
            
            self._log_api_call('split_dataset', 
                             (data, split_method, test_size, val_size), options, 
                             result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('split_dataset', 
                             (data, split_method, test_size, val_size), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    # 模型训练API
    
    def build_model(self, model_config=None):
        """
        构建模型
        
        Args:
            model_config: 模型配置
            
        Returns:
            构建的模型
        """
        try:
            if model_config is None:
                model_config = state_manager.get('model_config')
                
            start_time = time.time()
            result = self._safe_agent_call('model', 'build_model', model_config)
            execution_time = time.time() - start_time
            
            self._log_api_call('build_model', (model_config,), {}, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('build_model', (model_config,), {}, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def train_model(self, model, train_data, val_data=None, config=None):
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_data: 训练数据
            val_data: 验证数据
            config: 训练配置
            
        Returns:
            训练历史
        """
        try:
            if config is None:
                model_config = state_manager.get('model_config', {})
                config = model_config.get('training', {})
                
            # 更新训练状态
            training_status = {
                'is_training': True,
                'current_epoch': 0,
                'total_epochs': config.get('epochs', 100),
                'start_time': time.time(),
                'train_loss': [],
                'val_loss': []
            }
            state_manager.set('training_status', training_status)
            
            # 定义回调函数来更新训练状态
            def update_status_callback(epoch, metrics):
                training_status = state_manager.get('training_status', {})
                training_status.update({
                    'current_epoch': epoch + 1,
                    'train_loss': metrics.get('train_loss', []),
                    'val_loss': metrics.get('val_loss', []),
                    'metrics': metrics
                })
                state_manager.set('training_status', training_status)
            
            start_time = time.time()
            
            # 调用agent的train_model方法
            result = self._safe_agent_call('model', 'train_model',
                                         model, train_data, val_data, config,
                                         callbacks=[update_status_callback])
            
            execution_time = time.time() - start_time
            
            # 更新训练状态
            training_status = state_manager.get('training_status', {})
            training_status.update({
                'is_training': False,
                'end_time': time.time(),
                'best_epoch': result.get('data', {}).get('best_epoch', 0) if result.get('success') else 0
            })
            state_manager.set('training_status', training_status)
            
            # 保存训练结果
            if result.get('success'):
                state_manager.set('training_results', result.get('data'))
                state_manager.set('model_trained', True)
            
            self._log_api_call('train_model', 
                             (model, train_data, val_data), config, result, None, execution_time)
            return result
            
        except Exception as e:
            # 更新训练状态
            training_status = state_manager.get('training_status', {})
            training_status.update({
                'is_training': False,
                'end_time': time.time(),
                'error': str(e)
            })
            state_manager.set('training_status', training_status)
            
            self._log_api_call('train_model', 
                             (model, train_data, val_data), config, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def evaluate_model(self, model, test_data, metrics=None):
        """
        评估模型
        
        Args:
            model: 要评估的模型
            test_data: 测试数据
            metrics: 要使用的指标列表
            
        Returns:
            评估结果字典
        """
        try:
            start_time = time.time()
            result = self._safe_agent_call('evaluation', 'evaluate_model',
                                         model, test_data, metrics)
            execution_time = time.time() - start_time
            
            # 更新模型结果
            if result.get('success') and result.get('data'):
                model_results = state_manager.get('model_results', {})
                if 'test_metrics' not in model_results:
                    model_results['test_metrics'] = {}
                model_results['test_metrics'].update(result['data'])
                state_manager.set('model_results', model_results)
            
            self._log_api_call('evaluate_model', 
                             (model, test_data, metrics), {}, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('evaluate_model', 
                             (model, test_data, metrics), {}, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def analyze_model(self, model, data, analysis_types=None):
        """
        分析模型
        
        Args:
            model: 要分析的模型
            data: 用于分析的数据
            analysis_types: 要执行的分析类型列表
            
        Returns:
            分析结果字典
        """
        try:
            if analysis_types is None:
                analysis_types = ['feature_importance', 'error_analysis', 'model_complexity']
                
            start_time = time.time()
            result = self._safe_agent_call('evaluation', 'analyze_model',
                                         model, data, analysis_types)
            execution_time = time.time() - start_time
            
            # 更新模型结果
            if result.get('success') and result.get('data'):
                model_results = state_manager.get('model_results', {})
                if 'model_analysis' not in model_results:
                    model_results['model_analysis'] = {}
                model_results['model_analysis'].update(result['data'])
                state_manager.set('model_results', model_results)
            
            self._log_api_call('analyze_model', 
                             (model, data, analysis_types), {}, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('analyze_model', 
                             (model, data, analysis_types), {}, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    # 论文生成API
    
    def generate_paper_outline(self, results, paper_type='research'):
        """
        生成论文大纲
        
        Args:
            results: 研究结果
            paper_type: 论文类型
            
        Returns:
            论文大纲字典
        """
        try:
            start_time = time.time()
            result = self._safe_agent_call('writing', 'generate_paper_outline',
                                         results, paper_type)
            execution_time = time.time() - start_time
            
            # 更新论文生成状态
            if result.get('success') and result.get('data'):
                paper_generation = state_manager.get('paper_generation', {})
                paper_generation.update({
                    'outline': result['data'],
                    'status': 'outline_generated'
                })
                state_manager.set('paper_generation', paper_generation)
            
            self._log_api_call('generate_paper_outline', 
                             (results, paper_type), {}, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('generate_paper_outline', 
                             (results, paper_type), {}, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def generate_paper_section(self, section_name, outline, results=None, model_config=None, options=None):
        """
        生成论文章节
        
        Args:
            section_name: 章节名称
            outline: 论文大纲
            results: 研究结果
            model_config: 模型配置
            options: 生成选项
            
        Returns:
            生成的章节内容
        """
        try:
            if options is None:
                options = {}
                
            if results is None:
                results = state_manager.get('model_results', {})
                
            if model_config is None:
                model_config = state_manager.get('model_config')
                
            start_time = time.time()
            result = self._safe_agent_call('writing', 'generate_paper_section',
                                         section_name, outline, results, model_config, **options)
            execution_time = time.time() - start_time
            
            # 更新论文生成状态
            if result.get('success') and result.get('data'):
                paper_generation = state_manager.get('paper_generation', {})
                if 'sections' not in paper_generation:
                    paper_generation['sections'] = {}
                    
                paper_generation['sections'][section_name] = result['data']
                state_manager.set('paper_generation', paper_generation)
            
            self._log_api_call('generate_paper_section', 
                             (section_name, outline, results, model_config), options, 
                             result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('generate_paper_section', 
                             (section_name, outline, results, model_config), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def generate_paper_figures(self, results, figure_types=None, options=None):
        """
        生成论文图表
        
        Args:
            results: 研究结果
            figure_types: 图表类型列表
            options: 生成选项
            
        Returns:
            生成的图表列表
        """
        try:
            if figure_types is None:
                figure_types = ['performance', 'correlation', 'distribution', 'model_architecture']
                
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('writing', 'generate_paper_figures',
                                         results, figure_types, **options)
            execution_time = time.time() - start_time
            
            # 更新论文生成状态
            if result.get('success') and result.get('data'):
                paper_generation = state_manager.get('paper_generation', {})
                paper_generation['figures'] = result['data']
                state_manager.set('paper_generation', paper_generation)
            
            self._log_api_call('generate_paper_figures', 
                             (results, figure_types), options, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('generate_paper_figures', 
                             (results, figure_types), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def generate_full_paper(self, outline=None, results=None, model_config=None, options=None):
        """
        生成完整论文
        
        Args:
            outline: 论文大纲
            results: 研究结果
            model_config: 模型配置
            options: 生成选项
            
        Returns:
            生成的完整论文
        """
        try:
            if outline is None:
                outline = state_manager.get('paper_generation', {}).get('outline')
                
            if results is None:
                results = state_manager.get('model_results', {})
                
            if model_config is None:
                model_config = state_manager.get('model_config')
                
            if options is None:
                options = {}
                
            # 更新论文生成状态
            paper_generation = state_manager.get('paper_generation', {})
            paper_generation['status'] = 'generating'
            state_manager.set('paper_generation', paper_generation)
            
            start_time = time.time()
            result = self._safe_agent_call('writing', 'generate_full_paper',
                                         outline, results, model_config, **options)
            execution_time = time.time() - start_time
            
            # 更新论文生成状态
            if result.get('success') and result.get('data'):
                paper_generation = state_manager.get('paper_generation', {})
                paper_generation.update({
                    **result['data'],
                    'status': 'completed'
                })
                state_manager.set('paper_generation', paper_generation)
            else:
                paper_generation = state_manager.get('paper_generation', {})
                paper_generation.update({
                    'status': 'error',
                    'error': result.get('message', 'Unknown error')
                })
                state_manager.set('paper_generation', paper_generation)
            
            self._log_api_call('generate_full_paper', 
                             (outline, results, model_config), options, result, None, execution_time)
            return result
            
        except Exception as e:
            # 更新论文生成状态
            paper_generation = state_manager.get('paper_generation', {})
            paper_generation.update({
                'status': 'error',
                'error': str(e)
            })
            state_manager.set('paper_generation', paper_generation)
            
            self._log_api_call('generate_full_paper', 
                             (outline, results, model_config), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    # 工具和研究API
    
    def search_literature(self, query, num_results=10, options=None):
        """
        搜索相关文献
        
        Args:
            query: 搜索查询
            num_results: 返回结果数
            options: 搜索选项
            
        Returns:
            搜索结果列表
        """
        try:
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('research', 'search_literature',
                                         query, num_results, **options)
            execution_time = time.time() - start_time
            
            self._log_api_call('search_literature', 
                             (query, num_results), options, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('search_literature', 
                             (query, num_results), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def analyze_smiles(self, smiles, analysis_types=None):
        """
        分析SMILES结构
        
        Args:
            smiles: SMILES字符串
            analysis_types: 分析类型列表
            
        Returns:
            分析结果字典
        """
        try:
            if analysis_types is None:
                analysis_types = ['properties', 'fragments', 'descriptors']
                
            start_time = time.time()
            result = self._safe_agent_call('data', 'analyze_smiles',
                                         smiles, analysis_types)
            execution_time = time.time() - start_time
            
            self._log_api_call('analyze_smiles', 
                             (smiles, analysis_types), {}, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('analyze_smiles', 
                             (smiles, analysis_types), {}, None, e, None)
            return {"success": False, "message": str(e), "data": None}
    
    def visualize_chemical_space(self, molecules, method='tsne', features=None, options=None):
        """
        可视化化学空间
        
        Args:
            molecules: 分子列表
            method: 降维方法
            features: 特征类型
            options: 可视化选项
            
        Returns:
            可视化结果字典
        """
        try:
            if options is None:
                options = {}
                
            start_time = time.time()
            result = self._safe_agent_call('data', 'visualize_chemical_space',
                                         molecules, method, features, **options)
            execution_time = time.time() - start_time
            
            self._log_api_call('visualize_chemical_space', 
                             (molecules, method, features), options, result, None, execution_time)
            return result
            
        except Exception as e:
            self._log_api_call('visualize_chemical_space', 
                             (molecules, method, features), options, None, e, None)
            return {"success": False, "message": str(e), "data": None}

# 创建全局API客户端实例
try:
    api_client = APIClient()
    logger.info("API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize API client: {e}")
    api_client = None

# 导出便捷函数
def configure_model(config: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """配置模型的便捷函数"""
    if api_client:
        return api_client.configure_model(config)
    return False, "API客户端未初始化", None

def get_model_templates() -> List[Dict[str, Any]]:
    """获取模型模板的便捷函数"""
    if api_client:
        return api_client.get_model_templates()
    return []

def process_molecular_data(data_source, source_type='file', options=None):
    """处理分子数据的便捷函数"""
    if api_client:
        return api_client.process_molecular_data(data_source, source_type, options)
    return {"success": False, "message": "API客户端未初始化", "data": None}

def train_model(model, train_data, val_data=None, config=None):
    """训练模型的便捷函数"""
    if api_client:
        return api_client.train_model(model, train_data, val_data, config)
    return {"success": False, "message": "API客户端未初始化", "data": None}

def evaluate_model(model, test_data, metrics=None):
    """评估模型的便捷函数"""
    if api_client:
        return api_client.evaluate_model(model, test_data, metrics)
    return {"success": False, "message": "API客户端未初始化", "data": None}

def generate_full_paper(outline=None, results=None, model_config=None, options=None):
    """生成完整论文的便捷函数"""
    if api_client:
        return api_client.generate_full_paper(outline, results, model_config, options)
    return {"success": False, "message": "API客户端未初始化", "data": None}

# 导出
__all__ = [
    'api_client',
    'APIClient',
    'configure_model',
    'get_model_templates',
    'process_molecular_data',
    'train_model',
    'evaluate_model',
    'generate_full_paper'
]