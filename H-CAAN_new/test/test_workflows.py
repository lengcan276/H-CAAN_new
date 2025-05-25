"""
工作流测试
"""
import unittest
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.task_chain import TaskChain
from workflows.router import WorkflowRouter

class TestTaskChain(unittest.TestCase):
    """测试任务链"""
    
    def setUp(self):
        self.task_chain = TaskChain()
        
    def test_get_chain(self):
        """测试获取任务链"""
        # 获取数据处理链
        chain = self.task_chain.get_chain('data_processing')
        
        # 验证结果
        self.assertIsInstance(chain, list)
        self.assertGreater(len(chain), 0)
        self.assertEqual(chain[0]['name'], 'load_data')
        self.assertEqual(chain[1]['name'], 'preprocess_data')
        
    def test_validate_chain(self):
        """测试任务链验证"""
        # 有效任务链
        valid_chain = [
            {
                'name': 'load_data',
                'required_inputs': ['data_path'],
                'outputs': ['raw_data']
            },
            {
                'name': 'preprocess_data',
                'required_inputs': ['raw_data'],
                'outputs': ['processed_data']
            }
        ]
        
        self.assertTrue(self.task_chain.validate_chain(valid_chain))
        
        # 无效任务链（缺少依赖）
        invalid_chain = [
            {
                'name': 'preprocess_data',
                'required_inputs': ['raw_data'],
                'outputs': ['processed_data']
            }
        ]
        
        self.assertFalse(self.task_chain.validate_chain(invalid_chain))
        
    def test_optimize_chain(self):
        """测试任务链优化"""
        # 创建可并行的任务链
        chain = [
            {
                'name': 'task1',
                'required_inputs': ['input'],
                'outputs': ['output1']
            },
            {
                'name': 'task2',
                'required_inputs': ['input'],
                'outputs': ['output2']
            },
            {
                'name': 'task3',
                'required_inputs': ['output1', 'output2'],
                'outputs': ['final_output']
            }
        ]
        
        # 优化
        optimized = self.task_chain.optimize_chain(chain)
        
        # 验证结果
        self.assertEqual(len(optimized), 2)  # 两个批次
        self.assertEqual(len(optimized[0]), 2)  # 第一批次有两个并行任务
        self.assertEqual(len(optimized[1]), 1)  # 第二批次有一个任务
        
    def test_create_custom_chain(self):
        """测试创建自定义任务链"""
        # 创建自定义链
        tasks = ['load', 'preprocess', 'fuse', 'train']
        chain = self.task_chain.create_custom_chain(tasks)
        
        # 验证结果
        self.assertEqual(len(chain), 4)
        self.assertEqual(chain[0]['name'], 'load_data')
        self.assertEqual(chain[3]['name'], 'train_model')
        
    def test_dependencies(self):
        """测试任务依赖关系"""
        deps = self.task_chain._define_dependencies()
        
        # 验证依赖关系
        self.assertIn('preprocess_data', deps)
        self.assertIn('load_data', deps['preprocess_data'])
        self.assertIn('fuse_features', deps)
        self.assertIn('preprocess_data', deps['fuse_features'])

class TestWorkflowRouter(unittest.TestCase):
    """测试工作流路由器"""
    
    def setUp(self):
        self.router = WorkflowRouter()
        
    def test_route_request(self):
        """测试请求路由"""
        # 测试获取状态请求
        request = {
            'type': 'get_status',
            'params': {'workflow_id': 'test_workflow'}
        }
        
        result = self.router.route_request(request)
        
        # 验证结果
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'not_found')
        
    def test_execute_chain(self):
        """测试执行任务链"""
        # 准备测试数据
        import pandas as pd
        test_data = pd.DataFrame({
            'smiles': ['CCO'],
            'target': [1.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            # 执行数据处理链
            result = self.router.execute_chain('data_processing', {'data_path': temp_path})
            
            # 验证结果
            self.assertIn('workflow_id', result)
            self.assertIn('status', result)
            
            # 检查工作流状态
            workflow_id = result['workflow_id']
            self.assertIn(workflow_id, self.router.active_workflows)
            
        finally:
            os.unlink(temp_path)
            
    def test_execute_parallel_tasks(self):
        """测试并行任务执行"""
        # 准备测试任务
        task_groups = [
            ['load_data'],
            ['preprocess_data', 'fuse_features']
        ]
        
        inputs = {
            'data_path': 'test.csv',
            'raw_data': {'smiles': ['CCO']},
            'processed_data': {'features': np.random.rand(1, 100)}
        }
        
        # 由于需要实际的智能体，这里只测试结构
        workflow_id = f"parallel_test_{id(self)}"
        self.router.active_workflows[workflow_id] = {
            'status': 'running',
            'start_time': datetime.now()
        }
        
        # 获取状态
        status = self.router.get_workflow_status(workflow_id)
        
        # 验证结果
        self.assertEqual(status['status'], 'running')
        self.assertIn('duration', status)
        
    def test_cancel_workflow(self):
        """测试取消工作流"""
        # 创建测试工作流
        workflow_id = 'test_cancel'
        self.router.active_workflows[workflow_id] = {
            'status': 'running',
            'start_time': datetime.now()
        }
        
        # 取消工作流
        result = self.router.cancel_workflow(workflow_id)
        
        # 验证结果
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(self.router.active_workflows[workflow_id]['status'], 'cancelled')
        
    def test_cleanup_workflows(self):
        """测试清理完成的工作流"""
        from datetime import datetime, timedelta
        
        # 创建一些测试工作流
        # 旧的已完成工作流
        self.router.active_workflows['old_completed'] = {
            'status': 'completed',
            'end_time': datetime.now() - timedelta(hours=25)
        }
        
        # 新的已完成工作流
        self.router.active_workflows['new_completed'] = {
            'status': 'completed',
            'end_time': datetime.now() - timedelta(hours=1)
        }
        
        # 运行中的工作流
        self.router.active_workflows['running'] = {
            'status': 'running',
            'start_time': datetime.now()
        }
        
        # 清理
        self.router.cleanup_completed_workflows(retention_hours=24)
        
        # 验证结果
        self.assertNotIn('old_completed', self.router.active_workflows)
        self.assertIn('new_completed', self.router.active_workflows)
        self.assertIn('running', self.router.active_workflows)

if __name__ == '__main__':
    unittest.main()