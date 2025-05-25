"""
智能体单元测试
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataAgent
from agents.fusion_agent import FusionAgent
from agents.model_agent import ModelAgent
from agents.explain_agent import ExplainAgent
from agents.paper_agent import PaperAgent
from agents.multi_agent_manager import MultiAgentManager

class TestDataAgent(unittest.TestCase):
    """测试数据处理智能体"""
    
    def setUp(self):
        self.agent = DataAgent()
        self.test_data = pd.DataFrame({
            'smiles': ['CCO', 'CC(C)O', 'c1ccccc1'],
            'target': [1.0, 2.0, 3.0]
        })
        
    def test_load_raw_data(self):
        """测试数据加载"""
        # 创建临时CSV文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            # 加载数据
            result = self.agent.load_raw_data(temp_path)
            
            # 验证结果
            self.assertIn('smiles', result)
            self.assertIn('molecules', result)
            self.assertEqual(len(result['smiles']), 3)
            self.assertEqual(result['smiles'][0], 'CCO')
            
        finally:
            os.unlink(temp_path)
            
    def test_preprocess_data(self):
        """测试数据预处理"""
        # 准备测试数据
        raw_data = {
            'smiles': ['CCO', 'CC(C)O'],
            'molecules': [],
            'properties': {'target': [1.0, 2.0]}
        }
        
        # 添加分子对象
        from rdkit import Chem
        for smi in raw_data['smiles']:
            mol = Chem.MolFromSmiles(smi)
            raw_data['molecules'].append(mol)
            
        # 预处理
        result = self.agent.preprocess_data(raw_data)
        
        # 验证结果
        self.assertIn('smiles_features', result)
        self.assertIn('graph_features', result)
        self.assertIn('fingerprints', result)
        self.assertIn('molecular_descriptors', result)
        self.assertEqual(len(result['smiles_features']), 2)
        
    def test_validate_smiles(self):
        """测试SMILES验证"""
        from utils.data_utils import validate_smiles
        
        # 有效SMILES
        self.assertTrue(validate_smiles('CCO'))
        self.assertTrue(validate_smiles('c1ccccc1'))
        
        # 无效SMILES
        self.assertFalse(validate_smiles('invalid'))
        self.assertFalse(validate_smiles('CC('))

class TestFusionAgent(unittest.TestCase):
    """测试特征融合智能体"""
    
    def setUp(self):
        self.agent = FusionAgent()
        
    def test_fuse_features(self):
        """测试特征融合"""
        # 准备测试数据
        n_samples = 10
        processed_data = {
            'smiles_features': np.random.rand(n_samples, 100).tolist(),
            'fingerprints': np.random.randint(0, 2, (n_samples, 2048)).tolist(),
            'graph_features': []
        }
        
        # 创建模拟的图数据
        import torch
        from torch_geometric.data import Data
        for i in range(n_samples):
            x = torch.randn(5, 5)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index)
            processed_data['graph_features'].append(data)
            
        # 执行融合
        result = self.agent.fuse_features(processed_data)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], n_samples)
        self.assertEqual(result.shape[1], 256)  # 预期的特征维度
        
    def test_attention_weights(self):
        """测试注意力权重获取"""
        weights = self.agent.get_attention_weights()
        
        # 验证结果
        self.assertIn('smiles_attention', weights)
        self.assertIn('graph_attention', weights)
        self.assertIn('cross_modal_attention', weights)
        self.assertEqual(weights['cross_modal_attention'].shape, (3, 3))

class TestModelAgent(unittest.TestCase):
    """测试模型训练智能体"""
    
    def setUp(self):
        self.agent = ModelAgent()
        
    def test_train_model(self):
        """测试模型训练"""
        # 准备测试数据
        n_samples = 100
        n_features = 256
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        
        train_params = {
            'task_name': 'test',
            'model_dir': tempfile.mkdtemp()
        }
        
        # 训练模型
        model_path = self.agent.train_model(X, y, train_params)
        
        # 验证结果
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(model_path.endswith('.pkl'))
        
        # 清理
        os.remove(model_path)
        os.rmdir(train_params['model_dir'])
        
    def test_predict(self):
        """测试模型预测"""
        # 先训练模型
        n_samples = 100
        n_features = 256
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.rand(n_samples)
        
        train_params = {
            'task_name': 'test',
            'model_dir': tempfile.mkdtemp()
        }
        
        model_path = self.agent.train_model(X_train, y_train, train_params)
        
        # 准备测试数据
        X_test = np.random.rand(10, n_features)
        
        # 预测
        predictions, uncertainties = self.agent.predict(model_path, X_test)
        
        # 验证结果
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(uncertainties), 10)
        self.assertTrue(np.all(uncertainties >= 0))
        
        # 清理
        os.remove(model_path)
        os.rmdir(train_params['model_dir'])
        
    def test_evaluate_model(self):
        """测试模型评估"""
        # 准备测试数据
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        # 评估
        metrics = self.agent.evaluate_model(y_pred, y_true)
        
        # 验证结果
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('correlation', metrics)
        self.assertGreater(metrics['r2'], 0.9)

class TestExplainAgent(unittest.TestCase):
    """测试模型解释智能体"""
    
    def setUp(self):
        self.agent = ExplainAgent()
        
    def test_generate_explanations(self):
        """测试解释生成"""
        # 创建临时模型文件
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10)
        X = np.random.rand(50, 10)
        y = np.random.rand(50)
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            joblib.dump(model, f.name)
            model_path = f.name
            
        try:
            # 生成解释
            fused_features = np.random.rand(10, 10)
            result = self.agent.generate_explanations(model_path, fused_features)
            
            # 验证结果
            self.assertIn('feature_importance', result)
            self.assertIn('attention_weights', result)
            self.assertIn('case_studies', result)
            self.assertIn('visualizations', result)
            self.assertIn('text_report', result)
            
        finally:
            os.unlink(model_path)

class TestPaperAgent(unittest.TestCase):
    """测试论文生成智能体"""
    
    def setUp(self):
        self.agent = PaperAgent()
        
    def test_generate_paper(self):
        """测试论文生成"""
        # 准备测试数据
        results = {
            'metrics': {
                'r2': 0.89,
                'rmse': 0.35,
                'mae': 0.28
            }
        }
        
        explanations = {
            'feature_importance': {
                'Feature_1': 0.25,
                'Feature_2': 0.20
            }
        }
        
        metadata = {
            'title': 'Test Paper',
            'authors': 'Test Author',
            'keywords': 'test, paper',
            'sections': ['abstract', 'introduction', 'conclusion']
        }
        
        # 生成论文
        paper_path = self.agent.generate_paper(results, explanations, metadata)
        
        # 验证结果
        self.assertTrue(os.path.exists(paper_path))
        self.assertTrue(paper_path.endswith('.md'))
        
        # 检查内容
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Test Paper', content)
            self.assertIn('Test Author', content)
            
        # 清理
        os.remove(paper_path)

class TestMultiAgentManager(unittest.TestCase):
    """测试多智能体管理器"""
    
    def setUp(self):
        self.manager = MultiAgentManager()
        
    def test_dispatch_task(self):
        """测试任务分发"""
        # 创建测试数据文件
        test_data = pd.DataFrame({
            'smiles': ['CCO'],
            'target': [1.0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            # 分发数据加载任务
            result = self.manager.dispatch_task('load_data', data_path=temp_path)
            
            # 验证结果
            self.assertIn('smiles', result)
            self.assertEqual(len(result['smiles']), 1)
            
        finally:
            os.unlink(temp_path)
            
    def test_manage_workflow(self):
        """测试工作流管理"""
        # 准备输入数据
        input_data = {
            'data_path': 'test.csv',
            'labels': np.array([1.0, 2.0]),
            'train_params': {'task_name': 'test'}
        }
        
        # 测试工作流状态管理
        workflow_id = f"test_workflow_{id(self)}"
        self.manager.task_status[workflow_id] = {
            'status': 'running',
            'current_task': 'load_data',
            'completed_tasks': [],
            'results': {}
        }
        
        # 获取状态
        status = self.manager.get_task_status(workflow_id)
        
        # 验证结果
        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['current_task'], 'load_data')
        self.assertEqual(len(status['completed_tasks']), 0)

if __name__ == '__main__':
    unittest.main()