"""
工具函数测试
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import (
    validate_smiles, smiles_to_mol, calculate_molecular_descriptors,
    preprocess_dataset, split_dataset, augment_smiles, normalize_features
)
from utils.fusion_utils import (
    hierarchical_fusion, attention_based_fusion, weighted_average_fusion,
    concatenate_fusion, gated_fusion, compute_fusion_metrics
)
from utils.model_utils import (
    train_ensemble_model, evaluate_model, cross_validate_model,
    ensemble_predict, calculate_uncertainty, EarlyStopping
)
from utils.explanation_utils import (
    calculate_feature_importance, plot_feature_importance,
    analyze_prediction_errors, create_explanation_report
)
from utils.paper_utils import (
    PaperFormatter, create_paper_figures, save_paper_as_docx,
    paper_content_to_markdown, generate_bibtex
)

class TestDataUtils(unittest.TestCase):
    """测试数据处理工具函数"""
    
    def test_validate_smiles(self):
        """测试SMILES验证"""
        # 有效SMILES
        self.assertTrue(validate_smiles('CCO'))
        self.assertTrue(validate_smiles('CC(C)O'))
        self.assertTrue(validate_smiles('c1ccccc1'))
        
        # 无效SMILES
        self.assertFalse(validate_smiles('invalid'))
        self.assertFalse(validate_smiles('CC('))
        self.assertFalse(validate_smiles(''))
        
    def test_smiles_to_mol(self):
        """测试SMILES转分子对象"""
        # 有效转换
        mol = smiles_to_mol('CCO')
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 3)
        
        # 无效转换
        mol = smiles_to_mol('invalid')
        self.assertIsNone(mol)
        
    def test_calculate_molecular_descriptors(self):
        """测试分子描述符计算"""
        mol = smiles_to_mol('CCO')
        descriptors = calculate_molecular_descriptors(mol)
        
        # 验证描述符
        self.assertIn('molecular_weight', descriptors)
        self.assertIn('logp', descriptors)
        self.assertIn('hbd', descriptors)
        self.assertIn('hba', descriptors)
        self.assertAlmostEqual(descriptors['molecular_weight'], 46.07, places=1)
        
    def test_preprocess_dataset(self):
        """测试数据集预处理"""
        # 创建测试数据
        df = pd.DataFrame({
            'smiles': ['CCO', 'invalid', 'CC(C)O'],
            'target': [1.0, 2.0, 3.0]
        })
        
        # 预处理
        result = preprocess_dataset(df)
        
        # 验证结果
        self.assertEqual(len(result), 2)  # invalid SMILES被过滤
        self.assertIn('molecular_weight', result.columns)
        self.assertIn('logp', result.columns)
        
    def test_split_dataset(self):
        """测试数据集划分"""
        # 创建测试数据
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        
        # 划分数据集
        train_df, test_df = split_dataset(df, test_size=0.2)
        
        # 验证结果
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        self.assertEqual(len(train_df) + len(test_df), len(df))
        
    def test_augment_smiles(self):
        """测试SMILES增强"""
        smiles = 'c1ccccc1'
        augmented = augment_smiles(smiles, num_augmented=5)
        
        # 验证结果
        self.assertLessEqual(len(augmented), 5)
        self.assertIn(smiles, augmented)
        
        # 验证所有增强的SMILES都是有效的
        for aug_smi in augmented:
            self.assertTrue(validate_smiles(aug_smi))
            
    def test_normalize_features(self):
        """测试特征归一化"""
        features = np.array([[1, 2], [3, 4], [5, 6]])
        
        # 标准化
        normalized = normalize_features(features, method='standard')
        self.assertAlmostEqual(normalized.mean(), 0, places=5)
        self.assertAlmostEqual(normalized.std(), 1, places=5)
        
        # MinMax归一化
        normalized = normalize_features(features, method='minmax')
        self.assertAlmostEqual(normalized.min(), 0)
        self.assertAlmostEqual(normalized.max(), 1)

class TestFusionUtils(unittest.TestCase):
    """测试融合工具函数"""
    
    def setUp(self):
        # 准备测试数据
        self.modal_features = {
            'smiles': np.random.rand(10, 128),
            'graph': np.random.rand(10, 128),
            'fingerprint': np.random.rand(10, 128)
        }
        
    def test_hierarchical_fusion(self):
        """测试层次化融合"""
        # 注意力融合
        fused = hierarchical_fusion(self.modal_features, fusion_method='attention')
        self.assertEqual(fused.shape, (10, 128))
        
        # 加权平均融合
        fused = hierarchical_fusion(self.modal_features, fusion_method='weighted')
        self.assertEqual(fused.shape, (10, 128))
        
        # 拼接融合
        fused = hierarchical_fusion(self.modal_features, fusion_method='concatenate')
        self.assertEqual(fused.shape, (10, 384))  # 3 * 128
        
    def test_attention_based_fusion(self):
        """测试基于注意力的融合"""
        fused = attention_based_fusion(self.modal_features)
        
        # 验证结果
        self.assertEqual(fused.shape, (10, 128))
        self.assertFalse(np.isnan(fused).any())
        
    def test_weighted_average_fusion(self):
        """测试加权平均融合"""
        # 默认权重
        fused = weighted_average_fusion(self.modal_features)
        self.assertEqual(fused.shape, (10, 128))
        
        # 自定义权重
        weights = {'smiles': 0.5, 'graph': 0.3, 'fingerprint': 0.2}
        fused = weighted_average_fusion(self.modal_features, weights)
        self.assertEqual(fused.shape, (10, 128))
        
    def test_gated_fusion(self):
        """测试门控融合"""
        fused = gated_fusion(self.modal_features)
        
        # 验证结果
        self.assertEqual(fused.shape, (10, 128))
        self.assertFalse(np.isnan(fused).any())
        
    def test_compute_fusion_metrics(self):
        """测试融合指标计算"""
        fused_features = np.random.rand(10, 128)
        metrics = compute_fusion_metrics(self.modal_features, fused_features)
        
        # 验证指标
        self.assertIn('smiles_correlation', metrics)
        self.assertIn('compression_ratio', metrics)
        self.assertIn('feature_diversity', metrics)
        self.assertLess(metrics['compression_ratio'], 1.0)

class TestModelUtils(unittest.TestCase):
    """测试模型工具函数"""
    
    def setUp(self):
        # 准备测试数据
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.rand(100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.rand(20)
        
    def test_train_ensemble_model(self):
        """测试集成模型训练"""
        models = train_ensemble_model(self.X_train, self.y_train, 
                                    model_types=['rf', 'gbm'])
        
        # 验证结果
        self.assertIn('rf', models)
        self.assertIn('gbm', models)
        self.assertIsNotNone(models['rf'])
        self.assertIsNotNone(models['gbm'])
        
    def test_evaluate_model(self):
        """测试模型评估"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = evaluate_model(y_true, y_pred)
        
        # 验证指标
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('pearson', metrics)
        self.assertIn('mape', metrics)
        self.assertGreater(metrics['r2'], 0.9)
        
    def test_cross_validate_model(self):
        """测试交叉验证"""
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10)
        cv_results = cross_validate_model(model, self.X_train, self.y_train, cv_folds=3)
        
        # 验证结果
        self.assertIn('neg_mean_squared_error', cv_results)
        self.assertIn('neg_mean_absolute_error', cv_results)
        self.assertIn('r2', cv_results)
        self.assertEqual(len(cv_results['r2']), 3)
        
    def test_ensemble_predict(self):
        """测试集成预测"""
        # 训练模型
        models = train_ensemble_model(self.X_train, self.y_train, 
                                    model_types=['rf', 'gbm'])
        
        # 预测
        predictions = ensemble_predict(models, self.X_test)
        
        # 验证结果
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertFalse(np.isnan(predictions).any())
        
    def test_calculate_uncertainty(self):
        """测试不确定性计算"""
        # 训练多个模型
        models = train_ensemble_model(self.X_train, self.y_train, 
                                    model_types=['rf', 'gbm'])
        
        # 计算不确定性
        uncertainty = calculate_uncertainty(models, self.X_test)
        
        # 验证结果
        self.assertEqual(len(uncertainty), len(self.X_test))
        self.assertTrue(np.all(uncertainty >= 0))
        
    def test_early_stopping(self):
        """测试早停机制"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # 模拟训练过程
        losses = [1.0, 0.9, 0.85, 0.84, 0.835, 0.834, 0.833]
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss)
            if i < 6:
                self.assertFalse(should_stop)
            else:
                self.assertTrue(should_stop)

class TestExplanationUtils(unittest.TestCase):
    """测试解释工具函数"""
    
    def setUp(self):
        # 创建测试模型
        from sklearn.ensemble import RandomForestRegressor
        
        self.X = np.random.rand(100, 10)
        self.y = np.random.rand(100)
        self.feature_names = [f'Feature_{i}' for i in range(10)]
        
        self.model = RandomForestRegressor(n_estimators=10)
        self.model.fit(self.X, self.y)
        
    def test_calculate_feature_importance(self):
        """测试特征重要性计算"""
        importance_df = calculate_feature_importance(self.model, self.feature_names)
        
        # 验证结果
        self.assertEqual(len(importance_df), 10)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        self.assertAlmostEqual(importance_df['importance'].sum(), 1.0, places=5)
        
    def test_analyze_prediction_errors(self):
        """测试预测误差分析"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.2, 2.1, 2.8, 4.3, 4.7])
        
        analysis = analyze_prediction_errors(y_true, y_pred, X=self.X[:5])
        
        # 验证结果
        self.assertIn('mean_error', analysis)
        self.assertIn('std_error', analysis)
        self.assertIn('max_overestimate', analysis)
        self.assertIn('max_underestimate', analysis)
        self.assertIn('outlier_indices', analysis)
        self.assertIn('feature_error_correlation', analysis)
        
    def test_create_explanation_report(self):
        """测试解释报告创建"""
        report = create_explanation_report(self.model, self.X[:20], self.y[:20], 
                                         self.feature_names)
        
        # 验证报告内容
        self.assertIn('timestamp', report)
        self.assertIn('model_type', report)
        self.assertIn('n_samples', report)
        self.assertIn('n_features', report)
        self.assertIn('feature_importance', report)
        self.assertIn('performance', report)
        self.assertIn('error_analysis', report)

class TestPaperUtils(unittest.TestCase):
    """测试论文生成工具函数"""
    
    def test_paper_formatter(self):
        """测试论文格式化器"""
        formatter = PaperFormatter(template='ieee')
        
        # 添加章节
        formatter.add_section('Introduction', 'This is the introduction.', level=1)
        formatter.add_section('Methods', 'These are the methods.', level=1)
        
        # 转换为Markdown
        md_content = formatter.to_markdown()
        self.assertIn('# Introduction', md_content)
        self.assertIn('# Methods', md_content)
        
        # 转换为LaTeX
        latex_content = formatter.to_latex()
        self.assertIn('\\section{Introduction}', latex_content)
        self.assertIn('\\section{Methods}', latex_content)
        
    def test_format_references(self):
        """测试参考文献格式化"""
        formatter = PaperFormatter(template='ieee')
        
        references = [
            {
                'authors': 'Smith J, Doe J',
                'title': 'Test Paper',
                'journal': 'Test Journal',
                'volume': '1',
                'pages': '1-10',
                'year': '2024'
            }
        ]
        
        formatted = formatter.format_references(references)
        
        # 验证IEEE格式
        self.assertIn('[1]', formatted)
        self.assertIn('Smith J, Doe J', formatted)
        self.assertIn('"Test Paper,"', formatted)
        
    def test_paper_content_to_markdown(self):
        """测试论文内容转Markdown"""
        paper_content = {
            'title': 'Test Paper',
            'authors': 'Test Author',
            'abstract': 'This is the abstract.',
            'keywords': 'test, paper',
            'sections': [
                {
                    'title': 'Introduction',
                    'content': 'This is the introduction.',
                    'level': 2
                }
            ],
            'references': ['Reference 1', 'Reference 2']
        }
        
        md_content = paper_content_to_markdown(paper_content)
        
        # 验证内容
        self.assertIn('# Test Paper', md_content)
        self.assertIn('**Test Author**', md_content)
        self.assertIn('## Introduction', md_content)
        self.assertIn('1. Reference 1', md_content)
        
    def test_generate_bibtex(self):
        """测试BibTeX生成"""
        references = [
            {
                'type': 'article',
                'key': 'smith2024',
                'author': 'Smith, J. and Doe, J.',
                'title': 'Test Paper',
                'journal': 'Test Journal',
                'year': '2024',
                'volume': '1',
                'pages': '1-10'
            }
        ]
        
        bibtex = generate_bibtex(references)
        
        # 验证BibTeX格式
        self.assertIn('@article{smith2024,', bibtex)
        self.assertIn('author = {Smith, J. and Doe, J.}', bibtex)
        self.assertIn('title = {Test Paper}', bibtex)

if __name__ == '__main__':
    unittest.main()