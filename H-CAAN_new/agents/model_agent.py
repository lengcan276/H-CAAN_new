"""
模型训练与推断智能体
负责集成模型训练、预测和不确定性估计
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
logger = logging.getLogger(__name__)

class EnsembleModel:
    """集成模型，结合多个基模型"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # 完全禁用XGBoost的GPU支持，避免版本兼容问题
        # 使用传统的集成模型
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'gpr': GaussianProcessRegressor(random_state=42)
        }
        
        self.weights = None
        self.is_trained = False
        
        logger.info("使用CPU集成模型（RF + GBM + GPR）")
        
    def fit(self, X, y):
        """训练所有基模型"""
        # 划分验证集用于确定权重
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        predictions = []
        for name, model in self.models.items():
            logger.info(f"训练 {name} 模型...")
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            predictions.append(pred)
            
        # 计算最优权重
        predictions = np.array(predictions).T
        self.weights = self._optimize_weights(predictions, y_val)
        self.is_trained = True
        
        # 在全量数据上重新训练
        for name, model in self.models.items():
            model.fit(X, y)

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        """使用提供的训练集和验证集训练模型"""
        predictions = []
        
        # 在训练集上训练每个基模型
        for name, model in self.models.items():
            logger.info(f"训练 {name} 模型...")
            model.fit(X_train, y_train)
            
            # 在验证集上评估
            pred_val = model.predict(X_val)
            predictions.append(pred_val)
            
            # 计算验证集性能
            val_mse = np.mean((pred_val - y_val)**2)
            logger.info(f"{name} 验证集MSE: {val_mse:.4f}")
        
        # 使用验证集预测结果计算最优权重
        predictions = np.array(predictions).T
        self.weights = self._optimize_weights(predictions, y_val)
        self.is_trained = True
        
        logger.info(f"模型权重: RF={self.weights[0]:.3f}, GBM={self.weights[1]:.3f}, GPR={self.weights[2]:.3f}")
            
    def predict(self, X):
        """集成预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
            
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
            
        predictions = np.array(predictions).T
        return np.dot(predictions, self.weights)
        
    def predict_with_uncertainty(self, X):
        """预测并估计不确定性"""
        predictions = []
        uncertainties = []
        
        for name, model in self.models.items():
            if name == 'rf':
                # 使用随机森林的树预测分布
                tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                pred = tree_predictions.mean(axis=0)
                uncertainty = tree_predictions.std(axis=0)
            elif name == 'gpr':
                # 高斯过程直接提供不确定性
                pred, uncertainty = model.predict(X, return_std=True)
            else:
                # 其他模型使用简单估计
                pred = model.predict(X)
                uncertainty = np.ones_like(pred) * 0.1
                
            predictions.append(pred)
            uncertainties.append(uncertainty)
            
        # 加权平均
        predictions = np.array(predictions).T
        uncertainties = np.array(uncertainties).T
        
        final_pred = np.dot(predictions, self.weights)
        final_uncertainty = np.sqrt(np.dot(uncertainties**2, self.weights**2))
        
        return final_pred, final_uncertainty
        
    def _optimize_weights(self, predictions, y_true):
        """优化集成权重"""
        from scipy.optimize import minimize
        
        def loss(weights):
            weights = weights / weights.sum()
            pred = np.dot(predictions, weights)
            return np.mean((pred - y_true)**2)
            
        n_models = predictions.shape[1]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n_models
        
        result = minimize(loss, np.ones(n_models)/n_models, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x

class ModelAgent:
    """模型管理智能体"""
    
    def __init__(self):
        self.model = EnsembleModel()
        self.model_path = None
        self.training_history = []
        
    def train_model(self, split_data: Dict, train_params: Dict) -> str:
        """
        训练集成模型
        
        Args:
            split_data: 包含train/val/test的划分数据
            train_params: 训练参数
            
        Returns:
            模型保存路径
        """
        logger.info("开始训练模型...")
        
        # 提取目标属性名称
        target_property = train_params.get('target_property', 'target')
        
        # 提取训练集和验证集数据
        # 使用融合特征或指纹特征
        feature_type = train_params.get('feature_type', 'fingerprints')
        
        X_train = np.array(split_data['train'][feature_type])
        y_train = np.array(split_data['train']['labels'][target_property])
        
        X_val = np.array(split_data['val'][feature_type])
        y_val = np.array(split_data['val']['labels'][target_property])
        
        logger.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 使用已经划分好的数据训练
        self.model.fit_with_validation(X_train, y_train, X_val, y_val)
        
        # 在测试集上评估
        X_test = np.array(split_data['test'][feature_type])
        y_test = np.array(split_data['test']['labels'][target_property])
        
        predictions_test, _ = self.model.predict_with_uncertainty(X_test)
        test_metrics = self.evaluate_model(predictions_test, y_test)
        
        logger.info(f"测试集性能: {test_metrics}")
        
        # 保存模型
        model_dir = train_params.get('model_dir', 'data/models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"ensemble_model_{train_params.get('task_name', 'default')}.pkl")
        
        # 保存模型和相关信息
        model_info = {
            'model': self.model,
            'test_metrics': test_metrics,
            'train_params': train_params,
            'data_split': split_data.get('ratios', {}),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
        
        joblib.dump(model_info, model_path)
        self.model_path = model_path
        
        # 添加验证 - 确保文件确实被保存了
        if not os.path.exists(model_path):
            logger.error(f"模型保存失败: {model_path}")
            raise Exception(f"模型文件未成功保存到: {model_path}")
        else:
            # 验证文件大小
            file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
            logger.info(f"模型文件大小: {file_size:.2f} MB")
        
        # 记录训练历史
        import pandas as pd
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
            'test_metrics': test_metrics,
            'params': train_params,
            'model_path': model_path
        })
        
        logger.info(f"模型训练完成，保存至: {model_path}")
        logger.info(f"测试集性能指标: {test_metrics}")
        
        from utils.model_manager import ModelManager
        model_manager = ModelManager()
        model_manager.register_model(
            model_path=model_path,
            task_name=train_params.get('task_name', 'default'),
            metrics=test_metrics,
            metadata={
                'train_params': train_params,
                'data_split': split_data.get('ratios', {}),
                'n_samples': {
                    'train': len(X_train),
                    'val': len(X_val),
                    'test': len(X_test)
                }
            }
        )
        return model_path
        
    def predict(self, model_path: str, fused_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练好的模型进行预测
        
        Args:
            model_path: 模型路径
            fused_features: 待预测特征
            
        Returns:
            (预测结果, 不确定性估计)
        """
        logger.info(f"加载模型: {model_path}")
        
        # 加载模型信息
        model_info = joblib.load(model_path)
        
        # 兼容旧版本和新版本
        if isinstance(model_info, dict):
            model = model_info['model']
            logger.info(f"加载的模型测试集性能: {model_info.get('test_metrics', {})}")
        else:
            model = model_info
        
        # 预测
        predictions, uncertainties = model.predict_with_uncertainty(fused_features)
        
        logger.info(f"预测完成，样本数: {len(predictions)}")
        return predictions, uncertainties
        
    def evaluate_model(self, predictions: np.ndarray, true_values: np.ndarray) -> Dict:
        """评估模型性能"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'correlation': np.corrcoef(true_values, predictions)[0, 1]
        }
        
        return metrics
        
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if not self.model.is_trained:
            raise ValueError("模型尚未训练")
            
        # 从随机森林模型获取特征重要性
        rf_model = self.model.models['rf']
        return rf_model.feature_importances_
    

    
    def evaluate_modal_combination(self, modal_features: List[np.ndarray], 
                                labels: np.ndarray,
                                modal_indices: List[int],
                                train_idx: np.ndarray,
                                val_idx: np.ndarray,
                                weights: np.ndarray = None) -> Dict:
        """
        评估特定模态组合的实际性能
        
        Args:
            modal_features: 各模态特征列表
            labels: 目标标签
            modal_indices: 要使用的模态索引
            train_idx: 训练集索引
            val_idx: 验证集索引
            weights: 模态权重
            
        Returns:
            性能指标字典
        """
        # 选择指定模态的特征
        selected_features = [modal_features[i] for i in modal_indices]
        
        # 如果没有提供权重，使用均匀权重
        if weights is None:
            weights = np.ones(len(modal_indices)) / len(modal_indices)
        else:
            # 确保权重归一化
            weights = weights / weights.sum()
        
        # 融合特征
        fused_train = self._weighted_fusion(
            [feat[train_idx] for feat in selected_features], weights
        )
        fused_val = self._weighted_fusion(
            [feat[val_idx] for feat in selected_features], weights
        )
        
        # 使用当前的集成模型配置训练
        temp_model = EnsembleModel()
        temp_model.fit(fused_train, labels[train_idx])
        
        # 在验证集上评估
        predictions = temp_model.predict(fused_val)
        
        # 计算各种指标
        return {
            'rmse': np.sqrt(mean_squared_error(labels[val_idx], predictions)),
            'mae': mean_absolute_error(labels[val_idx], predictions),
            'r2': r2_score(labels[val_idx], predictions),
            'correlation': np.corrcoef(labels[val_idx], predictions)[0, 1]
        }

    def _weighted_fusion(self, features: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """加权融合特征"""
        # 确保所有特征具有相同的样本数
        n_samples = features[0].shape[0]
        
        # 获取最大特征维度
        max_dim = max(f.shape[1] if len(f.shape) > 1 else 1 for f in features)
        
        # 初始化融合特征
        fused = np.zeros((n_samples, max_dim))
        
        # 加权融合
        for i, (feat, weight) in enumerate(zip(features, weights)):
            # 处理维度不匹配
            if len(feat.shape) == 1:
                feat = feat.reshape(-1, 1)
            
            if feat.shape[1] < max_dim:
                # 填充
                feat = np.pad(feat, ((0, 0), (0, max_dim - feat.shape[1])), 'constant')
            elif feat.shape[1] > max_dim:
                # 截断
                feat = feat[:, :max_dim]
            
            fused += weight * feat
        
        return fused