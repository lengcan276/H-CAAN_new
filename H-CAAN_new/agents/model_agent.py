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

logger = logging.getLogger(__name__)

class EnsembleModel:
    """集成模型，结合多个基模型"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'gpr': GaussianProcessRegressor(random_state=42)
        }
        self.weights = None
        self.is_trained = False
        
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
        
    def train_model(self, fused_features: np.ndarray, labels: np.ndarray, 
                   train_params: Dict) -> str:
        """
        训练集成模型
        
        Args:
            fused_features: 融合特征向量
            labels: 目标值
            train_params: 训练参数
            
        Returns:
            模型保存路径
        """
        logger.info("开始训练模型...")
        
        # 数据检查
        if len(fused_features) != len(labels):
            raise ValueError("特征和标签数量不匹配")
            
        # 训练模型
        self.model.fit(fused_features, labels)
        
        # 保存模型
        model_dir = train_params.get('model_dir', 'data/models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"ensemble_model_{train_params.get('task_name', 'default')}.pkl")
        joblib.dump(self.model, model_path)
        self.model_path = model_path
        
        # 记录训练历史
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_samples': len(labels),
            'params': train_params,
            'model_path': model_path
        })
        
        logger.info(f"模型训练完成，保存至: {model_path}")
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
        
        # 加载模型
        model = joblib.load(model_path)
        
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