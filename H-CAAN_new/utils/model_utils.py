"""
模型训练和评估工具函数
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

def train_ensemble_model(X_train: np.ndarray, y_train: np.ndarray,
                        model_types: List[str] = ['rf', 'gbm', 'nn']) -> Dict:
    """训练集成模型"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    
    models = {}
    
    if 'rf' in model_types:
        logger.info("训练随机森林...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models['rf'] = rf
    
    if 'gbm' in model_types:
        logger.info("训练梯度提升...")
        gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbm.fit(X_train, y_train)
        models['gbm'] = gbm
    
    if 'nn' in model_types:
        logger.info("训练神经网络...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        models['nn'] = nn_model
    
    return models

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """评估模型性能"""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson': np.corrcoef(y_true, y_pred)[0, 1]
    }
    
    # 计算百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics['mape'] = mape
    
    return metrics

def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                        cv_folds: int = 5) -> Dict[str, np.ndarray]:
    """交叉验证"""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 计算多个指标
    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    cv_results = {}
    
    for score_name in scoring:
        scores = cross_val_score(model, X, y, cv=kfold, scoring=score_name)
        cv_results[score_name] = scores
    
    return cv_results

def ensemble_predict(models: Dict, X: np.ndarray,
                    weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """集成预测"""
    predictions = {}
    
    # 获取各模型预测
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = pred
    
    # 计算权重
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models.keys()}
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # 加权平均
    ensemble_pred = np.zeros(len(X))
    for name, pred in predictions.items():
        ensemble_pred += weights[name] * pred
    
    return ensemble_pred

def calculate_uncertainty(models: Dict, X: np.ndarray) -> np.ndarray:
    """计算预测不确定性"""
    predictions = []
    
    for model in models.values():
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 使用标准差作为不确定性度量
    uncertainty = np.std(predictions, axis=0)
    
    return uncertainty

def save_model(model, filepath: str):
    """保存模型"""
    joblib.dump(model, filepath)
    logger.info(f"模型已保存到: {filepath}")

def load_model(filepath: str):
    """加载模型"""
    model = joblib.load(filepath)
    logger.info(f"模型已从 {filepath} 加载")
    return model

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

def plot_learning_curve(train_scores: List[float], val_scores: List[float],
                       metric_name: str = 'Loss') -> None:
    """绘制学习曲线"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-', label=f'训练{metric_name}')
    plt.plot(epochs, val_scores, 'r-', label=f'验证{metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title('学习曲线')
    plt.legend()
    plt.grid(True)
    plt.show()