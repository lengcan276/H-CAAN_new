"""
模型解释工具函数
"""
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def calculate_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """计算特征重要性"""
    if hasattr(model, 'feature_importances_'):
        # 树模型
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 线性模型
        importances = np.abs(model.coef_)
    else:
        # 其他模型，使用置换重要性
        logger.warning("模型不支持直接获取特征重要性，使用置换方法")
        return None
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

def generate_shap_explanations(model, X: np.ndarray, 
                              feature_names: List[str]) -> Dict:
    """生成SHAP解释"""
    # 创建SHAP解释器
    if hasattr(model, 'predict'):
        explainer = shap.Explainer(model.predict, X)
    else:
        explainer = shap.Explainer(model, X)
    
    # 计算SHAP值
    shap_values = explainer(X)
    
    return {
        'shap_values': shap_values.values,
        'base_value': shap_values.base_values,
        'feature_names': feature_names
    }

def plot_feature_importance(importance_df: pd.DataFrame, top_k: int = 20):
    """绘制特征重要性图"""
    plt.figure(figsize=(10, 8))
    
    # 选择前K个特征
    top_features = importance_df.head(top_k)
    
    # 条形图
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('重要性得分')
    plt.ylabel('特征')
    plt.title(f'Top {top_k} 重要特征')
    plt.tight_layout()
    
    return plt.gcf()

def visualize_attention_weights(attention_matrix: np.ndarray,
                              labels: Optional[List[str]] = None):
    """可视化注意力权重"""
    plt.figure(figsize=(10, 8))
    
    # 热力图
    sns.heatmap(
        attention_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.title('注意力权重矩阵')
    plt.tight_layout()
    
    return plt.gcf()

def generate_counterfactual_explanations(model, X: np.ndarray,
                                       y_target: float,
                                       n_samples: int = 5) -> List[Dict]:
    """生成反事实解释"""
    counterfactuals = []
    
    # 简化实现：通过扰动特征生成反事实样本
    original_pred = model.predict(X.reshape(1, -1))[0]
    
    for _ in range(n_samples):
        # 随机扰动
        perturbation = np.random.normal(0, 0.1, X.shape)
        cf_sample = X + perturbation
        
        # 预测
        cf_pred = model.predict(cf_sample.reshape(1, -1))[0]
        
        # 计算变化
        feature_changes = cf_sample - X
        important_changes = np.argsort(np.abs(feature_changes))[-5:]
        
        counterfactuals.append({
            'original': X,
            'counterfactual': cf_sample,
            'original_pred': original_pred,
            'cf_pred': cf_pred,
            'important_changes': important_changes,
            'changes': feature_changes
        })
    
    return counterfactuals

def analyze_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray,
                            X: Optional[np.ndarray] = None) -> Dict:
    """分析预测误差"""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    analysis = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_overestimate': np.max(errors),
        'max_underestimate': np.min(errors),
        'outlier_indices': np.where(abs_errors > 2 * np.std(abs_errors))[0]
    }
    
    # 如果提供了特征，分析误差与特征的关系
    if X is not None:
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], abs_errors)[0, 1]
            correlations.append(corr)
        
        analysis['feature_error_correlation'] = correlations
    
    return analysis

def create_explanation_report(model, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]) -> Dict:
    """创建完整的解释报告"""
    report = {
        'timestamp': pd.Timestamp.now(),
        'model_type': type(model).__name__,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    
    # 特征重要性
    importance_df = calculate_feature_importance(model, feature_names)
    if importance_df is not None:
        report['feature_importance'] = importance_df.to_dict()
    
    # 预测性能
    y_pred = model.predict(X)
    from .model_utils import evaluate_model
    report['performance'] = evaluate_model(y, y_pred)
    
    # 误差分析
    report['error_analysis'] = analyze_prediction_errors(y, y_pred, X)
    
    return report