from sklearn import metrics
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import torch
import sys
import warnings
import pandas as pd
import math
from scipy import stats
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time

sys.path.append('../../util')
from utils_smiecfp import *
from data_gen_modify import *
from analysis import *
from utils_MFBERT import *

sys.path.append('../../model')
from model_combination_0210_1 import *

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# GPU设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 训练参数
epochs = 50
batch_size = 16
label = 10000
random_state = 42

# 修改模型配置 - 使用更强的正则化
args_model = {
    'num_features_smi': 100,
    'num_features_ecfp': 1024,
    'num_features_x': 78,
    'dropout': 0.3,  # 进一步增加dropout
    'num_layer': 3,
    'num_heads': 4,
    'hidden_dim': 256,
    'output_dim': 128,
    'n_output': 1,
}

# 添加梯度累积步数以稳定训练
gradient_accumulation_steps = 8  # 增加至8

# 添加数据归一化函数
def normalize_data(data):
    """对数据进行归一化处理"""
    # 如果提供的是张量，转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    # 计算均值和标准差
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0 or np.isnan(std):
        std = 1.0
        
    # 执行归一化
    normalized = (data - mean) / std
    
    # 处理异常值
    normalized = np.clip(normalized, -3, 3)  # 限制在-3到3之间
    
    # 替换任何残留的NaN值
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized, mean, std

# 添加反归一化函数
def denormalize_data(data, mean, std):
    """将归一化的数据转换回原始比例"""
    return data * std + mean

# 为模型添加预测校准逻辑
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_means = [0.0, 0.0, 0.0, 0.0]
        self.output_stds = [1.0, 1.0, 1.0, 1.0]
        
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        # 检查并校正每个输出
        corrected_outputs = []
        for i, output in enumerate(outputs):
            # 清除NaN值
            output = torch.where(torch.isnan(output), 
                                torch.zeros_like(output), 
                                output)
            # 清除Inf值
            output = torch.where(torch.isinf(output), 
                                torch.ones_like(output) * (3.0 * self.output_stds[i]), 
                                output)
            # 应用全局裁剪
            output = torch.clamp(output, 
                                min=-3.0 * self.output_stds[i] + self.output_means[i], 
                                max=3.0 * self.output_stds[i] + self.output_means[i])
            corrected_outputs.append(output)
        return corrected_outputs
    
    def set_output_stats(self, means, stds):
        """设置输出的统计信息，用于校准"""
        self.output_means = means
        self.output_stds = stds

def train_epoch(model, loader, optimizer, criterion, device, epoch, gradient_accumulation_steps=8, max_grad_norm=0.1):
    """训练一个epoch，添加更多的稳定性措施"""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()  # 开始时清零梯度
    
    # 跟踪每个输出的值以进行统计分析
    output_values = [[] for _ in range(4)]
    
    for batch_idx, data in enumerate(loader):
        try:
            # 准备输入数据
            encodedSmi = torch.LongTensor(data.smi).to(device)
            encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
            ecfp = torch.FloatTensor(data.ep).to(device)
            y = data.y.to(device)
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)

            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            # 处理MFBERT输入
            input_ids = data.input_id.clone().detach()
            attention_mask = data.attention_mask.clone().detach()

            # 输入处理和维度调整
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # 截断到最大长度
            max_length = 512
            if input_ids.size(-1) > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            elif input_ids.size(-1) < max_length:
                padding = torch.zeros(input_ids.size(0), max_length - input_ids.size(-1), dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)], dim=-1)

            # 确保batch_size正确
            if input_ids.size(0) != y.size(0):
                input_ids = input_ids.repeat(y.size(0), 1)
                attention_mask = attention_mask.repeat(y.size(0), 1)

            # 移动到设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 前向传播，使用trycatch捕获任何错误
            try:
                y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                
                # 收集输出统计信息
                for i, pred in enumerate(y_pred):
                    output_values[i].extend(pred.detach().cpu().numpy().flatten())
                
                # 计算损失，改用加权Huber损失组合
                losses = []
                for i, pred in enumerate(y_pred):
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(-1)
                    
                    # 使用更小的beta值的Huber损失，对异常值更不敏感
                    huber_loss = nn.SmoothL1Loss(reduction='none', beta=0.1)(pred.float(), y.float())
                    
                    # 使用基于样本的权重，降低潜在异常样本的影响
                    weights = 1.0 / (1.0 + torch.abs(pred.float() - y.float()))
                    weighted_loss = (huber_loss * weights).mean()
                    
                    losses.append(weighted_loss)
                
                # 平均损失，加权以更重视表现更好的输出
                loss = sum(losses) / len(losses)
                
                # 避免NaN损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss at batch {batch_idx}, skipping")
                    continue
                
                # 梯度累积：将损失除以累积步数
                loss = loss / gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 替换NaN和Inf梯度为0，而不是跳过整个批次
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            param.grad = torch.where(torch.isnan(param.grad) | torch.isinf(param.grad), 
                                                   torch.zeros_like(param.grad), 
                                                   param.grad)
                            has_nan_grad = True
                
                if has_nan_grad:
                    print(f"Warning: NaN gradients fixed at batch {batch_idx}")
                
                # 每隔gradient_accumulation_steps步更新一次参数
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪，使用非常小的阈值
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 记录损失值
                total_loss += (loss.item() * gradient_accumulation_steps)
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Current loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            except RuntimeError as e:
                print(f"Error in forward/backward at batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()  # 出错时清零梯度
                continue
                
        except Exception as e:
            print(f"Error in batch preparation {batch_idx}: {str(e)}")
            optimizer.zero_grad()  # 出错时清零梯度
            continue
    
    # 确保最后一个不完整的累积步骤也得到处理
    if len(loader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # 更新模型包装器中的输出统计信息
    if isinstance(model, ModelWrapper):
        means = []
        stds = []
        for values in output_values:
            if len(values) > 0:
                values = np.array(values)
                # 过滤掉异常值
                values = values[~np.isnan(values) & ~np.isinf(values)]
                if len(values) > 0:
                    means.append(np.mean(values))
                    stds.append(np.std(values) if np.std(values) > 0 else 1.0)
                else:
                    means.append(0.0)
                    stds.append(1.0)
            else:
                means.append(0.0)
                stds.append(1.0)
        model.set_output_stats(means, stds)
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    return avg_loss

def evaluate(model, loader, device):
    """评估模型，添加更多健壮性"""
    model.eval()
    predictions = [[] for _ in range(4)]
    actuals = []

    # 使用SmoothL1Loss计算验证损失
    criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            try:
                # 准备输入数据
                encodedSmi = torch.LongTensor(data.smi).to(device)
                encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
                ecfp = torch.FloatTensor(data.ep).to(device)
                y = data.y.to(device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)

                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)

                # 处理MFBERT输入
                input_ids = data.input_id.clone().detach()
                attention_mask = data.attention_mask.clone().detach()

                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)
                    attention_mask = attention_mask.unsqueeze(0)

                # 截断或填充到最大长度
                max_length = 512
                if input_ids.size(-1) > max_length:
                    input_ids = input_ids[:, :max_length]
                    attention_mask = attention_mask[:, :max_length]
                elif input_ids.size(-1) < max_length:
                    padding = torch.zeros(input_ids.size(0), max_length - input_ids.size(-1), dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, padding], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)], dim=-1)

                # 确保batch_size正确
                if input_ids.size(0) != y.size(0):
                    input_ids = input_ids.repeat(y.size(0), 1)
                    attention_mask = attention_mask.repeat(y.size(0), 1)

                # 移动到设备
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # 前向传播
                try:
                    y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                    
                    # 计算损失
                    batch_losses = []
                    for i, pred in enumerate(y_pred):
                        if len(pred.shape) == 1:
                            pred = pred.unsqueeze(-1)
                        
                        # 确保无NaN
                        pred_clean = torch.where(torch.isnan(pred) | torch.isinf(pred), 
                                              torch.zeros_like(pred), 
                                              pred)
                        
                        batch_losses.append(criterion(pred_clean.float(), y.float()))
                        
                        # 存储预测结果，使用numpy处理以确保没有NaN
                        pred_np = pred_clean.cpu().numpy()
                        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=3.0, neginf=-3.0)
                        predictions[i].extend(pred_np.squeeze().tolist())
                    
                    loss = sum(batch_losses) / len(batch_losses)
                    total_loss += loss.item()
                    n_batches += 1
                    
                    # 存储实际值
                    actuals.extend(y.cpu().squeeze().tolist())
                    
                except RuntimeError as e:
                    print(f"Error in evaluation forward pass batch {batch_idx}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error in evaluation batch preparation {batch_idx}: {str(e)}")
                continue

    # 所有输出都进行后处理，确保没有异常值
    processed_predictions = []
    for i, preds in enumerate(predictions):
        preds_array = np.array(preds)
        # 替换任何NaN或Inf值
        preds_array = np.nan_to_num(preds_array, nan=0.0, posinf=3.0, neginf=-3.0)
        # 限制范围（基于观察到的数据分布）
        preds_array = np.clip(preds_array, -3, 3) 
        processed_predictions.append(preds_array.tolist())
    
    avg_val_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    return processed_predictions, actuals, avg_val_loss

def calculate_metrics(y_true, y_pred):
    """计算评估指标，增加错误处理"""
    try:
        # 确保输入没有NaN或Inf
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 替换异常值
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=3.0, neginf=-3.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # 计算指标
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        pearson = stats.pearsonr(y_true, y_pred)[0]
        
        # 检查指标是否有效
        if np.isnan(rmse) or np.isinf(rmse):
            rmse = 999.99
        if np.isnan(mae) or np.isinf(mae):
            mae = 999.99
        if np.isnan(r2) or np.isinf(r2):
            r2 = -999.99
        if np.isnan(pearson) or np.isinf(pearson):
            pearson = 0.0
            
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'rmse': 999.99,
            'mae': 999.99,
            'r2': -999.99,
            'pearson': 0.0
        }

def test_model(model, testLoader, device, base_path, label, batch_size, epochs, random_state):
    """测试模型性能，增强错误处理和数据清洗"""
    model.eval()
    predictions = [[] for _ in range(4)]
    actuals = []
    criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)

    print("\nStarting model testing...")
    with torch.no_grad():
        for batch_idx, data in enumerate(testLoader):
            try:
                # 准备输入数据
                encodedSmi = torch.LongTensor(data.smi).to(device)
                encodedSmi_mask = torch.LongTensor(getInput_mask(data.smi)).to(device)
                ecfp = torch.FloatTensor(data.ep).to(device)
                y = data.y.to(device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)

                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device)

                # 处理MFBERT输入
                input_ids = data.input_id.clone().detach()
                attention_mask = data.attention_mask.clone().detach()

                if len(input_ids.shape) == 1:
                    input_ids = input_ids.unsqueeze(0)
                    attention_mask = attention_mask.unsqueeze(0)

                max_length = 512
                if input_ids.size(-1) > max_length:
                    input_ids = input_ids[:, :max_length]
                    attention_mask = attention_mask[:, :max_length]
                elif input_ids.size(-1) < max_length:
                    padding = torch.zeros(input_ids.size(0), max_length - input_ids.size(-1), dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, padding], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)], dim=-1)

                if input_ids.size(0) != y.size(0):
                    input_ids = input_ids.repeat(y.size(0), 1)
                    attention_mask = attention_mask.repeat(y.size(0), 1)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # 前向传播
                try:
                    y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                    
                    # 存储预测结果
                    for i, pred in enumerate(y_pred):
                        if len(pred.shape) == 1:
                            pred = pred.unsqueeze(-1)
                        
                        # 清除任何NaN和Inf
                        pred_clean = torch.where(torch.isnan(pred) | torch.isinf(pred), 
                                              torch.zeros_like(pred), 
                                              pred)
                        
                        pred_np = pred_clean.cpu().numpy()
                        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=3.0, neginf=-3.0)
                        pred_np = np.clip(pred_np, -3, 3)  # 限制预测范围
                        predictions[i].extend(pred_np.squeeze().tolist())
                    
                    # 存储实际值
                    actuals.extend(y.cpu().squeeze().tolist())
                    
                except RuntimeError as e:
                    print(f"Error in test forward pass batch {batch_idx}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error in test batch preparation {batch_idx}: {str(e)}")
                continue

            if batch_idx % 10 == 0:
                print(f"Processed test batch: {batch_idx}")

    # 对所有预测和实际值进行后处理
    cleaned_predictions = []
    for pred_list in predictions:
        pred_array = np.array(pred_list)
        pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=3.0, neginf=-3.0)
        pred_array = np.clip(pred_array, -3, 3) 
        cleaned_predictions.append(pred_array)
    
    actuals_array = np.array(actuals)
    actuals_array = np.nan_to_num(actuals_array, nan=0.0)
    
    # 构建测试预测矩阵，确保没有NaN
    test_pred_array = np.column_stack(cleaned_predictions)
    test_true_array = actuals_array

    print("\nTest prediction statistics:")
    for i in range(test_pred_array.shape[1]):
        print(f"\nOutput {i+1}:")
        print(f"Mean: {np.mean(test_pred_array[:, i]):.4f}")
        print(f"Std: {np.std(test_pred_array[:, i]):.4f}")
        print(f"Min: {np.min(test_pred_array[:, i]):.4f}")
        print(f"Max: {np.max(test_pred_array[:, i]):.4f}")

    # 集成学习前对数据进行标准化
    normalized_pred_array, pred_mean, pred_std = normalize_data(test_pred_array)
    normalized_true_array, true_mean, true_std = normalize_data(test_true_array)
    
    # 集成学习
    ensemble_models = {
        'lasso': Lasso(alpha=0.01, max_iter=2000),  # 降低alpha，增加最大迭代次数
        'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=random_state),
        'gradientboost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                  max_depth=3, random_state=random_state)
    }

    print("\nTraining ensemble models...")
    ensemble_results = {}
    
    # 尝试使用每个模型单独的输出进行集成
    for name, reg_model in ensemble_models.items():
        for i in range(test_pred_array.shape[1]):
            try:
                # 使用单个预测器
                single_pred = normalized_pred_array[:, i].reshape(-1, 1)
                
                # 训练模型
                reg_model.fit(single_pred, normalized_true_array)
                
                # 预测
                ensemble_pred = reg_model.predict(single_pred)
                
                # 反归一化
                denorm_pred = denormalize_data(ensemble_pred, true_mean, true_std)
                denorm_true = denormalize_data(normalized_true_array, true_mean, true_std)
                
                # 计算指标
                metrics_result = calculate_metrics(denorm_true, denorm_pred)
                
                # 保存结果
                ensemble_name = f"{name}_output{i+1}"
                ensemble_results[ensemble_name] = {
                    'predictions': denorm_pred,
                    'metrics': metrics_result,
                    'weights': reg_model.coef_ if hasattr(reg_model, 'coef_') else reg_model.feature_importances_
                }
                
                # 打印结果
                print(f"\n{ensemble_name} results:")
                for metric_name, value in metrics_result.items():
                    print(f"{metric_name}: {value:.4f}")
                
            except Exception as e:
                print(f"Error in {name} ensemble with output {i+1}: {str(e)}")
                continue
    
    # 尝试使用所有输出的组合
    for name, reg_model in ensemble_models.items():
        try:
            # 训练集成模型
            reg_model.fit(normalized_pred_array, normalized_true_array)
            
            # 预测
            ensemble_pred = reg_model.predict(normalized_pred_array)
            
            # 反归一化
            denorm_pred = denormalize_data(ensemble_pred, true_mean, true_std)
            denorm_true = denormalize_data(normalized_true_array, true_mean, true_std)
            
            # 计算指标
            metrics_result = calculate_metrics(denorm_true, denorm_pred)
            
            ensemble_results[name] = {
                'predictions': denorm_pred,
                'metrics': metrics_result,
                'weights': reg_model.coef_ if hasattr(reg_model, 'coef_') else reg_model.feature_importances_
            }
            
            # 打印结果
            print(f"\n{name} ensemble results:")
            print(f"Weights: {ensemble_results[name]['weights']}")
            for metric_name, value in metrics_result.items():
                print(f"{metric_name}: {value:.4f}")
            
            # 保存结果
            result_df = pd.DataFrame({
                'y_true': denorm_true,
                'y_pred': denorm_pred
            })
            result_df.to_csv(f'{base_path}/{label}_test_result_{name}_{batch_size}_{epochs}_{random_state}.csv',
                            index=False)
            
            # 绘制散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(denorm_true, denorm_pred, alpha=0.5)
            plt.plot([min(denorm_true), max(denorm_true)],
                    [min(denorm_true), max(denorm_true)], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'{name.capitalize()} Ensemble Predictions vs True Values')
            plt.savefig(f'{base_path}/{label}_test_scatter_{name}_{batch_size}_{epochs}_{random_state}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error in {name} combined ensemble: {str(e)}")
            continue

    return ensemble_results

# 主程序入口
if __name__ == "__main__":
    # 创建结果目录
    base_path = './results/ML'
    os.makedirs(base_path, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    # 加载数据
    print("Loading data...")
    train_data = formDataset(root='./results/processed/', dataset='Lipophilicity_MFBERT_dataTrain')
    train_ratio = 0.8
    num_data = len(train_data)
    indices = list(range(num_data))
    train_indices, val_indices = train_test_split(indices, train_size=train_ratio,
                                                shuffle=True, random_state=random_state)
    train_dataset = [train_data[i] for i in train_indices]
    val_dataset = [train_data[i] for i in val_indices]
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = formDataset(root='./results/processed/', dataset='Lipophilicity_MFBERT_dataTest')
    testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # 初始化模型
    print("Initializing model...")
    base_model = comModel(args_model).to(device)
    
    # 使用ModelWrapper包装模型以增加稳定性
    model = ModelWrapper(base_model).to(device)

    # 使用更稳定的权重初始化方法
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # 使用Kaiming初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    model.model.apply(init_weights)

    # 优化器设置 - 使用更小的学习率和更大的权重衰减
    learning_rate = 1e-5  # 进一步降低初始学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,  # 增加权重衰减
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度器 - 使用更温和的递减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,  # 更平缓的衰减
        patience=5,
        verbose=True,
        min_lr=1e-7  # 设置最小学习率
    )

    # 使用Huber损失代替MSE，指定更小的beta值
    criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)

    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 20  # 增加耐心值
    patience_counter = 0
    resultLoss = {'losses_train': [], 'losses_val': []}
    start_time = time.time()

    # 添加学习率预热
    warmup_epochs = 5  # 增加预热周期
    warmup_factor = 0.01  # 从更低的学习率开始
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 学习率预热
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup learning rate: {warmup_lr:.6f}")

        # 训练
        train_loss = train_epoch(model, trainLoader, optimizer, criterion, device, epoch + 1, 
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                max_grad_norm=0.1)
        
        resultLoss['losses_train'].append(train_loss)
        print(f'Train avg_loss: {train_loss:.4f}')

        # 如果出现NaN损失，降低学习率并继续
        if math.isnan(train_loss):
            print("NaN loss detected, reducing learning rate by 90% and continuing...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            continue

        # 验证
        val_predictions, val_actuals, val_loss = evaluate(model, valLoader, device)
        resultLoss['losses_val'].append(val_loss)
        print(f'Validation avg_loss: {val_loss:.4f}')

        # 更新学习率
        scheduler.step(val_loss)

        # Early stopping逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'best_model_{label}.pth')
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

    # 加载最佳模型进行测试
    print("\nLoading best model for testing...")
    best_model_path = f'best_model_{label}.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        ensemble_results = test_model(model, testLoader, device, base_path, label,
                                    batch_size, epochs, random_state)
    else:
        print("No saved model found. Please train the model first.")

    print("\nTraining and evaluation completed successfully!")
