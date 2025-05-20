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

sys.path.append('../util')
from utils_smiecfp import *
from data_gen_modify import *
from analysis import *
from utils_MFBERT import *

sys.path.append('../model')
from model_combination_0210_1 import *

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# GPU设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 训练参数
epochs = 100  # 增加最大epoch数
batch_size = 16
label = 10000
random_state = 42

# 模型配置 - 更平衡的设置
args_model = {
    'num_features_smi': 100,
    'num_features_ecfp': 1024,
    'num_features_x': 78,
    'dropout': 0.2,  # 适中的dropout
    'num_layer': 3,
    'num_heads': 4,
    'hidden_dim': 256,
    'output_dim': 128,
    'n_output': 1,
}

# 梯度累积步数
gradient_accumulation_steps = 4

# 添加EMA模型更新机制
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 数据归一化函数
def normalize_data(data):
    """对数据进行归一化处理"""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0 or np.isnan(std):
        std = 1.0
        
    normalized = (data - mean) / std
    normalized = np.clip(normalized, -5, 5)  # 允许更宽的范围
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized, mean, std

# 反归一化函数
def denormalize_data(data, mean, std):
    """将归一化的数据转换回原始比例"""
    return data * std + mean

# 模型包装器
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_means = [0.0, 0.0, 0.0, 0.0]
        self.output_stds = [1.0, 1.0, 1.0, 1.0]
        
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        corrected_outputs = []
        
        for i, output in enumerate(outputs):
            # 清除NaN和Inf
            output = torch.where(torch.isnan(output) | torch.isinf(output), 
                                torch.zeros_like(output), 
                                output)
            # 动态裁剪，基于历史统计数据
            output = torch.clamp(output, 
                                min=-5.0 * self.output_stds[i] + self.output_means[i], 
                                max=5.0 * self.output_stds[i] + self.output_means[i])
            corrected_outputs.append(output)
            
        return corrected_outputs
    
    def set_output_stats(self, means, stds):
        """设置输出的统计信息，用于校准"""
        self.output_means = means
        self.output_stds = stds

# 自定义损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MultiTaskLoss, self).__init__()
        self.reduction = reduction
        self.huber = nn.SmoothL1Loss(reduction='none', beta=0.2)
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        # 结合Huber损失和MSE损失
        huber_loss = self.huber(pred, target)
        mse_loss = self.mse(pred, target)
        
        # 动态加权
        weight = 1.0 / (1.0 + torch.abs(pred - target))
        combined_loss = 0.7 * huber_loss + 0.3 * mse_loss
        weighted_loss = combined_loss * weight
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        return weighted_loss

def train_epoch(model, loader, optimizer, criterion, device, epoch, gradient_accumulation_steps=4, max_grad_norm=0.5, ema=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()
    
    # 跟踪每个输出的统计信息
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

            try:
                # 前向传播
                y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                
                # 收集输出统计信息
                for i, pred in enumerate(y_pred):
                    if not torch.isnan(pred).any() and not torch.isinf(pred).any():
                        output_values[i].extend(pred.detach().cpu().numpy().flatten())
                
                # 计算损失
                losses = []
                for i, pred in enumerate(y_pred):
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(-1)
                    
                    # 使用自定义损失
                    loss = criterion(pred.float(), y.float())
                    losses.append(loss)
                
                # 平均损失
                loss = sum(losses) / len(losses)
                
                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss at batch {batch_idx}, skipping")
                    continue
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # 检查梯度
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        param.grad = torch.where(torch.isnan(param.grad) | torch.isinf(param.grad), 
                                               torch.zeros_like(param.grad), 
                                               param.grad)
                        has_nan_grad = True
                
                if has_nan_grad:
                    print(f"Warning: NaN gradients fixed at batch {batch_idx}")
                
                # 每隔gradient_accumulation_steps步更新一次参数
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新EMA
                    if ema is not None:
                        ema.update()
                
                # 记录损失
                total_loss += (loss.item() * gradient_accumulation_steps)
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Current loss: {loss.item() * gradient_accumulation_steps:.4f}")
                
            except RuntimeError as e:
                print(f"Error in forward/backward at batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue
                
        except Exception as e:
            print(f"Error in batch preparation {batch_idx}: {str(e)}")
            optimizer.zero_grad()
            continue
    
    # 处理最后一个不完整的梯度累积
    if len(loader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()
    
    # 更新模型包装器中的输出统计信息
    if isinstance(model, ModelWrapper):
        means = []
        stds = []
        for values in output_values:
            if len(values) > 0:
                values = np.array(values)
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
    """评估模型"""
    model.eval()
    predictions = [[] for _ in range(4)]
    actuals = []
    criterion = MultiTaskLoss(reduction='mean')
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

                try:
                    # 前向传播
                    y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                    
                    # 计算损失
                    batch_losses = []
                    for i, pred in enumerate(y_pred):
                        if len(pred.shape) == 1:
                            pred = pred.unsqueeze(-1)
                        
                        pred_clean = torch.where(torch.isnan(pred) | torch.isinf(pred), 
                                              torch.zeros_like(pred), 
                                              pred)
                        
                        loss = criterion(pred_clean.float(), y.float())
                        batch_losses.append(loss)
                        
                        # 存储预测结果
                        pred_np = pred_clean.cpu().numpy()
                        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=5.0, neginf=-5.0)
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

    # 处理预测结果
    processed_predictions = []
    for preds in predictions:
        preds_array = np.array(preds)
        preds_array = np.nan_to_num(preds_array, nan=0.0, posinf=5.0, neginf=-5.0)
        preds_array = np.clip(preds_array, -5, 5)
        processed_predictions.append(preds_array.tolist())
    
    avg_val_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    return processed_predictions, actuals, avg_val_loss

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        y_true = np.nan_to_num(y_true, nan=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=5.0, neginf=-5.0)
        
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        pearson = stats.pearsonr(y_true, y_pred)[0]
        
        # 检查指标有效性
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
    """测试模型性能"""
    model.eval()
    predictions = [[] for _ in range(4)]
    actuals = []

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
                        
                        pred_clean = torch.where(torch.isnan(pred) | torch.isinf(pred), 
                                              torch.zeros_like(pred), 
                                              pred)
                        
                        pred_np = pred_clean.cpu().numpy()
                        pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=5.0, neginf=-5.0)
                        pred_np = np.clip(pred_np, -5, 5)
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

    # 后处理预测和实际值
    cleaned_predictions = []
    for pred_list in predictions:
        pred_array = np.array(pred_list)
        pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=5.0, neginf=-5.0)
        pred_array = np.clip(pred_array, -5, 5)
        cleaned_predictions.append(pred_array)
    
    actuals_array = np.array(actuals)
    actuals_array = np.nan_to_num(actuals_array, nan=0.0)
    
    # 构建测试预测矩阵
    test_pred_array = np.column_stack(cleaned_predictions)
    test_true_array = actuals_array

    print("\nTest prediction statistics:")
    for i in range(test_pred_array.shape[1]):
        print(f"\nOutput {i+1}:")
        print(f"Mean: {np.mean(test_pred_array[:, i]):.4f}")
        print(f"Std: {np.std(test_pred_array[:, i]):.4f}")
        print(f"Min: {np.min(test_pred_array[:, i]):.4f}")
        print(f"Max: {np.max(test_pred_array[:, i]):.4f}")

    # 标准化数据用于集成学习
    normalized_pred_array, pred_mean, pred_std = normalize_data(test_pred_array)
    normalized_true_array, true_mean, true_std = normalize_data(test_true_array)
    
    # 优化集成模型超参数
    optimal_ensemble_configs = {
        'lasso': {'alpha': 0.01, 'max_iter': 3000},
        'elastic': {'alpha': 0.005, 'l1_ratio': 0.7, 'max_iter': 3000},
        'rf': {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'random_state': random_state},
        'gradientboost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': random_state}
    }
    
    ensemble_models = {
        'lasso': Lasso(**optimal_ensemble_configs['lasso']),
        'elastic': ElasticNet(**optimal_ensemble_configs['elastic']),
        'rf': RandomForestRegressor(**optimal_ensemble_configs['rf']),
        'gradientboost': GradientBoostingRegressor(**optimal_ensemble_configs['gradientboost'])
    }

    print("\nTraining ensemble models...")
    ensemble_results = {}
    
    # 使用完整的集成模型
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
            print(f"\n{name.capitalize()} ensemble results:")
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
    model = ModelWrapper(base_model).to(device)

    # 初始化权重
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

    # 优化器设置
    learning_rate = 3e-5
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(trainLoader) // gradient_accumulation_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=100
    )

    # 使用自定义损失函数
    criterion = MultiTaskLoss(reduction='mean')

    # 初始化EMA
    ema = EMA(model, decay=0.999)

    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    resultLoss = {'losses_train': [], 'losses_val': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练
        train_loss = train_epoch(model, trainLoader, optimizer, criterion, device, epoch + 1, 
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                max_grad_norm=0.5, ema=ema)
        
        resultLoss['losses_train'].append(train_loss)
        print(f'Train avg_loss: {train_loss:.4f}')

        # 如果出现NaN损失，降低学习率并继续
        if math.isnan(train_loss):
            print("NaN loss detected, reducing learning rate by 50% and continuing...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            continue

        # 评估使用EMA权重
        ema.apply_shadow()
        val_predictions, val_actuals, val_loss = evaluate(model, valLoader, device)
        ema.restore()
        
        resultLoss['losses_val'].append(val_loss)
        print(f'Validation avg_loss: {val_loss:.4f}')

        # Early stopping逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            ema.apply_shadow()  # 应用EMA权重
            torch.save(model.state_dict(), f'best_model_{label}.pth')
            ema.restore()  # 恢复原始权重
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
