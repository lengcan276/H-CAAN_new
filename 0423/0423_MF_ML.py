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

# GPU setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training parameters
epochs = 50
batch_size = 16
label = 10000
random_state = 42

# Model configuration
args_model = {
    'num_features_smi': 100,
    'num_features_ecfp': 1024,
    'num_features_x': 78,
    'dropout': 0.1,
    'num_layer': 3,
    'num_heads': 4,
    'hidden_dim': 256,
    'output_dim': 128,
    'n_output': 1,
}

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
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
            
            if batch_idx == 0 and epoch == 1:
                print(f"Original input_ids shape: {input_ids.shape}")
                print(f"Original attention_mask shape: {attention_mask.shape}")
                print(f"Batch size from y: {y.size(0)}")
                print(f"Target shape: {y.shape}")
                print(f"Target values: {y[:5].cpu().numpy()}")
            
            # 处理input_ids维度
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
            
            if batch_idx == 0 and epoch == 1:
                print(f"Processed input_ids shape: {input_ids.shape}")
                print(f"Processed attention_mask shape: {attention_mask.shape}")
            
            # 移动到设备
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 前向传播
            y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
            
            # Debug: 打印预测值
            if batch_idx == 0:
                for i, pred in enumerate(y_pred):
                    if torch.isnan(pred).any():
                        print(f"Warning: NaN in prediction {i}")
                    print(f"Prediction {i} range: {pred.min().item():.4f} to {pred.max().item():.4f}")
            
            # 计算损失
            losses = []
            for i, pred in enumerate(y_pred):
                if len(pred.shape) == 1:
                    pred = pred.unsqueeze(-1)
                # 添加梯度裁剪
                pred = torch.clamp(pred, min=-100, max=100)
                loss = criterion(pred.float(), y.float())
                # Debug: 打印每个损失值
                if batch_idx == 0:
                    print(f"Loss {i}: {loss.item():.4f}")
                losses.append(loss)
            
            loss = sum(losses) / len(losses)
            
            # 检查损失值是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 检查梯度
            if batch_idx == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 10:
                            print(f"Large gradient in {name}: {grad_norm:.4f}")
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
            
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Current loss: {loss.item():.4f}")
            
    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    return avg_loss

def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    predictions = [[] for _ in range(4)]
    actuals = []
    
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
                y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                
                # 存储预测结果
                for i, pred in enumerate(y_pred):
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(-1)
                    pred = torch.clamp(pred, min=-100, max=100)
                    pred_np = pred.cpu().numpy()
                    pred_np = np.clip(pred_np, -100, 100)
                    predictions[i].extend(pred_np.squeeze().tolist())
                actuals.extend(y.cpu().squeeze().tolist())
                
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
            
    return predictions, actuals

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    return {
        'rmse': np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'r2': metrics.r2_score(y_true, y_pred),
        'pearson': stats.pearsonr(y_true, y_pred)[0]
    }

def test_model(model, testLoader, device, base_path, label, batch_size, epochs, random_state):
    """测试模型性能"""
    model.eval()
    total_loss = 0
    n_batches = 0
    predictions = [[] for _ in range(4)]
    actuals = []
    criterion = nn.MSELoss(reduction='mean')
    
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
                y_pred = model(encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask)
                
                # 计算损失
                batch_losses = []
                for i, pred in enumerate(y_pred):
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(-1)
                    pred = torch.clamp(pred, min=-100, max=100)
                    batch_losses.append(criterion(pred.float(), y.float()))
                    pred_np = pred.cpu().numpy()
                    pred_np = np.clip(pred_np, -100, 100)
                    predictions[i].extend(pred_np.squeeze().tolist())
                
                loss = sum(batch_losses) / len(batch_losses)
                total_loss += loss.item()
                n_batches += 1
                actuals.extend(y.cpu().squeeze().tolist())
                
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {str(e)}")
                continue
            
            if batch_idx % 10 == 0:
                print(f"Processed test batch: {batch_idx}")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    print(f"\nTest average loss: {avg_loss:.4f}")
    
    test_pred_array = np.column_stack([np.array(pred) for pred in predictions])
    test_true_array = np.array(actuals)
    
    test_pred_array = np.nan_to_num(test_pred_array, nan=0.0, posinf=100, neginf=-100)
    test_pred_array = np.clip(test_pred_array, -100, 100)
    
    print("\nTest prediction statistics:")
    for i in range(test_pred_array.shape[1]):
        print(f"\nOutput {i+1}:")
        print(f"Mean: {np.mean(test_pred_array[:, i]):.4f}")
        print(f"Std: {np.std(test_pred_array[:, i]):.4f}")
        print(f"Min: {np.min(test_pred_array[:, i]):.4f}")
        print(f"Max: {np.max(test_pred_array[:, i]):.4f}")
    
    # 集成学习
    ensemble_models = {
        'lasso': Lasso(alpha=0.1),
        'elastic': ElasticNet(alpha=0.5, l1_ratio=0.5),
        'rf': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'gradientboost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                                  max_depth=3, random_state=random_state)
    }
    
    print("\nTraining ensemble models...")
    ensemble_results = {}
    for name, reg_model in ensemble_models.items():
        try:
            # 训练集成模型
            reg_model.fit(test_pred_array, test_true_array)
            
            # 预测
            ensemble_pred = reg_model.predict(test_pred_array)
            
            # 计算指标
            metrics = calculate_metrics(test_true_array, ensemble_pred)
            ensemble_results[name] = {
                'predictions': ensemble_pred,
                'metrics': metrics,
                'weights': reg_model.coef_ if hasattr(reg_model, 'coef_') else reg_model.feature_importances_
            }
            
            # 保存结果
            result_df = pd.DataFrame({
                'y_true': test_true_array,
                'y_pred': ensemble_pred
            })
            result_df.to_csv(f'{base_path}/{label}_test_result_{name}_{batch_size}_{epochs}_{random_state}.csv', 
                            index=False)
            
            # 打印结果
            print(f"\n{name.capitalize()} ensemble results:")
            print(f"Weights: {ensemble_results[name]['weights']}")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # 绘制散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(test_true_array, ensemble_pred, alpha=0.5)
            plt.plot([min(test_true_array), max(test_true_array)], 
                    [min(test_true_array), max(test_true_array)], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'{name.capitalize()} Ensemble Predictions vs True Values')
            plt.savefig(f'{base_path}/{label}_test_scatter_{name}_{batch_size}_{epochs}_{random_state}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error in {name} ensemble: {str(e)}")
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
    model = comModel(args_model).to(device)
    
    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
    
    model.apply(init_weights)
    
    # 优化器设置
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    criterion = nn.MSELoss(reduction='mean')
    
    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    resultLoss = {'losses_train': [], 'losses_val': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练
        train_loss = train_epoch(model, trainLoader, optimizer, criterion, device, epoch + 1)
        resultLoss['losses_train'].append(train_loss)
        print(f'Train avg_loss: {train_loss:.4f}')
        
        if math.isnan(train_loss):
            print("NaN loss detected, reducing learning rate and retrying...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            continue
        
        # 验证
        val_predictions, val_actuals = evaluate(model, valLoader, device)
        val_loss = sum(criterion(torch.tensor(pred), torch.tensor(val_actuals)) 
                       for pred in val_predictions) / len(val_predictions)
        resultLoss['losses_val'].append(val_loss.item())
        print(f'Validation avg_loss: {val_loss:.4f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # Early stopping
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
        
        # 保存集成结果
        for name, results in ensemble_results.items():
            np.save(f'{base_path}/{label}_test_weights_{name}_{batch_size}_{epochs}_{random_state}.npy', 
                    results['weights'])
    else:
        print("No saved model found. Please train the model first.")
    
    print("\nTraining and evaluation completed successfully!")
