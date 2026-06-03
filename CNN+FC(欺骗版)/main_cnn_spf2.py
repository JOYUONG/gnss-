# CNN+分支+BN处理+FC模型、1个干扰源
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import geshi_cnn_spf2 as geshiann

import logging
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment 
from tqdm.auto import tqdm

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 定义全局区域尺寸变量
global area_size,number_receiver
# 确保area_size是全局变量
area_size = 4000  # 全局区域大小，单位：米
number_receiver = 1200  # 接收机数量

# 定义计算区域对角线长度的函数
def calculate_area_diagonal(x_min, x_max, y_min, y_max):
    """计算区域对角线长度，作为动态归一化因子"""
    width = x_max - x_min
    height = y_max - y_min
    return np.sqrt(width**2 + height**2)

# 自定义数据集类
class InterferenceDataset(Dataset):
    """自定义数据集类 - 修改为每个干扰源一个样本"""
    def __init__(self, h5_path):
        self.samples = []
        self.targets = []  # 只存储位置
        self.wcl_pos = []
        self.grid_transforms = []
        self.area_diagonals = []
        
        # 读取HDF5文件
        with h5py.File(h5_path, 'r') as f:
            # 获取所有样本组
            # 遍历所有样本组
            for key in tqdm(f.keys(), desc="加载数据集"):
                if not key.startswith('sample_'):
                    continue
                # 加载数据
                group = f[key]
                receivers = group['receivers'][:]
                jammers = group['jammers'][:]
                
                # 转换数据结构
                receiver_list = []
                for rec in receivers:
                    receiver_list.append({
                        'x': rec[0],
                        'y': rec[1],
                        'ri_pw': rec[2] if len(rec) > 2 else 1.0  # 假设功率为1
                    })
                
                # 筛选受干扰的接收机（ri_pw > 0）
                affected_receivers = [rec for rec in receiver_list if rec['ri_pw'] > 0]
                
                # 如果没有受干扰的接收机，跳过这个样本
                if len(affected_receivers) == 0:
                    continue
                
                # 计算区域中心和对角线长度作为归一化因子
                X_min = min([rec['x'] for rec in affected_receivers])
                X_max = max([rec['x'] for rec in affected_receivers])
                Y_min = min([rec['y'] for rec in affected_receivers])
                Y_max = max([rec['y'] for rec in affected_receivers])
                
                # 扩展边界以确保覆盖可能的干扰源位置
                X_min -= 100
                X_max += 100
                Y_min -= 100
                Y_max += 100
                
                # 确保在全局范围内
                X_min = max(0, X_min)
                X_max = min(area_size, X_max)
                Y_min = max(0, Y_min)
                Y_max = min(area_size, Y_max)
                
                # 计算区域范围和中心位置
                X_center = (X_min + X_max) / 2  # 区域中心x坐标
                Y_center = (Y_min + Y_max) / 2  # 区域中心y坐标
                
                # 计算区域对角线长度作为动态归一化因子
                area_diagonal = calculate_area_diagonal(X_min, X_max, Y_min, Y_max)
                # 确保对角线长度不为0
                area_diagonal = max_diagonal = max(1.0, area_diagonal)
                self.area_diagonals.append(area_diagonal)
                
                # 创建位置矩阵
                FIXED_GRID_CELL_SIZE = 10  # 米
                M2 = np.zeros((100, 100), dtype=np.float32)  # 干扰功率矩阵
                
                # 计算需要的网格数量
                X_range = X_max - X_min
                Y_range = Y_max - Y_min
                num_grid_x = max(1, int(X_range / FIXED_GRID_CELL_SIZE) + 20)  # 增加边界
                num_grid_y = max(1, int(Y_range / FIXED_GRID_CELL_SIZE) + 20)
                
                # 填充M2矩阵（受干扰接收机位置和功率）
                if num_grid_x > 100 or num_grid_y > 100:
                    # 计算降采样因子
                    scale_factor = max(num_grid_x/100, num_grid_y/100)
                    adjusted_cell_size = FIXED_GRID_CELL_SIZE * scale_factor
                    # 存储调整后的网格参数
                    self.grid_transforms.append([X_center, Y_center, adjusted_cell_size])
                    # 填充M2矩阵
                    for i, receiver in enumerate(affected_receivers):
                        x, y = receiver['x'], receiver['y']
                        jam_pw = receiver['ri_pw']
                        
                        # 计算降采样后的网格索引 (0-99)
                        i_idx = min(int((y - Y_min)/adjusted_cell_size), 99)
                        j_idx = min(int((x - X_min)/adjusted_cell_size), 99)
                        
                        M2[i_idx, j_idx] += jam_pw
                else:
                    L = FIXED_GRID_CELL_SIZE
                    self.grid_transforms.append([X_center, Y_center, L])
                    for i, receiver in enumerate(affected_receivers):
                        x, y = receiver['x'], receiver['y']
                        jam_pw = receiver['ri_pw']
                        
                        # 计算网格索引，考虑padding
                        i_idx = min(int((y - Y_min)/FIXED_GRID_CELL_SIZE) , 99)
                        j_idx = min(int((x - X_min)/FIXED_GRID_CELL_SIZE) , 99)
                        
                        M2[i_idx, j_idx] += jam_pw
                
                # 存储转换后的矩阵
                self.samples.append(M2)
                
                # 存储目标值（只存储位置）
                # 相对归一化：相对于区域中心，使用区域对角线长度
                position = np.array([(jammers[0][0] - X_center)/area_diagonal, 
                                       (jammers[0][1] - Y_center)/area_diagonal], dtype=np.float32)
                self.targets.append(position)
                
                # 计算加权质心位置 (这里假设ri_pw=1，所以是算术平均)
                weighted_x = np.mean([rec['x'] for rec in affected_receivers])
                weighted_y = np.mean([rec['y'] for rec in affected_receivers])
                # 归一化WCL位置
                wcl_position = [(weighted_x - X_center)/area_diagonal, (weighted_y - Y_center)/area_diagonal]
                self.wcl_pos.append(wcl_position)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        M2 = self.samples[idx]
        position = self.targets[idx]
        
        # 转换为PyTorch张量并添加通道维度
        M2_tensor = torch.tensor(M2).unsqueeze(0).float()  # 形状: (1, 100, 100)
        
        # 确保position始终是形状为(2,)的一维张量
        position_tensor = torch.tensor(position).float().reshape(2)
        
        # 确保wcl_position始终是形状为(2,)的一维张量
        wcl_position = self.wcl_pos[idx]
        wcl_position_tensor = torch.tensor(wcl_position, dtype=torch.float32).reshape(2)
        
        # 获取网格变换参数和区域对角线长度
        if idx < len(self.grid_transforms):
            X_center, Y_center, L = self.grid_transforms[idx]
        else:
            X_center = Y_center = L = 0
        
        # 获取区域对角线长度
        if idx < len(self.area_diagonals):
            area_diagonal = self.area_diagonals[idx]
        else:
            area_diagonal = area_size
        
        # 确保grid_params始终是形状为(4,)的一维张量
        grid_params = torch.tensor([X_center/float(area_diagonal), Y_center/float(area_diagonal), 
                                  L/float(area_diagonal), area_diagonal], dtype=torch.float32).reshape(4)
        
        return M2_tensor, position_tensor, wcl_position_tensor, grid_params

# CNN卷积降维模型
class CNNReduction(nn.Module):
    def __init__(self):
        super(CNNReduction, self).__init__()
        
        # M2的卷积路径
        self.conv1_M2 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.pool1_M2 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv2_M2 = nn.Conv2d(20, 3, kernel_size=3, stride=1, padding=1)
        self.pool2_M2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, M2):
        # 处理M2
        M2 = torch.relu(self.conv1_M2(M2))
        M2 = self.pool1_M2(M2)  # 输出: (batch, 20, 20, 20)
        M2 = torch.relu(self.conv2_M2(M2))
        MC2 = self.pool2_M2(M2)  # 输出: (batch, 3, 10, 10)
        
        return MC2

# 完整模型（包含分支、BN处理和DNN）
class InterferenceModel(nn.Module):
    def __init__(self):
        super(InterferenceModel, self).__init__()
        
        # 卷积降维路径
        self.cnn_reduction = CNNReduction()
        
        # BN层（用于区域归一化处理）
        self.bn = nn.BatchNorm2d(3)
        
        # 分支参数
        self.K2 = nn.Parameter(torch.randn(1, 3, 1, 1))
        self.B1 = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.K4 = nn.Parameter(torch.randn(1, 3, 1, 1))
        self.B2 = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # 特征展平后的维度
        self.flat_features = 3 * 10 * 10  # 卷积后的特征图大小
        
        # 原始路径
        self.fc_A = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 置信度估计分支 - 保留用于兼容性
        self.confidence_fc = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()  # 置信度在[0,1]之间
        )
        
        # 增强版A路（包含全局坐标系下的相对位置编码）
        self.fc_A_enhanced = nn.Sequential(
            nn.Linear(self.flat_features + 4, 128),  # 额外添加4个网格参数特征（新增区域对角线长度）
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 位置(x,y)
            nn.Tanh()          # 使用Tanh替代Sigmoid，输出范围为[-1,1]，适应相对归一化
        )
        
        # 增强版置信度估计分支（包含全局坐标系下的相对位置编码）
        self.confidence_fc_enhanced = nn.Sequential(
            nn.Linear(self.flat_features + 4, 64),  # 额外添加4个网格参数特征（新增区域对角线长度）
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()  # 置信度在[0,1]之间
        )

    def forward(self, M2, wcl_pos, grid_params):
        # 通过CNN降维
        MC2 = self.cnn_reduction(M2)
        
        # 通过BN层
        MN2 = self.bn(MC2)
        
        # 分支A处理,只对A路使用BN处理
        A = MN2 * self.K2 + self.B1
        A = A.view(-1, self.flat_features)  # 展平
        
        # 如果提供了网格参数，将其整合到特征中（全局坐标系下的相对位置编码）
        if grid_params is not None and wcl_pos is not None:
            # 将网格参数添加到特征向量中
            A_enhanced = torch.cat([A, grid_params], dim=1)
            pos_pred_cnn = self.fc_A_enhanced(A_enhanced)
        else:
            # 回退到原始方法
            pos_pred_cnn = self.fc_A(A)
            return pos_pred_cnn
        
        # 计算置信度
        if grid_params is not None:
            alpha = self.confidence_fc_enhanced(A_enhanced)
        else:
            alpha = self.confidence_fc(A)
        
        # 自适应位置估计: X_final = α ⊙ X_CNN + (1 - α) ⊙ X_WCL
        pos_pred_final = alpha * pos_pred_cnn + (1 - alpha) * wcl_pos
        return pos_pred_final, pos_pred_cnn, alpha

# 训练模型函数
def train_model():
    # 参数设置
    h5_path = "training_data_cnn_spf2.h5"
    num_samples = 1000
    batch_size = 64
    if not os.path.exists(h5_path):
        print("正在生成训练数据...")
        generate_training_data(h5_path, num_samples)
    
    # 初始化模型
    model = InterferenceModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 准备数据集
    dataset = InterferenceDataset(h5_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    pos_criterion = nn.MSELoss()    # 平滑L1损失函数,分段使用均方误差和平均绝对误差

    # 训练阶段1：降低Loss1（位置预测）
    print("=== 阶段1:训练位置预测(Loss1)===")
    best_val_loss = float('inf')
    epochs = 10
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for M2, pos_labels, wcl_pos, grid_params in train_loader:
            M2 = M2.to(device)
            pos_labels = pos_labels.to(device)
            wcl_pos = wcl_pos.to(device)
            grid_params = grid_params.to(device)
            
            optimizer.zero_grad()
            # 使用L_CAGE损失函数：0.7*L_GNN + 0.3*L_adapt
            pos_pred_final, pos_pred_cnn, _ = model(M2, wcl_pos, grid_params)
            # 第一部分：CNN预测的损失
            loss1 = pos_criterion(pos_pred_cnn, pos_labels)
            # 第二部分：融合位置的损失
            loss_adapt = pos_criterion(pos_pred_final, pos_labels)
            loss = 0.7 * loss1 + 0.3 * loss_adapt
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, wcl_pos, grid_params in val_loader:
                M2 = M2.to(device)
                pos_labels = pos_labels.to(device)
                wcl_pos = wcl_pos.to(device)
                grid_params = grid_params.to(device)
                
                pos_pred_final, pos_pred_cnn, _ = model(M2, wcl_pos, grid_params)
                loss1 = pos_criterion(pos_pred_cnn, pos_labels)
                loss_adapt = pos_criterion(pos_pred_final, pos_labels)
                loss = 0.7 * loss1 + 0.3 * loss_adapt
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_cnnspfmodel2_stage.pth")
            print("模型已更新！")
    
    # 打印训练完成信息
    print("训练完成！")

class NeuralLocalizer:
    def __init__(self):
        # 初始化模型
        self.model = InterferenceModel().to(device)
        self.model.load_state_dict(torch.load('best_cnnspfmodel2_stage.pth', map_location=device))
        self.model.eval()
        
    def predict(self, receivers):
        """根据受干扰的接收机数据预测干扰源位置"""
        # 检查是否有受干扰的接收机
        interfered = [rec for rec in receivers if rec['ri_pw'] > 0]
        if not interfered:
            return [0.0, 0.0]
        
        # 计算加权质心（权重为干扰功率）
        # 由于ri_pw=1，简单计算算术平均
        weighted_x = np.mean([rec['x'] for rec in interfered])
        weighted_y = np.mean([rec['y'] for rec in interfered])
        
        # 计算受干扰区域范围
        X_min = np.min([rec['x'] for rec in interfered])
        X_max = np.max([rec['x'] for rec in interfered])
        Y_min = np.min([rec['y'] for rec in interfered])
        Y_max = np.max([rec['y'] for rec in interfered])
        
        # 扩展边界以确保覆盖可能的干扰源位置
        X_min -= 100
        X_max += 100
        Y_min -= 100
        Y_max += 100
        
        # 确保在全局范围内
        X_min = max(0, X_min)
        X_max = min(area_size, X_max)
        Y_min = max(0, Y_min)
        Y_max = min(area_size, Y_max)
        
        # 计算区域范围和中心位置
        X_center = (X_min + X_max) / 2  # 区域中心x坐标
        Y_center = (Y_min + Y_max) / 2  # 区域中心y坐标
        
        # 计算区域对角线长度作为动态归一化因子
        area_diagonal = calculate_area_diagonal(X_min, X_max, Y_min, Y_max)
        # 确保对角线长度不为0
        area_diagonal = max(1.0, area_diagonal)
        
        # 归一化WCL位置
        wcl_position = [(weighted_x - X_center)/area_diagonal, (weighted_y - Y_center)/area_diagonal]
        
        # 1. 创建位置矩阵 M1 和 M2
        FIXED_GRID_CELL_SIZE = 10  # 米，根据区域大小自动调整网格单元大小
        M2 = np.zeros((100, 100), dtype=np.float32)  # 干扰功率矩阵
        
        # 计算需要的网格数量，确保覆盖整个受干扰区域
        X_range = X_max - X_min
        Y_range = Y_max - Y_min
        num_grid_x = max(1, int(X_range / FIXED_GRID_CELL_SIZE) + 20)  # 增加边界
        num_grid_y = max(1, int(Y_range / FIXED_GRID_CELL_SIZE) + 20)
        
        # 填充M2矩阵（受干扰接收机位置和功率）
        if num_grid_x > 100 or num_grid_y > 100:
            # 计算降采样因子
            scale_factor = max(num_grid_x/100, num_grid_y/100)
            adjusted_cell_size = FIXED_GRID_CELL_SIZE * scale_factor
            # 填充M2矩阵
            for i, receiver in enumerate(interfered):
                x, y = receiver['x'], receiver['y']
                jam_pw = receiver['ri_pw']
                
                # 计算降采样后的网格索引 (0-99)
                i_idx = min(int((y - Y_min)/adjusted_cell_size), 99)
                j_idx = min(int((x - X_min)/adjusted_cell_size), 99)
                
                M2[i_idx, j_idx] += jam_pw
        else:
            # 使用原始网格大小
            for i, receiver in enumerate(interfered):
                x, y = receiver['x'], receiver['y']
                jam_pw = receiver['ri_pw']
                
                # 计算网格索引
                i_idx = min(int((y - Y_min)/FIXED_GRID_CELL_SIZE), 99)
                j_idx = min(int((x - X_min)/FIXED_GRID_CELL_SIZE), 99)
                
                M2[i_idx, j_idx] += jam_pw
        
        # 转换为PyTorch张量并添加通道维度
            M2_tensor = torch.tensor(M2).unsqueeze(0).unsqueeze(0).float().to(device)   # 形状: (1, 1, 100, 100)
            
            # 准备其他参数
            wcl_pos_tensor = torch.tensor(wcl_position, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 归一化网格变换参数
            grid_params = torch.tensor([X_center/float(area_diagonal), Y_center/float(area_diagonal), 
                                      FIXED_GRID_CELL_SIZE/float(area_diagonal), area_diagonal], 
                                      dtype=torch.float32).unsqueeze(0).to(device)
            
            # 模型预测  
            with torch.no_grad():
                pos_pred_final, _, _ = self.model(M2_tensor, wcl_pos_tensor, grid_params)
                # 转换为NumPy数组
                pos_pred_final = pos_pred_final.numpy()[0]  # 形状: (2,)
            
            # 反归一化位置坐标 (相对归一化 -> 实际坐标)
            pos_pred_actual = np.array([
                pos_pred_final[0] * area_diagonal + X_center,
                pos_pred_final[1] * area_diagonal + Y_center
            ])
            
            return pos_pred_actual
    
def generate_training_data(h5_path, num_samples):
    """预生成训练数据"""
    if os.path.exists(h5_path):
        os.remove(h5_path)  # 强制重新生成数据
    
    with h5py.File(h5_path, 'w') as f:
        with tqdm(total=num_samples, desc="生成训练数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, jammers = geshiann.generation_data(
                    x_min=0, x_max = area_size, y_min=0, y_max=area_size,
                    num_receivers = number_receiver, num_interferers=1
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('jammers', data=jammers)
                pbar.update(1)
                pbar.set_postfix(last_sample=f"接收机:{len(receivers)} 干扰源:{len(jammers)}")

def calculate_paired_errors(true_sources, estimated_points):
    """增强版误差计算"""
    # 矩阵化计算所有距离
    true_sources = np.array(true_sources)  # 转换为 NumPy 数组
    estimated_points = np.array(estimated_points)  # 转换为 NumPy 数组

    dist_matrix = np.linalg.norm(
        true_sources[:, np.newaxis, :2] - estimated_points[np.newaxis, :, :2],
        axis=2
    )
    
    # 使用匈牙利算法寻找最优匹配
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # 收集匹配误差
    matched_errors = dist_matrix[row_ind, col_ind]
    
    # 处理未匹配点
    return {
        'matched_pairs': len(matched_errors),
        'distances': matched_errors.tolist(),
        'unmatched_true': len(true_sources) - len(matched_errors),
        'unmatched_est': len(estimated_points) - len(matched_errors)
    }

def generate_data(path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_interferers):
    """生成测试数据"""
    if os.path.exists(path):
        os.remove(path)  # 强制重新生成数据
    
    with h5py.File(path, 'w') as f:
        with tqdm(total=num_samples, desc="生成测试数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, jam_pos = geshiann.generation_data(
                    x_min, x_max, y_min, y_max,
                    num_receivers, num_interferers
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('jammers', data=jam_pos)
                pbar.update(1)
                pbar.set_postfix(last_sample=f"接收机:{len(receivers)} 干扰源:{len(jam_pos)}")

def calculate_paired_errors_pw(true_powers, estimated_powers):
    """增强版误差计算"""
    # 矩阵化计算所有距离
    true_powers = np.array(true_powers)  # 转换为 NumPy 数组
    estimated_powers = np.array(estimated_powers)  # 转换为 NumPy 数组

    dist_matrix = np.abs(true_powers - estimated_powers)
    
    return dist_matrix

# --------------------- 主程序部分 ---------------------
if __name__ == "__main__":
    # 训练模型
    #train_model()
    
    # 初始化定位器
    locator = NeuralLocalizer()
    
    # 仿真参数
    num_simulations = 1000
    x_min, x_max = 0, area_size
    y_min, y_max = 0, area_size
    num_interferers = 1
    num_receivers = number_receiver

    # 统计误差
    all_errors = []
    error2 = []

    with tqdm(range(num_simulations), desc="运行仿真", unit="sim") as sim_pbar:
        for sim in sim_pbar:
            # 生成测试数据
            test_path = "test_data-cnnspf2.h5"
            num_samples = 1     
            generate_data(test_path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_interferers)

            #数据加载与处理
            with h5py.File(test_path, 'r') as f:
                # 遍历所有样本组
                for sample_key in f:
                    if not sample_key.startswith('sample_'):
                        continue
                    
                    sample = f[sample_key]
                    receivers = sample['receivers'][:]
                    true_jammers = sample['jammers'][:]
                    
                    # 转换数据格式
                    receiver_list = []
                    for rec in receivers:
                        receiver_list.append({
                            'x': rec[0],
                            'y': rec[1],
                            'ri_pw': rec[2] if len(rec) > 2 else 0.0
                        })
                    
                    # 使用模型进行预测
                    predicted_position = locator.predict(receiver_list)
                    
                    # 计算误差
                    errors = calculate_paired_errors(true_jammers, [predicted_position])
                    all_errors.extend(errors['distances'])
                    """ 
                    # 可视化结果
                    plt.figure(figsize=(10, 8))
                    
                    # 绘制真实的干扰源位置
                    true_x, true_y = true_jammers[0][0], true_jammers[0][1]
                    plt.scatter(true_x, true_y, marker='x', color='red', s=100, label='真实干扰源')
                    
                    # 绘制预测的干扰源位置
                    pred_x, pred_y = predicted_position[0], predicted_position[1]
                    plt.scatter(pred_x, pred_y, marker='o', color='blue', s=100, label='预测干扰源')
                    
                    # 绘制正常工作的接收机
                    normal_recv = [rec for rec in receiver_list if rec['ri_pw'] == 0]
                    if normal_recv:
                        normal_x = [rec['x'] for rec in normal_recv]
                        normal_y = [rec['y'] for rec in normal_recv]
                        plt.scatter(normal_x, normal_y, marker='.', color='green', s=50, label='正常接收机')
                    
                    # 绘制受干扰的接收机
                    jammed_recv = [rec for rec in receiver_list if rec['ri_pw'] > 0]
                    if jammed_recv:
                        jammed_x = [rec['x'] for rec in jammed_recv]
                        jammed_y = [rec['y'] for rec in jammed_recv]
                        plt.scatter(jammed_x, jammed_y, marker='+', color='orange', s=50, label='受干扰接收机')
                    
                    # 添加标签和图例
                    plt.title(f'干扰源定位结果 (仿真 {sim+1})')
                    plt.xlabel('X坐标 (米)')
                    plt.ylabel('Y坐标 (米)')
                    plt.grid(True)
                    plt.legend()
                    plt.axis([0, area_size, 0, area_size])
                    plt.tight_layout()
                    
                    # 显示图形
                    plt.show()
                     """
                    # 更新进度条
                    sim_pbar.set_postfix(last_error=f"{errors['distances'][0]:.2f}m")
    
    # 计算统计数据
    if all_errors:
        max_error = max(all_errors)
        min_error = min(all_errors)
        avg_error = sum(all_errors) / len(all_errors)
        rmse = np.sqrt(np.mean(np.square(all_errors)))
        
        # 输出统计结果
        print(f"\n仿真结果统计 ({num_simulations} 次仿真):")
        print(f"最大误差: {max_error:.2f} 米")
        print(f"最小误差: {min_error:.2f} 米")
        print(f"平均误差: {avg_error:.2f} 米")
        print(f"均方根误差 (RMSE): {rmse:.2f} 米")
        
        # 绘制误差分布直方图
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(all_errors, bins=10, range=(0, 100), density=True, alpha=0.75)
        plt.xlabel('定位误差 (米)')
        plt.ylabel('频率')
        plt.title('干扰源定位误差分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("没有有效的误差数据可供分析")
   

