# CNN+WCL模型，干扰源1个
# 位置预测精度10米级，功率预测1dBm级
# 固定网格单元大小，自适应区域选择
# 区域过大时采用的是扩大网格单元大小
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
#import geshi_ann as geshiann
#import map5_data3 as geshiann
import map6_data as geshiann
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

# 定义全局区域尺寸变量
global area_size
# 确保area_size是全局变量
area_size = 4500  # 全局区域大小，单位：米

# 定义计算区域对角线长度的函数
def calculate_area_diagonal(x_min, x_max, y_min, y_max):
    """计算区域对角线长度，作为动态归一化因子"""
    width = x_max - x_min
    height = y_max - y_min
    return np.sqrt(width**2 + height**2)

#print(f"Using device: {device}")

# 自定义数据集类
class InterferenceDataset(Dataset):
    """自定义数据集类 - 修改为每个干扰源一个样本"""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.samples = []  # 存储(M1, M2)元组
        self.targets = []  # 存储目标值(位置和功率)
        self.wcl_pos = []  # 存储WCL位置
        self.grid_transforms = []  # 存储网格变换参数 (用于全局坐标系下的相对位置编码)
        self.area_diagonals = []  # 存储每个样本的区域对角线长度，作为动态归一化因子
        
        # 数据加载与处理
        with h5py.File(h5_path, 'r') as f:
            # 遍历所有样本组
            for key in tqdm(f.keys(), desc="加载数据集"):
                if not key.startswith('sample_'):
                    continue
                      
                group = f[key]
                receivers = group['receivers'][:]
                jammers = group['jammers'][:]
                jammers_pw = group['jammers_pw'][:]
                # 转换字节字符串为普通字符串
                receivers = receivers.astype([
                    ('x', 'f4'), ('y', 'f4'), 
                    ('ri_pw', 'f4')
                ])
                #affected_mask = (receivers['ri_pw'] != -np.inf) & (receivers['ri_pw'] >= -80)
                affected_mask = receivers['ri_pw'] >= -80
                affected_receivers = receivers[affected_mask]
                # 计算加权质心（权重为干扰功率）
                weights = np.power(10, affected_receivers['ri_pw'] / 10.0)  # 功率转换成线性值
                total_weight = np.sum(weights)
                
                # 防止除零错误
                if len(affected_receivers) > 0 and total_weight > 0:
                    weighted_x = np.sum(affected_receivers['x'] * weights) / total_weight
                    weighted_y = np.sum(affected_receivers['y'] * weights) / total_weight
                else:
                    # 当没有有效的受影响接收器或权重总和为0时，使用默认值
                    weighted_x = np.mean(receivers['x']) if len(receivers) > 0 else 0
                    weighted_y = np.mean(receivers['y']) if len(receivers) > 0 else 0
                
                # 1. 创建位置矩阵 M1 和 M2
                # 使用固定网格单元大小，自适应区域选择
                FIXED_GRID_CELL_SIZE = 5  # 米，根据区域大小自动调整网格单元大小
                
                # 计算受干扰区域范围
                if len(affected_receivers) > 0:
                    X_min = np.min(affected_receivers['x'])
                    X_max = np.max(affected_receivers['x'])
                    Y_min = np.min(affected_receivers['y'])
                    Y_max = np.max(affected_receivers['y'])
                    
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
                    X_range = X_max - X_min
                    Y_range = Y_max - Y_min
                    X_center = (X_min + X_max) / 2  # 区域中心x坐标
                    Y_center = (Y_min + Y_max) / 2  # 区域中心y坐标
                    
                    # 计算区域对角线长度作为动态归一化因子
                    area_diagonal = calculate_area_diagonal(X_min, X_max, Y_min, Y_max)
                    # 确保对角线长度不为0
                    area_diagonal = max(1.0, area_diagonal)
                    
                    # 归一化WCL位置
                    self.wcl_pos.append([(weighted_x - X_center)/area_diagonal, (weighted_y - Y_center)/area_diagonal])
                    
                    # 记录区域对角线长度
                    self.area_diagonals.append(area_diagonal)
                    
                    # 记录区域中心位置
                    #self.grid_transforms.append([X_center, Y_center, FIXED_GRID_CELL_SIZE])
                else:
                    # 如果没有受干扰的接收机，使用默认值
                    self.wcl_pos.append([0.0, 0.0])
                    self.area_diagonals.append(area_size)  # 默认使用全局区域大小
                    self.grid_transforms.append([0.0, 0.0, FIXED_GRID_CELL_SIZE])
                    
                
                # 计算需要的网格数量，确保覆盖整个受干扰区域
                num_grid_x = max(1, int(X_range / FIXED_GRID_CELL_SIZE) + 20)  # 增加边界
                num_grid_y = max(1, int(Y_range / FIXED_GRID_CELL_SIZE) + 20)
                
                # 初始化干扰功率矩阵
                M2 = np.zeros((100, 100), dtype=np.float32)
                
                if num_grid_x > 100 or num_grid_y > 100:
                    # 计算降采样因子
                    scale_factor = max(num_grid_x/100, num_grid_y/100)
                    adjusted_cell_size = FIXED_GRID_CELL_SIZE * scale_factor
                    self.grid_transforms.append([X_center, Y_center, adjusted_cell_size])
                    # 填充M2矩阵（受干扰接收机位置和功率）
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
                
                # 存储目标值（干扰源位置和功率）
                # 这里假设每个场景只有一个干扰源，取第一个干扰源的位置和功率
               
                # 对于有效干扰区域，使用相对归一化；否则使用全局归一化
                if len(affected_receivers) > 0:
                    # 相对归一化：相对于区域中心，使用区域对角线长度
                    position = np.array([(jammers[0] - X_center)/area_diagonal, 
                                            (jammers[1] - Y_center)/area_diagonal], dtype=np.float32)
                    # 功率归一化（假设最大功率为1000）
                    power = np.array([jammers_pw/100.0], dtype=np.float32) 
                    self.targets.append((position, power))
                else:
                    print("无有效受影响接收机")
                    # 添加默认目标值以保持samples和targets数组长度一致
                    default_position = np.array([0.0, 0.0], dtype=np.float32)
                    default_power = np.array([0.0], dtype=np.float32)
                    self.targets.append((default_position, default_power))
                    continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        M2 = self.samples[idx]
        position, power = self.targets[idx]
        
        # 转换为PyTorch张量并添加通道维度
        M2_tensor = torch.tensor(M2).unsqueeze(0)  # 形状: (1, 100, 100)
        
        # 目标值(干扰源位置、功率)
        position_tensor = torch.tensor(position)
        # 确保power_tensor始终是标量形状 [1]
        power_tensor = torch.tensor(power).view(1)
        
        # 计算WCL位置
        wcl_position = self.wcl_pos[idx]
        wcl_position_tensor = torch.tensor(wcl_position, dtype=torch.float32)
        
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
        
        # 归一化网格变换参数
        grid_params = torch.tensor([X_center/float(area_diagonal), Y_center/float(area_diagonal), 
                                  L/float(area_diagonal), area_diagonal], dtype=torch.float32)
        
        return M2_tensor, position_tensor, power_tensor, wcl_position_tensor, grid_params

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
    def __init__(self, cnn_output_channels=3, cnn_output_size=10):
        super(InterferenceModel, self).__init__()
        
        # CNN降维部分 - 确保输出指定维度
        self.cnn_reduction = CNNReduction()
        
        # 特征维度参数
        self.c = cnn_output_channels
        self.h = cnn_output_size
        self.w = cnn_output_size
        self.flat_features = self.c * self.h * self.w
        
        # BN处理 (单BN层即可)
        self.bn = nn.BatchNorm2d(self.c)
        
        # 分支A（位置预测）的参数
        self.K1 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 
        self.K2 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 
        self.B1 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 
        
        # 分支B（功率预测）的参数
        self.K3 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 
        self.K4 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 
        self.B2 = nn.Parameter(torch.randn(self.c, self.h, self.w)) 

        
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
        
        # 增强版B路（包含全局坐标系下的相对位置编码）
        self.fc_B_enhanced = nn.Sequential(
            nn.Linear(self.flat_features + 4, 128),  # 额外添加4个网格参数特征（新增区域对角线长度）
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 只需1个输出值 (功率)
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

    def forward(self, M2, wcl_pos, grid_params=None):
        # 通过CNN降维
        MC2 = self.cnn_reduction(M2)
        
        # 通过BN层
        MN2 = self.bn(MC2)
        
        # 分支A处理,只对A路使用BN处理
        A = MN2 * self.K2 + self.B1
        A = A.view(-1, self.flat_features)  # 展平
        
        # 如果提供了网格参数，将其整合到特征中（全局坐标系下的相对位置编码）
        if grid_params is not None:
            # 将网格参数添加到特征向量中
            A_enhanced = torch.cat([A, grid_params], dim=1)
            pos_pred_cnn = self.fc_A_enhanced(A_enhanced)
        else:
            # 回退到原始方法
            pos_pred_cnn = self.fc_A(A)
        
        # 分支B处理
        B = MC2 * self.K4 + self.B2
        B = B.view(-1, self.flat_features)  # 展平
        
        # 如果提供了网格参数，将其整合到特征中
        if grid_params is not None:
            B_enhanced = torch.cat([B, grid_params], dim=1)
            power = self.fc_B_enhanced(B_enhanced)
        else:
            # 回退到原始方法
            power = self.fc_B(B)
        
        # 计算置信度
        if grid_params is not None:
            alpha = self.confidence_fc_enhanced(A_enhanced)
        else:
            alpha = self.confidence_fc(A)
        
        # 自适应位置估计: X_final = α ⊙ X_CNN + (1 - α) ⊙ X_WCL
        pos_pred_final = alpha * pos_pred_cnn + (1 - alpha) * wcl_pos
        return pos_pred_final, pos_pred_cnn, power, alpha

# 训练模型函数
def train_model():
    # 参数设置
    h5_path = "training2_data10w_4500.h5"
    num_samples = 100000
    batch_size = 64
    
    # 生成训练数据
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
    pw_criterion = nn.SmoothL1Loss()    # 平滑L1损失函数,分段使用均方误差和平均绝对误差
    pos_criterion = nn.SmoothL1Loss()    # 平滑L1损失函数,分段使用均方误差和平均绝对误差

    # 训练阶段1：降低Loss1（位置预测）
    print("=== 阶段1:训练位置预测(Loss1)===")
    best_val_loss = float('inf')
    for epoch in range(50):
        # 训练阶段
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels, wcl_pos, grid_params in train_loader:
            M2 = M2.to(device)
            pos_labels = pos_labels.to(device)
            power_labels = power_labels.to(device)
            wcl_pos = wcl_pos.to(device)
            grid_params = grid_params.to(device)
            
            optimizer.zero_grad()
            # 使用L_CAGE损失函数：0.7*L_GNN + 0.3*L_adapt
            pos_pred_final, pos_pred_cnn, power_pred, _ = model(M2, wcl_pos, grid_params)
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
            for M2, pos_labels, power_labels, wcl_pos, grid_params in val_loader:
                M2 = M2.to(device)
                pos_labels = pos_labels.to(device)
                wcl_pos = wcl_pos.to(device)
                grid_params = grid_params.to(device)
        
                # 使用L_CAGE损失函数：0.7*L_GNN + 0.3*L_adapt
                pos_pred_final, pos_pred_cnn, power_pred, _ = model(M2, wcl_pos, grid_params)
                # 第一部分：CNN预测的损失
                loss1 = pos_criterion(pos_pred_cnn, pos_labels)
                # 第二部分：融合位置的损失
                loss_adapt = pos_criterion(pos_pred_final, pos_labels)
                loss = 0.7 * loss1 + 0.3 * loss_adapt
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), "best_cnn41model_stage1.pth")
    
    # 训练阶段2：降低Loss2（功率预测）
    print("\n=== 阶段2:训练功率预测(Loss2)===")
    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels, wcl_pos, grid_params in train_loader:
            M2 = M2.to(device)
            power_labels = power_labels.to(device)
            wcl_pos = wcl_pos.to(device)
            grid_params = grid_params.to(device)

            optimizer.zero_grad()
            _,_, power_pred, _ = model(M2, wcl_pos, grid_params)  # 只关注功率预测
            power_labels = power_labels.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
            power_pred = power_pred.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
            loss = pw_criterion(power_pred, power_labels)  # Loss2
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, power_labels, wcl_pos, grid_params in val_loader:
                M2 = M2.to(device)
                power_labels = power_labels.to(device)
                wcl_pos = wcl_pos.to(device)
                grid_params = grid_params.to(device)

                _,_, power_pred, _ = model(M2, wcl_pos, grid_params)
                power_labels = power_labels.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
                power_pred = power_pred.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
                loss = pw_criterion(power_pred, power_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), "best_cnn41model_stage2.pth")
    
    # 训练阶段3：降低Loss3（综合损失和置信度学习）
    print("\n=== 阶段3:训练综合性能和置信度学习===")
    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels, wcl_pos, grid_params in train_loader:
            M2 = M2.to(device)
            pos_labels = pos_labels.to(device)
            power_labels = power_labels.to(device)
            wcl_pos = wcl_pos.to(device)
            grid_params = grid_params.to(device)
            
            optimizer.zero_grad()
            # 使用L_CAGE损失函数：0.7*L_GNN + 0.3*L_adapt
            pos_pred_final, pos_pred_cnn, power_pred, _ = model(M2, wcl_pos, grid_params)
            # 第一部分：CNN预测的损失
            loss1 = pos_criterion(pos_pred_cnn, pos_labels)
            # 第二部分：融合位置的损失
            loss_adapt = pos_criterion(pos_pred_final, pos_labels)
            # 第三部分：功率预测损失
            power_pred = power_pred.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
            power_labels = power_labels.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
            loss2 = pw_criterion(power_pred, power_labels)
            
            # 综合损失
            loss = 0.7 * loss1 + 0.3 * loss_adapt + 0.1 * loss2
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, power_labels, wcl_pos, grid_params in val_loader:
                M2 = M2.to(device)
                pos_labels = pos_labels.to(device)
                power_labels = power_labels.to(device)
                wcl_pos = wcl_pos.to(device)
                grid_params = grid_params.to(device)
                
                # 使用L_CAGE损失函数：0.7*L_GNN + 0.3*L_adapt
                pos_pred_final, pos_pred_cnn, power_pred, _ = model(M2, wcl_pos, grid_params)
                # 第一部分：CNN预测的损失
                loss1 = pos_criterion(pos_pred_cnn, pos_labels)
                # 第二部分：融合位置的损失
                loss_adapt = pos_criterion(pos_pred_final, pos_labels)
                # 第三部分：功率预测损失
                power_pred = power_pred.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
                power_labels = power_labels.squeeze()  # 从[batch_size, 1, 1, 1]变为[batch_size]
                loss2 = pw_criterion(power_pred, power_labels)
                
                # 综合损失
                loss = 0.7 * loss1 + 0.3 * loss_adapt + 0.1 * loss2
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_cnn2model4311_stage3.pth")
    
    print("\n训练完成")

class NeuralLocalizer:
    """神经网络定位器"""
    def __init__(self):
        # 初始化模型并加载训练好的权重
        self.model = InterferenceModel().to(device)
        # 尝试加载训练好的模型权重
        model_path = "best_cnn2model4311_stage3.pth"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"成功加载模型权重: {model_path}")
            except Exception as e:
                print(f"加载模型权重失败: {e}")
        else:
            print(f"未找到模型权重文件: {model_path}")
        
        # 设置为评估模式
        self.model.eval()
        
        # 初始化网格变换参数列表
        self.grid_transforms = []

    def predict(self, receivers):
        """预测干扰源位置"""
        # 初始化干扰源位置预测
        pred_jam_pos_final = np.zeros(2)
        pred_jam_pos_cnn = np.zeros(2)
        pred_jam_pos_wcl = np.zeros(2)
        pred_jam_pw = 0.0
        alpha = np.zeros(2)
        
        # 找出受干扰的接收机
        interfered = receivers[receivers['ri_pw'] >= -80]  # 过滤出受干扰的接收机
        
        # 如果没有受干扰的接收机，返回默认值
        if len(interfered) == 0:
            return pred_jam_pos_final, pred_jam_pos_cnn, pred_jam_pos_wcl, alpha, pred_jam_pw
        
        # 计算加权质心（权重为干扰功率）
        weights = np.power(10, interfered['ri_pw'] / 10.0)  # 功率转换成线性值
        total_weight = np.sum(weights)
        weighted_x = np.sum(interfered['x'] * weights) / total_weight
        weighted_y = np.sum(interfered['y'] * weights) / total_weight

        # 1. 创建位置矩阵 M2
        # 使用固定网格单元大小，自适应区域选择
        FIXED_GRID_CELL_SIZE = 5  # 米，根据区域大小自动调整网格单元大小
        L = FIXED_GRID_CELL_SIZE 
        # 计算受干扰区域范围
        X_min = np.min(interfered['x'])
        X_max = np.max(interfered['x'])
        Y_min = np.min(interfered['y'])
        Y_max = np.max(interfered['y'])
        
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
        X_range = X_max - X_min
        Y_range = Y_max - Y_min
        X_center = (X_min + X_max) / 2  # 区域中心x坐标
        Y_center = (Y_min + Y_max) / 2  # 区域中心y坐标
        
        # 计算区域对角线长度作为动态归一化因子
        area_diagonal = calculate_area_diagonal(X_min, X_max, Y_min, Y_max)
        # 确保对角线长度不为0
        area_diagonal = max(1.0, area_diagonal)
        
        # 使用相对归一化处理WCL位置
        wcl_pos = [(weighted_x - X_center)/area_diagonal, (weighted_y - Y_center)/area_diagonal]
            
        # 计算需要的网格数量，确保覆盖整个受干扰区域
        num_grid_x = max(1, int(X_range / FIXED_GRID_CELL_SIZE) + 20)  # 增加边界
        num_grid_y = max(1, int(Y_range / FIXED_GRID_CELL_SIZE) + 20)
        
        # 初始化干扰功率矩阵
        M2 = np.zeros((100, 100), dtype=np.float32)
        
        if num_grid_x > 100 or num_grid_y > 100:
            # 计算降采样因子
            scale_factor = max(num_grid_x/100, num_grid_y/100)
            adjusted_cell_size = FIXED_GRID_CELL_SIZE * scale_factor
            L = adjusted_cell_size
            print(f"Adjusted cell size: {adjusted_cell_size}")

            # 填充M2矩阵（受干扰接收机位置和功率）
            for i, receiver in enumerate(interfered):
                x, y = receiver['x'], receiver['y']
                jam_pw = receiver['ri_pw']
                
                # 计算降采样后的网格索引 (0-99)
                i_idx = min(int((y - Y_min)/adjusted_cell_size), 99)
                j_idx = min(int((x - X_min)/adjusted_cell_size), 99)
                
                M2[i_idx, j_idx] += jam_pw
        
        else:
            for i, receiver in enumerate(interfered):
                x, y = receiver['x'], receiver['y']
                jam_pw = receiver['ri_pw']
                
                # 计算网格索引，考虑padding
                i_idx = min(int((y - Y_min)/FIXED_GRID_CELL_SIZE), 99)
                j_idx = min(int((x - X_min)/FIXED_GRID_CELL_SIZE), 99)
                
                M2[i_idx, j_idx] += jam_pw
        
        # 计算WCL位置转为Tensor
        wcl_pos_tensor = torch.tensor(wcl_pos, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 创建网格变换参数张量并归一化，添加区域对角线长度
        grid_params = torch.tensor([X_center/area_diagonal, Y_center/area_diagonal, 
                                  L/area_diagonal, area_diagonal], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 转换为PyTorch张量并添加通道维度
        M2_tensor = torch.tensor(M2).unsqueeze(0).unsqueeze(0).float().to(device)   # 形状: (1, 1, 100, 100)
        
        # 模型预测
        with torch.no_grad():
            pos_pred_final, pos_pred_cnn, power_pred, alpha = self.model(M2_tensor, wcl_pos_tensor, grid_params)
            
            # 转换为NumPy数组
            pos_pred_cnn = pos_pred_cnn.cpu().numpy()[0]  # 形状: (2,)
            pos_pred_final = pos_pred_final.cpu().numpy()[0]  # 形状: (2,)
            power_pred = power_pred.cpu().numpy()[0]  # 形状: (1,)
            alpha = alpha.cpu().numpy()[0]  # 形状: (2,)
        
        # 反归一化位置坐标（相对归一化 -> 实际坐标）
        # 对于相对归一化，我们需要：预测值 * 区域对角线长度 + 区域中心坐标
        pos_pred_actual_cnn = pos_pred_cnn * area_diagonal + np.array([X_center, Y_center])
        pos_pred_actual_final = pos_pred_final * area_diagonal + np.array([X_center, Y_center])
        # 将wcl_pos转换为NumPy数组再进行运算
        wcl_pos_actual = np.array(wcl_pos) * area_diagonal + np.array([X_center, Y_center])
        power_pred = power_pred * 100.0
        
        # 确保坐标在有效范围内
        pos_pred_actual_cnn = np.clip(pos_pred_actual_cnn, 0, area_size)
        pos_pred_actual_final = np.clip(pos_pred_actual_final, 0, area_size)
        wcl_pos_actual = np.clip(wcl_pos_actual, 0, area_size)
        
        # 返回融合后的位置、CNN预测位置、WCL位置和置信度
        return pos_pred_actual_final, pos_pred_actual_cnn, wcl_pos_actual, alpha, power_pred
    
def generate_training_data(h5_path, num_samples):
    """预生成训练数据"""
    if os.path.exists(h5_path):
        os.remove(h5_path)  # 强制重新生成数据
    
    with h5py.File(h5_path, 'w') as f:
        with tqdm(total=num_samples, desc="生成训练数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, jammers, jammers_pw = geshiann.generation_data(
                    x_min=0, x_max=area_size, y_min=0, y_max=area_size,
                    num_receivers=200, num_interferers=1
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('jammers', data=jammers)
                grp.create_dataset('jammers_pw', data=jammers_pw)
                pbar.update(1)
                pbar.set_postfix(last_sample=f"接收机:{len(receivers)} 干扰源:{len(jammers)}")

def calculate_paired_errors(true_sources, estimated_points):
    """增强版误差计算"""
    # 确保输入是NumPy数组
    true_sources = np.array(true_sources)  # 转换为 NumPy 数组
    estimated_points = np.array(estimated_points[0])  # 转换为 NumPy 数组

    # 如果true_sources是单个点（一维数组），转换为二维数组
    if len(true_sources.shape) == 1:
        true_sources = true_sources.reshape(1, -1)
    
    # 计算欧几里得距离
    if len(true_sources) > 0 and len(estimated_points) > 0:
        dis = np.sqrt((true_sources[0][0]-estimated_points[0])**2 + (true_sources[0][1]-estimated_points[1])**2)
    else:
        dis = 0.0  # 默认值，当没有有效数据时
    
    return dis

def generate_data(path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_interferers):
    """生成测试数据"""
    if os.path.exists(path):
        os.remove(path)  # 强制重新生成数据
    
    with h5py.File(path, 'w') as f:
        with tqdm(total=num_samples, desc="生成测试数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, jam_pos, jam_pw = geshiann.generation_data(
                    x_min, x_max, y_min, y_max,
                    num_receivers, num_interferers
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('jammers', data=jam_pos)
                grp.create_dataset('jammers_pw', data=jam_pw)
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
    train_model()
    
    # 初始化定位器
    locator = NeuralLocalizer()
    
    # 仿真参数
    num_simulations = 10
    x_min, x_max = 0, area_size
    y_min, y_max = 0, area_size
    num_interferers = 1
    num_receivers = 200

    # 统计误差
    all_errors = []
    all_errors_cnn = []
    all_errors_wcl = []
    all_errors_pw = []
    error2 = []
    error2_cnn = []
    error2_wcl = []
    error2_pw = []
    all_alphas = []

    with tqdm(range(num_simulations), desc="运行仿真", unit="sim") as sim_pbar:
        for sim in sim_pbar:
            # 生成测试数据
            test_path = "test_data-cnn.h5"
            num_samples = 1     
            generate_data(test_path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_interferers)

            #数据加载与处理
            with h5py.File(test_path, 'r') as f:
                # 遍历所有样本组
                for key in tqdm(f.keys(), desc="加载测试数据"):   #进度条显示加载过程
                    if not key.startswith('sample_'):           #只处理以 sample_ 开头的组（每组代表一个场景）
                        continue
                         
                    group = f[key]
                    receivers = group['receivers'][:]
                    jammers = group['jammers'][:]                    
                    jammers_pw = group['jammers_pw'][:]
                    # 转换字节字符串为普通字符串
                    receivers = receivers.astype([
                        ('x', 'f4'), ('y', 'f4') 
                        ,('ri_pw', 'f4')
                    ])
                    
            # 使用神经网络预测 - 获取融合位置、CNN预测位置、WCL位置和置信度

            pred_jam_pos_final, pred_jam_pos_cnn, pred_jam_pos_wcl, alpha, pred_jam_pw = locator.predict(receivers)
            
            # 保存置信度
            all_alphas.append(alpha)
            
            # 计算误差
            # 融合位置误差
            errors = calculate_paired_errors(jammers, [pred_jam_pos_final])
            # CNN预测误差
            errors_cnn = calculate_paired_errors(jammers, [pred_jam_pos_cnn])
            # WCL位置误差
            errors_wcl = calculate_paired_errors(jammers, [pred_jam_pos_wcl])
            # 功率预测误差
            errors_pw = calculate_paired_errors_pw(jammers_pw, [pred_jam_pw])[0]
            
            all_errors.append(errors)
            all_errors_cnn.append(errors_cnn)
            all_errors_wcl.append(errors_wcl)
            all_errors_pw.append(errors_pw)
            
            # 计算平方误差
            squared_errors = errors**2
            squared_errors_cnn = errors_cnn**2
            squared_errors_wcl = errors_wcl**2
            squared_errors_pw = errors_pw**2
            
            error2.append(squared_errors)
            error2_cnn.append(squared_errors_cnn)
            error2_wcl.append(squared_errors_wcl)
            error2_pw.append(squared_errors_pw)
            
            # 打印当前仿真结果
            print(f"\n仿真 {sim+1}/{num_simulations}:")
            print(f"  融合位置误差: {errors:.2f}m")
            print(f"  CNN预测误差: {errors_cnn:.2f}m")
            print(f"  WCL位置误差: {errors_wcl:.2f}m")
            print(f"  置信度α(x,y): ({alpha[0]:.2f}, {alpha[1]:.2f})")
            
            # 可视化
            plt.figure(figsize=(12, 8))
            
            # 绘制接收机
            normal_mask = receivers['ri_pw'] < -80
            interfered_mask = receivers['ri_pw'] >= -80
            
            plt.scatter(receivers[normal_mask]['x'], receivers[normal_mask]['y'],
                        c='lightgray', s=10, alpha=0.5, label='正常接收机')
            plt.scatter(receivers[interfered_mask]['x'], receivers[interfered_mask]['y'],
                        c='red', s=20, alpha=0.7, label='受干扰接收机')
            
            # 绘制真实干扰源
            plt.scatter(jammers[0], jammers[1], marker='*',
                        s=250, c='blue', edgecolors='black',
                        linewidths=1, label='真实干扰源')
            
            # 绘制融合预测位置
            plt.scatter(pred_jam_pos_final[0], pred_jam_pos_final[1], marker='X',
                        s=200, c='lime', edgecolors='black',
                        linewidths=1, label=f'融合预测 (误差: {errors:.1f}m)')
            
            # 绘制CNN预测位置
            plt.scatter(pred_jam_pos_cnn[0], pred_jam_pos_cnn[1], marker='o',
                        s=150, c='orange', edgecolors='black',
                        linewidths=1, label=f'CNN预测 (误差: {errors_cnn:.1f}m)')
            
            # 绘制WCL位置
            plt.scatter(pred_jam_pos_wcl[0], pred_jam_pos_wcl[1], marker='s',
                        s=150, c='purple', edgecolors='black',
                        linewidths=1, label=f'WCL位置 (误差: {errors_wcl:.1f}m)')
            
            # 添加置信度文本
            plt.text(x_max - 200, y_max - 50, 
                        f'置信度α(x,y): ({alpha[0]:.2f}, {alpha[1]:.2f})',
                        fontsize=20, bbox=dict(facecolor='white', alpha=0.8))
            
            # 添加误差线
            # plt.plot([jammers[0], pred_jam_pos_final[0]], 
            #             [jammers[1], pred_jam_pos_final[1]],
            #             'g--', alpha=0.7, linewidth=2)
            # plt.plot([jammers[0], pred_jam_pos_cnn[0]], 
            #             [jammers[1], pred_jam_pos_cnn[1]],
            #             'orange--', alpha=0.7, linewidth=2)
            # plt.plot([jammers[0], pred_jam_pos_wcl[0]], 
            #             [jammers[1], pred_jam_pos_wcl[1]],
            #             'purple--', alpha=0.7, linewidth=2)
            
            # 添加图例和标注
            plt.title(f'干扰源定位仿真 (第 {sim+1} 次)', fontsize=24)
            plt.xlabel('X 坐标 (米)', fontsize=20)
            plt.ylabel('Y 坐标 (米)', fontsize=20)
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.legend(loc='upper right', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.xlim(x_min-50, x_max+50)
            plt.ylim(y_min-50, y_max+50)
            plt.show()
            
            # 保存图形
            # plt.savefig(f'interference_localization_{sim+1}.png', dpi=150)
            # plt.close()

            
    # 统计误差
    if len(all_errors) > 0:
        max_error = np.max(all_errors)
        min_error = np.min(all_errors)
        mean_error = np.mean(all_errors)
    else:
        max_error = min_error = mean_error = 0

    if len(all_errors_cnn) > 0:
        max_error_cnn = np.max(all_errors_cnn)
        min_error_cnn = np.min(all_errors_cnn)
        mean_error_cnn = np.mean(all_errors_cnn)
    else:
        max_error_cnn = min_error_cnn = mean_error_cnn = 0

    if len(all_errors_wcl) > 0:
        max_error_wcl = np.max(all_errors_wcl)
        min_error_wcl = np.min(all_errors_wcl)
        mean_error_wcl = np.mean(all_errors_wcl)
    else:
        max_error_wcl = min_error_wcl = mean_error_wcl = 0

    if len(all_errors_pw) > 0:
        max_error_pw = np.max(all_errors_pw)
        min_error_pw = np.min(all_errors_pw)
        mean_error_pw = np.mean(all_errors_pw)
    else:
        max_error_pw = min_error_pw = mean_error_pw = 0

    # 计算RMSE
    if len(error2) > 0:
        error2 = np.array(error2)
        mean_error2 = np.mean(error2)
        RMSE_error = np.sqrt(mean_error2)
    else:
        RMSE_error = 0

    if len(error2_cnn) > 0:
        error2_cnn = np.array(error2_cnn)
        mean_error2_cnn = np.mean(error2_cnn)
        RMSE_error_cnn = np.sqrt(mean_error2_cnn)
    else:
        RMSE_error_cnn = 0

    if len(error2_wcl) > 0:
        error2_wcl = np.array(error2_wcl)
        mean_error2_wcl = np.mean(error2_wcl)
        RMSE_error_wcl = np.sqrt(mean_error2_wcl)
    else:
        RMSE_error_wcl = 0

    if len(error2_pw) > 0:
        error2_pw = np.array(error2_pw)
        mean_error2_pw = np.mean(error2_pw)
        RMSE_error_pw = np.sqrt(mean_error2_pw)
    else:
        RMSE_error_pw = 0

    # 计算置信度统计
    if len(all_alphas) > 0:
        all_alphas = np.array(all_alphas)
        mean_alpha_x = np.mean(all_alphas[:, 0])
        mean_alpha_y = np.mean(all_alphas[:, 1])
        std_alpha_x = np.std(all_alphas[:, 0])
        std_alpha_y = np.std(all_alphas[:, 1])
    else:
        mean_alpha_x = mean_alpha_y = std_alpha_x = std_alpha_y = 0

    # 输出统计结果
    print("\n仿真结果统计:")
    print(f" 融合位置最小误差为 {min_error:.2f}m")
    print(f" 融合位置最大误差为 {max_error:.2f}m")
    print(f" 融合位置平均误差为 {mean_error:.2f}m, RMSE为 {RMSE_error:.2f}m")
    print()
    print(f" CNN预测最小误差为 {min_error_cnn:.2f}m")
    print(f" CNN预测最大误差为 {max_error_cnn:.2f}m")
    print(f" CNN预测平均误差为 {mean_error_cnn:.2f}m, RMSE为 {RMSE_error_cnn:.2f}m")
    print()
    print(f" WCL位置最小误差为 {min_error_wcl:.2f}m")
    print(f" WCL位置最大误差为 {max_error_wcl:.2f}m")
    print(f" WCL位置平均误差为 {mean_error_wcl:.2f}m, RMSE为 {RMSE_error_wcl:.2f}m")
    print()
    print(f" 功率预测最小误差为 {min_error_pw:.2f}dBm")
    print(f" 功率预测最大误差为 {max_error_pw:.2f}dBm")
    print(f" 功率预测平均误差为 {mean_error_pw:.2f}dBm, RMSE为 {RMSE_error_pw:.2f}dBm")
    print()
    print(f" 置信度α(x)均值: {mean_alpha_x:.2f}, 标准差: {std_alpha_x:.2f}")
    print(f" 置信度α(y)均值: {mean_alpha_y:.2f}, 标准差: {std_alpha_y:.2f}")
    
    # 绘制误差分布直方图
    plt.figure(figsize=(12, 6))
    
    # 确定合适的bins范围
    all_pos_errors = all_errors + all_errors_cnn + all_errors_wcl
    max_pos_error = np.max(all_pos_errors)
    bins = np.arange(0, max_pos_error + 10, 10)  # 区间宽度固定10米
    
    # 绘制融合位置误差直方图
    plt.hist(all_errors, bins=bins, alpha=0.5, color='green',
                label=f'融合预测 (均值: {mean_error:.1f}m, RMSE: {RMSE_error:.1f}m)', 
                density=True)
    
    # 绘制CNN预测误差直方图
    plt.hist(all_errors_cnn, bins=bins, alpha=0.5, color='red',
                label=f'CNN预测 (均值: {mean_error_cnn:.1f}m, RMSE: {RMSE_error_cnn:.1f}m)', 
                density=True)
    
    # 绘制WCL位置误差直方图
    plt.hist(all_errors_wcl, bins=bins, alpha=0.5, color='blue',
                label=f'WCL位置 (均值: {mean_error_wcl:.1f}m, RMSE: {RMSE_error_wcl:.1f}m)', 
                density=True)
    
    plt.xticks(bins)
    plt.xlabel('定位误差 (米)', fontsize=14)
    plt.ylabel('概率密度', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title('不同定位方法的误差分布对比', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制功率误差直方图
    plt.figure(figsize=(8, 6))

    all_errors_pw = np.array(all_errors_pw)
    bins_pw = np.arange(0, max_error_pw + 0.5, 0.5)  # 区间宽度固定0.5dbm
    plt.hist(all_errors_pw, bins=bins_pw, alpha=0.6, color='blue',
            label=f'功率预测 (均值: {mean_error_pw:.1f}dBm, RMSE: {RMSE_error_pw:.1f}dBm)', 
            density=True)
    plt.xticks(bins_pw)
    plt.xlabel('功率误差 (dBm)', fontsize=14)
    plt.ylabel('概率密度', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title('功率预测误差分布', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    
    # 绘制x方向置信度直方图
    plt.subplot(1, 2, 1)
    plt.hist(all_alphas[:, 0], bins=20, alpha=0.6, color='blue',
            label=f'α(x) (均值: {mean_alpha_x:.2f}, 标准差: {std_alpha_x:.2f})', 
            density=True)
    plt.xlabel('置信度α(x)', fontsize=14)
    plt.ylabel('概率密度', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title('x方向置信度分布', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    
    # 绘制y方向置信度直方图
    plt.subplot(1, 2, 2)
    plt.hist(all_alphas[:, 1], bins=20, alpha=0.6, color='green',
            label=f'α(y) (均值: {mean_alpha_y:.2f}, 标准差: {std_alpha_y:.2f})', 
            density=True)
    plt.xlabel('置信度α(y)', fontsize=14)
    plt.ylabel('概率密度', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.title('y方向置信度分布', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    
    plt.tight_layout()

    plt.show()
   

   

