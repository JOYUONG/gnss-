# CNN+分支+BN处理+FC模型、1个干扰源
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import geshi_ann as geshiann

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

# 自定义数据集类
class InterferenceDataset(Dataset):
    """自定义数据集类 - 修改为每个干扰源一个样本"""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.samples = []  # 存储(M1, M2)元组
        self.targets = []  # 存储目标值(位置和功率)
        
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
                
                # 1. 创建位置矩阵 M1 和 M2
                M2 = np.zeros((100, 100), dtype=np.float32)  # 干扰功率矩阵
                # 填充M2矩阵（受干扰接收机位置和功率）
                for i, receiver in enumerate(affected_receivers):
                        x, y = receiver['x'], receiver['y']
                        # 计算网格索引 (0-99)
                        i_idx = min(int(y / 10), 99)
                        j_idx = min(int(x / 10), 99)
                        # 累加干扰功率
                        # 确保jam_pw[i]是标量值
                        jam_pw = receiver['ri_pw']
                        M2[i_idx, j_idx] += jam_pw

                # 存储转换后的矩阵
                self.samples.append((M2))
                
                # 存储目标值（干扰源位置和功率）
                # 这里假设每个场景只有一个干扰源，取第一个干扰源的位置和功率
                if len(jammers) > 0:
                    jammer = jammers[0]
                    # 位置归一化到[0,1]范围
                    position = np.array([jammer[0]/1000.0, jammer[1]/1000.0], dtype=np.float32)
                    # 功率归一化（假设最大功率为1000）
                    power = np.array([jammers_pw/100.0], dtype=np.float32) 
                    self.targets.append((position, power))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        M2 = self.samples[idx]
        position, power = self.targets[idx]
        
        # 转换为PyTorch张量并添加通道维度
        M2_tensor = torch.tensor(M2).unsqueeze(0)  # 形状: (1, 100, 100)
        
        # 目标值(干扰源位置、功率)
        position_tensor = torch.tensor(position)
        power_tensor = torch.tensor(power)
        
        return M2_tensor, position_tensor, power_tensor

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
        
        # DNN全连接网络 - A路（位置预测）
        self.fc_A = nn.Sequential(
            nn.Linear(self.flat_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 2),  # 位置(x,y)
            nn.Sigmoid()       # 归一化到[0,1]范围
        )
        
        # DNN全连接网络 - B路（功率预测）
        self.fc_B = nn.Sequential(
            nn.Linear(self.flat_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),  # 只需1个输出值 (功率)
            # 移除Sigmoid，功率值可能大于1
        )

    def forward(self, M2):
        # 通过CNN降维
        MC2 = self.cnn_reduction(M2)
        
        # 通过BN层
        MN2 = self.bn(MC2)
        
        # 分支A处理,只对A路使用BN处理
        A = MN2 * self.K2 + self.B1
        A = A.view(-1, self.flat_features)  # 展平
        pos = self.fc_A(A)
        
        # 分支B处理
        B = MC2 * self.K4 + self.B2
        B = B.view(-1, self.flat_features)  # 展平
        power = self.fc_B(B)
        
        return pos, power

# 训练模型函数
def train_model():
    # 参数设置
    h5_path = "training_data.h5"
    num_samples = 100000
    batch_size = 64
    
    # 生成训练数据
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
    
    # 训练阶段1：降低Loss1（位置预测）
    print("=== 阶段1:训练位置预测(Loss1)===")
    best_val_loss = float('inf')
    for epoch in range(30):
        # 训练阶段
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels in train_loader:
            M2 = M2.to(device)
            pos_labels = pos_labels.to(device)
            power_labels = power_labels.to(device)
            
            optimizer.zero_grad()
            pos_pred, _ = model(M2)  # 只关注位置预测
            loss = nn.MSELoss()(pos_pred, pos_labels)  # Loss1
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, power_labels in val_loader:
                M2 = M2.to(device)
                pos_labels = pos_labels.to(device)
                
                pos_pred, _ = model(M2)
                loss = nn.MSELoss()(pos_pred, pos_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_stage1.pth")
    
    # 训练阶段2：降低Loss2（功率预测）
    print("\n=== 阶段2:训练功率预测(Loss2)===")
    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels in train_loader:
            M2 = M2.to(device)
            power_labels = power_labels.to(device)
            
            optimizer.zero_grad()
            _, power_pred = model(M2)  # 只关注功率预测
            loss = torch.mean(torch.abs(power_pred - power_labels))  # Loss2
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, power_labels in val_loader:
                M2 = M2.to(device)
                power_labels = power_labels.to(device)
                
                _, power_pred = model(M2)
                loss = torch.mean(torch.abs(power_pred - power_labels))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_stage2.pth")
    
    # 训练阶段3：降低Loss3（综合损失）
    print("\n=== 阶段3:训练综合性能(Loss3)===")
    best_val_loss = float('inf')
    for epoch in range(20):
        model.train()
        train_loss = 0
        for M2, pos_labels, power_labels in train_loader:
            M2 = M2.to(device)
            pos_labels = pos_labels.to(device)
            power_labels = power_labels.to(device)
            
            optimizer.zero_grad()
            pos_pred, power_pred = model(M2)
            
            # 计算综合损失
            loss1 = nn.MSELoss()(pos_pred, pos_labels)
            loss2 = torch.mean(torch.abs(power_pred - power_labels))
            loss = loss1 * loss2  # Loss3
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for M2, pos_labels, power_labels in val_loader:
                M2 = M2.to(device)
                pos_labels = pos_labels.to(device)
                power_labels = power_labels.to(device)
                
                pos_pred, power_pred = model(M2)
                loss1 = nn.MSELoss()(pos_pred, pos_labels)
                loss2 = torch.mean(torch.abs(power_pred - power_labels))
                loss = loss1 * loss2
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_stage3.pth")
    
    print("\n训练完成")

class NeuralLocalizer:
    """神经网络定位器"""
    def __init__(self, model_path="best_model_stage3.pth"):
        self.model = InterferenceModel().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, receivers):
        """执行预测 - 只考虑单个干扰源场景"""
        # 找出所有受干扰的接收机
        #mask = (receivers['ri_pw'] != -np.inf) & (receivers['ri_pw'] >= -80)
        mask = receivers['ri_pw'] >= -80
        interfered = receivers[mask]
        #interfered = receivers
        # 如果没有受干扰的接收机，返回空数组
        if len(interfered) == 0:
            return np.empty((0, 2))
        
        # 1. 创建位置矩阵 M1 和 M2
        M2 = np.zeros((100, 100), dtype=np.float32)  # 干扰功率矩阵
        # 填充M2矩阵（受干扰接收机位置和功率）
        for i, receiver in enumerate(interfered):
                x, y = receiver['x'], receiver['y']
                # 计算网格索引 (0-99)
                i_idx = min(int(y / 10), 99)
                j_idx = min(int(x / 10), 99)
                jam_pw = receiver['ri_pw']

                M2[i_idx, j_idx] += jam_pw
                #M2 = np.log10(M2)    
        # 转换为PyTorch张量并添加通道维度
        M2_tensor = torch.tensor(M2).unsqueeze(0).unsqueeze(0).float()   # 形状: (1, 100, 100)
        # 模型预测  
        with torch.no_grad():
            pos_pred, power_pred = self.model(M2_tensor)
            # 转换为NumPy数组
            pos_pred = pos_pred.numpy()[0]  # 形状: (2,)
            power_pred = power_pred.numpy()[0]  # 形状: (1,)
        
        # 4. 反归一化位置坐标 (0-1范围 -> 0-1000实际坐标)
        pos_pred_actual = pos_pred * 1000.0
        power_pred = power_pred * 100.0
        return pos_pred_actual, power_pred
    
def generate_training_data(h5_path, num_samples):
    """预生成训练数据"""
    if os.path.exists(h5_path):
        os.remove(h5_path)  # 强制重新生成数据
    
    with h5py.File(h5_path, 'w') as f:
        with tqdm(total=num_samples, desc="生成训练数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, jammers, jammers_pw = geshiann.generation_data(
                    x_min=0, x_max=1000, y_min=0, y_max=1000,
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
    #train_model()
    
    # 初始化定位器
    locator = NeuralLocalizer()
    
    # 仿真参数
    num_simulations = 10
    x_min, x_max = 0, 1000
    y_min, y_max = 0, 1000
    num_interferers = 1
    num_receivers = 200

    # 统计误差
    all_errors = []
    all_errors_pw = []
    error2 = []
    error2_pw = []

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
                    
            predicted_postions  = []
            predicted_power = []
            # 使用神经网络预测
            pred_jam_pos, pred_jam_pw = locator.predict(receivers)
            predicted_postions.append(pred_jam_pos)
            predicted_power.append(pred_jam_pw)
            # 计算误差
            #current_predicted = np.array(predicted_postions)
            #print(spoofers)
            #print(predicted_postions)

            errors = calculate_paired_errors(jammers, predicted_postions)

            errors_pw = calculate_paired_errors_pw(jammers_pw, predicted_power)
          
            #print(errors)
            #print(cluster_errors)
            all_errors.extend(errors['distances'])
            '''
            predicted_spoofers = locator.predict(receivers)
            
            # 计算误差
            errors = calculate_paired_errors(spoofers, predicted_spoofers)['distances']
            print(errors)
            '''
            all_errors.extend(errors['distances'])
            all_errors_pw.extend(errors_pw.flatten())
            
            # 计算平方误差并添加到 error2 列表
            squared_errors = [e ** 2 for e in errors['distances']]  # 计算每个误差的平方
            squared_errors_pw = [e ** 2 for e in errors_pw]  # 计算每个误差的平方
            error2.extend(squared_errors)  # 添加到 error2 列表
            error2_pw.extend(squared_errors_pw)
            #print(all_errors)
            
            # 更新进度条
            #if errors:
            #    sim_pbar.set_postfix(avg_error=f"{np.mean(errors):.2f}m")
            
            # 可视化
            plt.figure(figsize=(8, 6))
            
            # 绘制接收机
            normal_mask = receivers['ri_pw'] < -80
            interfered_mask = receivers['ri_pw'] >= -80
            
            plt.scatter(receivers[normal_mask]['x'], receivers[normal_mask]['y'],
                        c='lightgray', s=10, alpha=0.5, label='正常接收机')
            plt.scatter(receivers[interfered_mask]['x'], receivers[interfered_mask]['y'],
                        c='red', s=20, alpha=0.7, label='受干扰接收机')
            
            # 绘制真实干扰源
            plt.scatter(jammers[:, 0], jammers[:, 1], marker='*',
                        s=250, c='blue', edgecolors='black',
                        linewidths=1, label='真实干扰源')
            predicted_postions = np.array(predicted_postions)
            # 绘制预测干扰源
            if len(predicted_postions) > 0:
                plt.scatter(predicted_postions[:, 0], predicted_postions[:, 1], marker='X',
                            s=200, c='lime', edgecolors='black',
                            linewidths=1, label='预测干扰源')
                
                # 添加误差线
                '''
                for i in range(len(spoofers)):
                    plt.plot([spoofers[i, 0], predicted_postions[i, 0]], 
                             [spoofers[i, 1], predicted_postions[i, 1]],
                             'r--', alpha=0.7)
                '''
            # 添加图例和标注
            #plt.title(f'干扰源定位仿真 (第 {sim+1} 次)')
            plt.xlabel('X 坐标 (米)')
            plt.ylabel('Y 坐标 (米)')
            plt.tick_params(axis='x', labelsize=32)
            plt.tick_params(axis='y', labelsize=32)# 单独调整y轴刻度
            #plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.xlim(x_min-50, x_max+50)
            plt.ylim(y_min-50, y_max+50)
            plt.show()

            # 保存图形
            #plt.tight_layout()
            #plt.savefig(f'spoofer_localization_{sim+1}.png', dpi=150)
            #plt.close()
            
    max_error = np.max(all_errors)
    min_error = np.min(all_errors)
    mean_error = np.mean(all_errors)if all_errors else 0

    max_error_pw = np.max(all_errors_pw)
    min_error_pw = np.min(all_errors_pw)
    mean_error_pw = np.mean(all_errors_pw)if all_errors_pw else 0

    error2 = np.array(error2)
    error2_pw = np.array(error2_pw)
    mean_error2= np.mean(error2)
    mean_error2_pw= np.mean(error2_pw)
    RMSE_error = np.sqrt(mean_error2)if mean_error2 else 0
    RMSE_error_pw = np.sqrt(mean_error2_pw)if mean_error2_pw else 0
    # 输出统计结果
    print("\n仿真结果统计:")
    print(f" 位置最小误差为 {min_error:.2f}m")
    print(f" 位置最大误差为 {max_error:.2f}m")
    print(f" 位置平均误差为 {mean_error:.2f}m, RMSE为 {RMSE_error:.2f}m")
    
    print(f" PW最小误差为 {min_error_pw:.2f}dBm")
    print(f" PW最大误差为 {max_error_pw:.2f}dBm")
    print(f" PW平均误差为 {mean_error_pw:.2f}dBm, RMSE为 {RMSE_error_pw:.2f}dBm")
    

    
     
    plt.figure(figsize=(6, 6))
    #bins = np.linspace(0, np.max(cluster_errors), 30)

    bins = np.arange(0, max_error + 10, 10)  # 区间宽度固定10米
    plt.hist(all_errors, bins=bins, alpha=0.6, color='red',
        label=f'({mean_error:.1f}, {RMSE_error:.1f})', 
        density=True)
        #label=f'($\sigma^2$={np.sqrt(cluster_var):.1f} m$^2$)', density=False)
    plt.xticks(bins)  # 显示所有刻度
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    plt.xlabel('error (m)')
    plt.tick_params(axis='x', labelsize=32)
    plt.tick_params(axis='y', labelsize=32)# 单独调整y轴刻度
    #plt.ylabel('probability  density')
    #plt.title('定位误差分布直方图')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  
   

    plt.figure(figsize=(6, 6))

    bins = np.arange(0, max_error_pw + 0.5, 0.5)  # 区间宽度固定0.5dbm
    plt.hist(all_errors_pw, bins=bins, alpha=0.6, color='blue',
        label=f'({mean_error_pw:.1f}, {RMSE_error_pw:.1f})', 
        density=True)
        #label=f'($\sigma^2$={np.sqrt(cluster_var):.1f} m$^2$)', density=False)
    plt.xticks(bins)  # 显示所有刻度
    #plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    plt.xlabel('error (dBm)')
    plt.tick_params(axis='x', labelsize=32)
    plt.tick_params(axis='y', labelsize=32)# 单独调整y轴刻度
    #plt.ylabel('probability  density')
    #plt.title('功率误差分布直方图')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()
   

