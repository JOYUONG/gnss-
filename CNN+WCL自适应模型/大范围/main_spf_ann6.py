import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import geshi_spf_ann3 as geshispfann2

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
    #继承 PyTorch 的 Dataset 类，需实现 __len__ 和 __getitem__ 方法
    def __init__(self, h5_path):
        self.h5_path = h5_path      #HDF5 格式数据文件的路径
        self.samples = []           #用于存储处理后的样本列表
        
        #数据加载与处理
        with h5py.File(h5_path, 'r') as f:
            # 遍历所有样本组
            for key in tqdm(f.keys(), desc="加载数据集"):   #进度条显示加载过程
                if not key.startswith('sample_'):           #只处理以 sample_ 开头的组（每组代表一个场景）
                    continue
                    
                group = f[key]
                receivers = group['receivers'][:]
                spoofers = group['spoofers'][:]
                
                # 转换字节字符串为普通字符串
                receivers = receivers.astype([
                    ('x', 'f4'), ('y', 'f4') 
                    ,('spoofer_count', 'i4')
                    ,('spoofer_types', 'U50')
                    , ('spoofer_indices', 'U50')
                ])
                
                # 为每个干扰源创建样本
                for spoofer in spoofers:
                    sp_x, sp_y = spoofer
                    
                    # 找出受当前干扰源影响的接收机（距离<=150米）
                    distances = np.sqrt((receivers['x'] - sp_x)**2 + (receivers['y'] - sp_y)**2)
                    affected_mask = distances <= 150
                    affected_receivers = receivers[affected_mask]
                    
                    # 跳过没有受影响接收机的干扰源
                    if len(affected_receivers) == 0:
                        continue
                    
                    # 计算特征
                    # 1. 中心坐标（所有受影响接收机的几何中心）
                    #centroid_x = np.mean(affected_receivers['x'])
                    #centroid_y = np.mean(affected_receivers['y'])
                    
                    # 2. 分布方差（接收机位置的离散程度）
                    var_x = np.var(affected_receivers['x'])
                    var_y = np.var(affected_receivers['y'])
                    
                    # 3. 长宽比（接收机分布区域的形状特征）
                    min_x, max_x = np.min(affected_receivers['x']), np.max(affected_receivers['x'])
                    min_y, max_y = np.min(affected_receivers['y']), np.max(affected_receivers['y'])
                    width = max_x - min_x
                    height = max_y - min_y

                    # 1. 中心坐标（接收机分布区域的几何中心）
                    centroid_x = (min_x + max_x) / 2
                    centroid_y = (min_y + max_y) / 2

                    if height > 0:
                        aspect_ratio = width / height 

                        # 4. 接收机数量（干扰源影响到的接收机数量）
                        # 计算宽度和长度的二分之一位置
                        half_width = width / 2
                        half_length = height / 2

                        # 找出接收机分布区域的边界
                        left_boundary = min_x + half_width
                        right_boundary = max_x - half_width
                        bottom_boundary = min_y + half_length
                        top_boundary = max_y - half_length

                        # 计算左下侧接收机点数目
                        left_bottom_count = np.sum((affected_receivers['x'] <= left_boundary) & (affected_receivers['x'] >= min_x)
                                            & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
                        # 计算左上侧接收机点数目
                        left_top_count = np.sum((affected_receivers['x'] <= left_boundary) & (affected_receivers['x'] >= min_x)
                                                & (affected_receivers['y'] >= top_boundary) & (affected_receivers['y'] <= max_y))
                        
                        # 计算右下侧接收机点数目
                        right_bottom_count = np.sum((affected_receivers['x'] <= max_x) & (affected_receivers['x'] >= right_boundary)
                                            & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
                        # 计算右上侧接收机点数目
                        right_top_count = np.sum((affected_receivers['x'] <= max_x) & (affected_receivers['x'] >= right_boundary)
                                                & (affected_receivers['y'] >= top_boundary) & (affected_receivers['y'] <= max_y))

                    else:
                        aspect_ratio = -1.0
                        # 4. 接收机数量（干扰源影响到的接收机数量）
                        # 计算宽度和长度的二分之一位置
                        half_width = width / 2
                        #half_length = height / 2

                        # 找出接收机分布区域的边界
                        left_boundary = min_x + half_width
                        right_boundary = max_x - half_width
                        #bottom_boundary = min_y + half_length
                        #top_boundary = max_y - half_length

                        # 计算左下侧接收机点数目
                        #left_bottom_count = np.sum((affected_receivers['x'] <= left_boundary) & (affected_receivers['x'] >= min_x)
                        #                    & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
                        left_bottom_count = 0
                        # 计算左上侧接收机点数目
                        left_top_count = np.sum((affected_receivers['x'] <= left_boundary) & (affected_receivers['x'] >= min_x))
                        
                        # 计算右下侧接收机点数目
                        #right_bottom_count = np.sum((affected_receivers['x'] <= max_x) & (affected_receivers['x'] >= right_boundary)
                        #                    & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
                        right_bottom_count = 0
                        # 计算右上侧接收机点数目
                        right_top_count = np.sum((affected_receivers['x'] <= max_x) & (affected_receivers['x'] >= right_boundary))
                        
                    interfered_num = len(affected_receivers)
                    lbc_ratio = left_bottom_count / interfered_num if interfered_num > 0 else 1.0 
                    ltc_ratio = left_top_count / interfered_num if interfered_num > 0 else 1.0
                    rbc_ratio = right_bottom_count / interfered_num if interfered_num > 0 else 1.0
                    rtc_ratio = right_top_count / interfered_num if interfered_num > 0 else 1.0
                    # 特征向量(归一化)
                    features = np.array([
                    #    centroid_x / 1000.0,  # 归一化
                    #    centroid_y / 1000.0,  # 归一化
                        var_x / 1000.0,       # 归一化
                        var_y / 1000.0,        # 归一化
                        aspect_ratio,
                        lbc_ratio,
                        ltc_ratio,
                        rbc_ratio,
                        rtc_ratio
                    ])
                    
                    # 标签：中心坐标与干扰源实际位置的差值
                    dx = (sp_x - centroid_x) / 1000.0  # 归一化
                    dy = (sp_y - centroid_y) / 1000.0  # 归一化
                    
                    self.samples.append((features, np.array([dx, dy])))
    
    def __len__(self):              #返回样本总数
        return len(self.samples)
    
    def __getitem__(self, idx):     #返回PyTorch张量格式的特征和标签
        features, label = self.samples[idx]
        return torch.FloatTensor(features), torch.FloatTensor(label)

class SpooferLocalizer(nn.Module):      
    #继承自 PyTorch 的 nn.Module 基类,表示这是一个神经网络模型
    """干扰源定位模型 - 全连接网络"""
    def __init__(self, input_size=7):
        super().__init__()          #调用父类 nn.Module 的初始化方法
        #网络结构,定义了一个全连接（FC）神经网络
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),     #线性层1,input_size为输入维数,128为输出维数
            nn.ReLU(),
            nn.Linear(128, 64),             #线性层2,128为输入维数,64为输出维数
            nn.ReLU(),
            nn.Linear(64, 32),              #线性层3,64为输入维数,32为输出维数
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出dx, dy  #线性层4,32为输入维数，2为输出维数

            #激活函数：每层线性层后使用 ReLU 激活函数（最后一层除外）
            #输出层：最终输出维度为 2，对应定位结果（dx, dy）
            
        )
    
    def forward(self, x):
        #定义数据如何通过网络
        #输入 x 直接通过 self.fc 序列
        #输出为二维向量（dx, dy）
        return self.fc(x)

def generate_training_data(h5_path, num_samples):
    """预生成训练数据"""
    if os.path.exists(h5_path):
        os.remove(h5_path)  # 强制重新生成数据
    
    with h5py.File(h5_path, 'w') as f:
        with tqdm(total=num_samples, desc="生成训练数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, spoofers = geshispfann2.generation_data(
                    x_min=0, x_max=2000, y_min=0, y_max=2000,
                    num_receivers=100, num_spoofers=1
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('spoofers', data=spoofers)
                pbar.update(1)
                pbar.set_postfix(last_sample=f"接收机:{len(receivers)} 干扰源:{len(spoofers)}")

def train_model():
    """模型训练函数"""
    # 参数设置
    h5_path = "training_data.h5"
    num_samples = 1000     # 训练样本数量
    batch_size = 64         # 批次大小
    
    # 生成训练数据
    generate_training_data(h5_path, num_samples)
    
    # 初始化模型
    model = SpooferLocalizer().to(device)    # 初始化模型并移动到指定设备（CPU/GPU）
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #优化器参数：学习率 1e-3，权重衰减 1e-4（L2正则化）
    criterion = nn.MSELoss()     # 使用均方误差损失函数
    
    # 准备数据集
    dataset = InterferenceDataset(h5_path)  # 从H5文件加载数据集
    train_size = int(0.8 * len(dataset))    # 训练集比例80%
    val_size = len(dataset) - train_size    # 验证集比例20%
    train_set, val_set = random_split(dataset, [train_size, val_size])  # 随机划分训练集和验证集
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)    # 训练集随机打乱
    val_loader = DataLoader(val_set, batch_size=batch_size)                      # 验证集无需随机打乱
    
    # 训练循环
    best_val_loss = float('inf')        ## 初始化最佳验证损失为无穷大
    for epoch in range(50):  # 增加训练轮数
        # 训练阶段
        model.train()            # 设置模型为训练模式
        train_loss = 0
        with tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", unit="batch") as train_pbar:
            for features, labels in train_pbar:
                # 将数据加载到设备
                features = features.to(device)
                labels = labels.to(device)
                
                #  前向传播
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                #  反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 验证阶段
        model.eval()       # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():   # 禁用梯度计算
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        # 计算平均损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("保存最佳模型")

class NeuralLocalizer:
    """神经网络定位器"""
    def __init__(self, model_path="best_model.pth"):
        self.model = SpooferLocalizer().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, receivers):
        """执行预测 - 只考虑单个干扰源场景"""
        # 找出所有受干扰的接收机
        #interfered = receivers[receivers['spoofer_count'] > 0]
        interfered = receivers
        # 如果没有受干扰的接收机，返回空数组
        if len(interfered) == 0:
            return np.empty((0, 2))
        
        # 计算特征
        # 1. 中心坐标
        #centroid_x = np.mean(interfered[:,0])
        #centroid_y = np.mean(interfered[:,1])
        
        # 2. 方差
        var_x = np.var(interfered['x'])
        var_y = np.var(interfered['y'])
        
        # 3. 长宽比
        min_x, max_x = np.min(interfered['x']), np.max(interfered['x'])
        min_y, max_y = np.min(interfered['y']), np.max(interfered['y'])
        width = max_x - min_x
        height = max_y - min_y

        
        
        # 1. 中心坐标（接收机分布区域的几何中心）
        centroid_x = (min_x + max_x) / 2
        centroid_y = (min_y + max_y) / 2
        if height > 0 :
            aspect_ratio = width / height 
            # 4. 接收机数量（干扰源影响到的接收机数量）
            # 计算宽度和长度的二分之一位置
            half_width = width / 2
            half_length = height / 2

            # 找出接收机分布区域的边界
            left_boundary = min_x + half_width
            right_boundary = max_x - half_width
            bottom_boundary = min_y + half_length
            top_boundary = max_y - half_length

            # 计算左下侧接收机点数目
            left_bottom_count = np.sum((interfered['x'] <= left_boundary) & (interfered['x'] >= min_x)
                                & (interfered['y'] >= min_y)& (interfered['y'] <= bottom_boundary))
            # 计算左上侧接收机点数目
            left_top_count = np.sum((interfered['x'] <= left_boundary) & (interfered['x'] >= min_x)
                                    & (interfered['y'] >= top_boundary) & (interfered['y'] <= max_y))
            
            # 计算右下侧接收机点数目
            right_bottom_count = np.sum((interfered['x'] <= max_x) & (interfered['x'] >= right_boundary)
                                & (interfered['y'] >= min_y)& (interfered['y'] <= bottom_boundary))
            # 计算右上侧接收机点数目
            right_top_count = np.sum((interfered['x'] <= max_x) & (interfered['x'] >= right_boundary)
                                    & (interfered['y'] >= top_boundary) & (interfered['y'] <= max_y))
            
        else :
            aspect_ratio = -1.0
            half_width = width / 2
            #half_length = height / 2

            # 找出接收机分布区域的边界
            left_boundary = min_x + half_width
            right_boundary = max_x - half_width
            #bottom_boundary = min_y + half_length
            #top_boundary = max_y - half_length

            # 计算左下侧接收机点数目
            #left_bottom_count = np.sum((affected_receivers['x'] <= left_boundary) & (affected_receivers['x'] >= min_x)
            #                    & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
            left_bottom_count = 0
            # 计算左上侧接收机点数目
            left_top_count = np.sum((interfered['x'] <= left_boundary) & (interfered['x'] >= min_x))
            
            # 计算右下侧接收机点数目
            #right_bottom_count = np.sum((affected_receivers['x'] <= max_x) & (affected_receivers['x'] >= right_boundary)
            #                    & (affected_receivers['y'] >= min_y)& (affected_receivers['y'] <= bottom_boundary))
            right_bottom_count = 0
            # 计算右上侧接收机点数目
            right_top_count = np.sum((interfered['x'] <= max_x) & (interfered['x'] >= right_boundary))
        interfered_num = len(interfered)
        lbc_ratio = left_bottom_count / interfered_num if interfered_num > 0 else 1.0 
        ltc_ratio = left_top_count / interfered_num if interfered_num > 0 else 1.0
        rbc_ratio = right_bottom_count / interfered_num if interfered_num > 0 else 1.0
        rtc_ratio = right_top_count / interfered_num if interfered_num > 0 else 1.0
        # 特征向量(归一化)
        features = np.array([
        #    centroid_x / 1000.0,  # 归一化
        #    centroid_y / 1000.0,  # 归一化
            var_x / 1000.0,       # 归一化
            var_y / 1000.0,        # 归一化
            aspect_ratio,
            lbc_ratio,
            ltc_ratio,
            rbc_ratio,
            rtc_ratio
        ])
        
        # 使用模型预测dx, dy
        with torch.no_grad():
            #将特征转为PyTorch张量
            tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            #模型预测并转回NumPy数组
            outputs = self.model(tensor).cpu().numpy()[0]
        
        # 反归一化并计算干扰源位置
        dx = outputs[0] * 1000.0
        dy = outputs[1] * 1000.0
        predicted_spoofer = np.array([[centroid_x + dx, centroid_y + dy]])
        
        return predicted_spoofer


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

def generate_data(path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_spoofers):
    """生成测试数据"""
    if os.path.exists(path):
        os.remove(path)  # 强制重新生成数据
    
    with h5py.File(path, 'w') as f:
        with tqdm(total=num_samples, desc="生成测试数据", unit="sample") as pbar:
            for i in range(num_samples):        #生成xxx个样本
                # 只生成1个干扰源
                receivers, spoofers = geshispfann2.generation_data(
                    x_min, x_max, y_min, y_max,
                    num_receivers, num_spoofers
                )
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('receivers', data=receivers)
                grp.create_dataset('spoofers', data=spoofers)
                pbar.update(1)
                pbar.set_postfix(last_sample=f"接收机:{len(receivers)} 干扰源:{len(spoofers)}")


# --------------------- 主程序部分 ---------------------
if __name__ == "__main__":
    # 训练模型
    #train_model()
    
    # 初始化定位器
    locator = NeuralLocalizer()
    
    # 仿真参数
    num_simulations = 1
    x_min, x_max = 0, 2200
    y_min, y_max = 0, 2200
    num_spoofers = 1
    num_receivers = 100
    
    # 统计误差
    all_errors = []
    all_cluster_errors = []
    error2 = []
    cluster_error2 = []
    
    
    with tqdm(range(num_simulations), desc="运行仿真", unit="sim") as sim_pbar:
        for sim in sim_pbar:
            # 生成测试数据
            test_path = "test_data.h5"
            num_samples = 1     
            generate_data(test_path, num_samples,x_min, x_max, y_min, y_max,num_receivers, num_spoofers)

            #数据加载与处理
            with h5py.File(test_path, 'r') as f:
                # 遍历所有样本组
                for key in tqdm(f.keys(), desc="加载测试数据"):   #进度条显示加载过程
                    if not key.startswith('sample_'):           #只处理以 sample_ 开头的组（每组代表一个场景）
                        continue
                        
                    group = f[key]
                    receivers = group['receivers'][:]
                    spoofers = group['spoofers'][:]
                    
                    # 转换字节字符串为普通字符串
                    receivers = receivers.astype([
                        ('x', 'f4'), ('y', 'f4') 
                        ,('spoofer_count', 'i4')
                        ,('spoofer_types', 'U50')
                        , ('spoofer_indices', 'U50')
                    ])
                    # 为每个干扰源创建样本
                    for spoofer in spoofers:
                        sp_x, sp_y = spoofer
                        
                        # 找出受当前干扰源影响的接收机（距离<=300米）
                        distances = np.sqrt((receivers['x'] - sp_x)**2 + (receivers['y'] - sp_y)**2)
                        affected_mask = distances <= 300
                        affected_receivers = receivers[affected_mask]

            predicted_postions  = []

            # 使用神经网络预测
            predicted_spoofers = locator.predict(affected_receivers)
            predicted_postions.append(predicted_spoofers[0])
            # 计算误差
            current_predicted = np.array(predicted_postions)
            #print(spoofers)
            #print(predicted_postions)
            errors = calculate_paired_errors(spoofers, current_predicted)
          
            #print(errors)
            #print(cluster_errors)
            all_errors.extend(errors['distances'])
            '''
            predicted_spoofers = locator.predict(receivers)
            
            # 计算误差
            errors = calculate_paired_errors(spoofers, predicted_spoofers)['distances']
            print(errors)
            '''
            # 计算平方误差并添加到 error2 列表
            squared_errors = [e ** 2 for e in errors['distances']]  # 计算每个误差的平方

            error2.extend(squared_errors)  # 添加到 error2 列表
            #print(all_errors)
            
            # 更新进度条
            #if errors:
            #    sim_pbar.set_postfix(avg_error=f"{np.mean(errors):.2f}m")
            
            # 可视化
            plt.figure(figsize=(8, 6))
            
            # 绘制接收机
            normal_mask = receivers['spoofer_count'] == 0
            interfered_mask = receivers['spoofer_count'] > 0
            
            plt.scatter(receivers[normal_mask]['x'], receivers[normal_mask]['y'],
                        c='lightgray', s=10, alpha=0.5, label='正常接收机')
            plt.scatter(receivers[interfered_mask]['x'], receivers[interfered_mask]['y'],
                        c='red', s=20, alpha=0.7, label='受干扰接收机')
            
            # 绘制真实干扰源
            plt.scatter(spoofers[:, 0], spoofers[:, 1], marker='*',
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
    error2 = np.array(error2)
    mean_error2= np.mean(error2)
    RMSE_error = np.sqrt(mean_error2)if mean_error2 else 0
    print(f" 最小误差为 {min_error:.2f}m")
    print(f" 最大误差为 {max_error:.2f}m")
    print(f" 平均误差为 {mean_error:.2f}m, RMSE为 {RMSE_error:.2f}m")
    

    plt.figure(figsize=(6, 6))
    #bins = np.linspace(0, np.max(cluster_errors), 30)
    max_error = np.max(all_errors)
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
    plt.show()



