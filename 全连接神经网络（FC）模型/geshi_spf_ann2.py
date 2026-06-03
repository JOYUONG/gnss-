import numpy as np 
import matplotlib.pyplot as plt 
import random
import json
from datetime import datetime , timezone
import math
import pandas as pd
from scipy.spatial import distance
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
def generate_positions(num_points, x_min, x_max, y_min, y_max, num_candidates=100):
    """生成尽可能分散的点位"""
    if num_points == 0:
        return np.empty((0, 2))
    
    positions = np.empty((0, 2))
    # 生成第一个随机点
    first_point = np.array([[np.random.uniform(x_min, x_max), 
                        np.random.uniform(y_min, y_max)]])
    positions = np.vstack([positions, first_point])
    
    # 生成后续点
    for _ in range(num_points - 1):
        candidates = np.random.uniform(
            low=[x_min, y_min],
            high=[x_max, y_max],
            size=(num_candidates, 2)
        )
        # 计算每个候选点到现有点的最小距离
        dists = np.linalg.norm(positions - candidates[:, np.newaxis, :], axis=2)
        min_dists = np.min(dists, axis=1)
        # 选择最大最小距离的候选点
        best_idx = np.argmax(min_dists)
        positions = np.vstack([positions, candidates[best_idx]])
    
    return positions

'''
def generate_positions(num_points, x_min, x_max, y_min, y_max, num_candidates=100, border_ratio=0.05):
    """生成边界附近且尽可能分散的点位"""
    if num_points == 0:
        return np.empty((0, 2))
    
    # 计算边界实际宽度
    border_width_x = (x_max - x_min) * border_ratio
    border_width_y = (y_max - y_min) * border_ratio
    
    positions = np.empty((0, 2))
    
    # 生成第一个点（强制在边界）
    edge = np.random.choice(['left', 'right', 'top', 'bottom'])
    if edge == 'left':
        x = np.random.uniform(x_min, x_min + border_width_x)
        y = np.random.uniform(y_min, y_max)
    elif edge == 'right':
        x = np.random.uniform(x_max - border_width_x, x_max)
        y = np.random.uniform(y_min, y_max)
    elif edge == 'top':
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_max - border_width_y, y_max)
    else: # bottom
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_min + border_width_y)
    positions = np.vstack([positions, [x, y]])
    
    # 生成后续点
    for _ in range(num_points - 1):
        # 生成候选点（全在边界）
        edge_choices = np.random.randint(0, 4, size=num_candidates)
        candidates = np.zeros((num_candidates, 2))
        
        # 左边界 (x_min ~ x_min+border_width)
        left_mask = edge_choices == 0
        if np.any(left_mask):
            candidates[left_mask, 0] = np.random.uniform(
                x_min, x_min + border_width_x, size=np.sum(left_mask))
            candidates[left_mask, 1] = np.random.uniform(
                y_min, y_max, size=np.sum(left_mask))
        
        # 右边界 (x_max-border_width ~ x_max)
        right_mask = edge_choices == 1
        if np.any(right_mask):
            candidates[right_mask, 0] = np.random.uniform(
                x_max - border_width_x, x_max, size=np.sum(right_mask))
            candidates[right_mask, 1] = np.random.uniform(
                y_min, y_max, size=np.sum(right_mask))
        
        # 上边界 (y_max-border_width ~ y_max)
        top_mask = edge_choices == 2
        if np.any(top_mask):
            candidates[top_mask, 0] = np.random.uniform(
                x_min, x_max, size=np.sum(top_mask))
            candidates[top_mask, 1] = np.random.uniform(
                y_max - border_width_y, y_max, size=np.sum(top_mask))
        
        # 下边界 (y_min ~ y_min+border_width)
        bottom_mask = edge_choices == 3
        if np.any(bottom_mask):
            candidates[bottom_mask, 0] = np.random.uniform(
                x_min, x_max, size=np.sum(bottom_mask))
            candidates[bottom_mask, 1] = np.random.uniform(
                y_min, y_min + border_width_y, size=np.sum(bottom_mask))
        
        # 选择最大最小距离的点
        dists = distance.cdist(candidates, positions, 'euclidean')
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        positions = np.vstack([positions, candidates[best_idx]])
    
    return positions
'''

def generation_data( x_min, x_max,y_min, y_max,
                    num_receivers, num_spoofers):
    '''
    # 区域大小
    x_min, x_max = 0, 10000
    y_min, y_max = 0, 10000

    # 接收机数量
    num_receivers = 1000
    # 干扰源数量
    num_jammers = 3
    num_spoofers = 3
    '''
    # 生成三维接收机位置（添加高度）
    #np.random.seed(42)
    '''
    receivers = np.random.uniform(low=[x_min, y_min], 
                                 high=[x_max, y_max],
                                 size=(num_receivers, 2))
    '''
    
    def get_rows_cols(n):
        #计算最佳的行列划分，使网格尽量接近正方形 
        max_row = int(np.sqrt(n))
        for row in range(max_row, 0, -1):
            if n % row == 0:
                return row, n // row
        return 1, n  # 处理特殊情况（如n=0或质数）
    # 计算网格的行列数
    rows, cols = get_rows_cols(num_receivers)

    # 计算每个网格的尺寸
    dx = (x_max - x_min) / cols  # 网格宽度
    dy = (y_max - y_min) / rows  # 网格高度

    # 生成网格内随机点
    receivers = []
    for j in range(rows):
        for i in range(cols):
            # 计算当前网格的边界
            x_start = x_min + i * dx
            x_end = x_start + dx
            y_start = y_min + j * dy
            y_end = y_start + dy
            
            # 在网格内生成随机坐标
            x = np.random.uniform(x_start, x_end)
            y = np.random.uniform(y_start, y_end)
            receivers.append([x, y])

    receivers = np.array(receivers) 
    
    # 生成欺骗式干扰源（二维坐标）
    #spoofers = np.random.uniform(low=[x_min, y_min],
    #                            high=[x_max, y_max],
    #                            size=(num_spoofers, 2))
    spoofers = generate_positions(num_spoofers, x_min, x_max, y_min, y_max)
    # 定义欺骗干扰类型选项，并为每个spoofer分配唯一的类型
    '''
    spoof_options = [17, 18, 19, 20, 21,  # BDS
                     33, 34, 35, 36,      # GPS
                     49, 50, 51,          # GLONASS
                     65, 66, 67, 68]      # GALILEO
    '''
    #if num_spoofers > len(spoof_options):
    #    raise ValueError(f"num_spoofers {num_spoofers} exceeds available options {len(spoof_options)}")
    #spoofer_types = random.sample(spoof_options, num_spoofers)  # 随机分配唯一类型
    # 使用random.choices允许重复选择
    #spoofer_types = random.choices(spoof_options, k=num_spoofers)
    spoofer_types = [18,18,18]
    #spoofer_types = [18,19,20,21]
    #spoofer_types = [18,18,18,18,18]
    # 计算三维距离矩阵（考虑接收机高度）
    def calculate_distance(receivers, transmitters):

        return np.linalg.norm(receivers[:, np.newaxis, :] - transmitters,axis=2)

    dists_spoofers = calculate_distance(receivers, spoofers)

    # 统计欺骗式干扰信息
    #计算200米半径范围内的接收机
    
    spoofer_mask = dists_spoofers <= 200  # 直接使用米单位判断
    spoofer_counts = np.sum(spoofer_mask, axis=1)
    spoofer_indices = []
    spoofer_types_list = []
    for i in range(num_receivers):
        indices = np.where(spoofer_mask[i])[0]
        if indices.size > 0:
            indices_str = ','.join(map(str, indices))
            # 获取对应类型并去重(保留顺序)
            type_set = set()
            unique_types = []
            for idx in indices:
                t = str(spoofer_types[idx])
                if t not in type_set:
                    type_set.add(t)
                    unique_types.append(t)
            types_str = ','.join(unique_types)
        else:
            indices_str = ''
            types_str = ''
        spoofer_indices.append(indices_str)
        spoofer_types_list.append(types_str)
    
    # 构建接收机数据结构
    dtype = [('x', 'f4'),  ('y', 'f4')
            , ('spoofer_count', 'i4'), ('spoofer_types', 'S50')
            ,('spoofer_indices', 'S50')
            ] 

    receivers_with_cnr = np.zeros(num_receivers, dtype=dtype)
    receivers_with_cnr['x'] = receivers[:, 0]
    receivers_with_cnr['y'] = receivers[:, 1]

    receivers_with_cnr['spoofer_count'] = spoofer_counts
    receivers_with_cnr['spoofer_types'] = spoofer_types_list    # 存储干扰源索引
    receivers_with_cnr['spoofer_indices'] = spoofer_indices     # 受到干扰的接收机的索引
    '''
    plt.figure(figsize=(24, 16))
    plt.scatter(receivers[:,0], receivers[:,1],
            s=50, alpha=0.7, label='Receivers')
    #plt.scatter(jammers['x'], jammers['y'], marker='x', c='red',
    #        s=100, label='Jammers')
    plt.scatter(spoofers[:,0], spoofers[:,1], marker='x', c='blue',
            s=100, label='Spoofers (200m radius)')  # 添加半径说明
    #plt.colorbar(label='CNR (dB)')
    plt.title("Receiver Distribution with CNR and Interferers")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnr_distribution.png')
    plt.show()
    '''
    '''
    def plot_spoofer_distribution(receivers, spoofers, radius=150):
        """
        绘制每个欺骗源的接收机分布图
        :param receivers: 接收机坐标数组 (num_receivers, 3) [x,y,z]
        :param spoofers: 欺骗源坐标数组 (num_spoofers, 2) [x,y]
        :param radius: 影响半径（米）
        """
        num_spoofers = spoofers.shape[0]
        
        # 计算二维距离矩阵（忽略高度）
        dists = distance.cdist(receivers[:, :2], spoofers, 'euclidean')  # shape: (num_receivers, num_spoofers)

        # 创建子图
        fig, axs = plt.subplots(1, num_spoofers, figsize=(6, 6), sharey=True)
        if num_spoofers == 1:
            axs = [axs]  # 确保单个子图也能用循环处理
        
        # 统一颜色编码
        colors = ['#2ca02c', '#d62728']  # 未受影响/受影响颜色

        for j in range(num_spoofers):
            # 当前欺骗源信息
            sp_x, sp_y = spoofers[j]
            ax = axs[j]
            
            # 获取影响状态
            affected = dists[:, j] <= radius
            
            # 绘制所有接收机
            ax.scatter(receivers[:, 0], receivers[:, 1], 
                    c=[colors[0] if not a else colors[1] for a in affected],
                    s=10, alpha=0.6, edgecolors='none'
                    , label=f'Affected: {np.sum(affected)}  / {num_receivers}')
            
            # 绘制欺骗源位置
            ax.scatter(sp_x, sp_y, marker='*', s=200, 
                    c='gold', edgecolors='black', linewidths=1,
                    label=f'Spoofer {j+1}\n({sp_x:.0f}, {sp_y:.0f})')
            
            # 绘制影响范围
            circle = plt.Circle((sp_x, sp_y), radius, 
                            color='red', alpha=0.2, linestyle='--', fill=False)
            ax.add_patch(circle)
            
            # 图例和标注
            ax.set_xlabel('X (m)', fontsize=9)
            if j == 0:
                ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_title(f'Spoofer #{j+1} Coverage (Radius={radius}m)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置显示范围（以欺骗源为中心显示2倍半径范围）
            display_radius = radius * 2
            ax.set_xlim(sp_x - display_radius, sp_x + display_radius)
            ax.set_ylim(sp_y - display_radius, sp_y + display_radius)

        plt.tight_layout()
        plt.savefig('spoofer_distribution.png')
        plt.show()
    plot_spoofer_distribution(receivers, spoofers)
    '''
    return receivers_with_cnr, spoofers