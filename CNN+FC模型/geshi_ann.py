import numpy as np 
import matplotlib.pyplot as plt 
import random
from datetime import datetime , timezone
import math
import pandas as pd
from scipy.spatial import distance
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 参数
f_mhz = 2046
gnss_power = -130
noise_floor = -90
#interferers_power = -60  # 每个干扰源的发射功率 (dBm)
frequency = 2.4        # 频率 (GHz)
threshold_jammer = -130
threshold_spoofer = -155

def free_space_loss(d_km, freq_mhz):
    return 32.45 + 20*np.log10(freq_mhz) + 20*np.log10(d_km)

def calculate_pw(dists_jammers, pt_jammer, jammers_freq):
    # 压制式干扰的接收功率
    d_km_jammers = dists_jammers / 1000
    L_jammers = free_space_loss(d_km_jammers, jammers_freq)
    interference_power_jammers = pt_jammer - L_jammers

    return interference_power_jammers 

def calculate_interference(distance, interferer_power, frequency):
    
    # 计算单个干扰源在接收机处产生的干扰功率 (dBm)
    # 参数: 
    #     distance: 接收机到干扰源的距离 (m)
    #     interferer_power: 干扰源发射功率 (dBm)
    #     frequency: 频率 (GHz)
    # 返回:
    #     接收机处的干扰功率 (dBm)
    
    # 自由空间路径损耗 (dB)
    # 公式: FSPL = 20log10(d) + 20log10(f) + 32.45
    # 其中 d 单位为 km, f 单位为 GHz
    distance_km = distance / 1000.0
    fspl = 20 * np.log10(distance_km) + 20 * np.log10(frequency) + 32.45
    
    # 接收功率 = 发射功率 - 路径损耗
    # 假设天线增益为0 dB
    interference_power = interferer_power - fspl
    return interference_power

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
                    num_receivers, num_interferers):
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
    
    receivers = np.random.uniform(low=[x_min, y_min], 
                                 high=[x_max, y_max],
                                 size=(num_receivers, 2))
    
    """ 
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
            np.random.seed(1234)
            # 在网格内生成随机坐标
            x = np.random.uniform(x_start, x_end)
            y = np.random.uniform(y_start, y_end)
            receivers.append([x, y])

    receivers = np.array(receivers) 
    """
    # 生成干扰源（二维坐标）
    interferers = np.random.uniform(low=[x_min, y_min],
                               high=[x_max, y_max],
                               size=(num_interferers, 2))
    interferers_power = np.random.uniform(low=-55, high=-50,size=(num_interferers, 1))
    #spoofers = generate_positions(num_spoofers, x_min, x_max, y_min, y_max)
    """ 
    # 创建计算区域
    x = np.linspace(0, 2000, 100)
    y = np.linspace(0, 2000, 100)
    X, Y = np.meshgrid(x, y)

    # 初始化干扰功率网格
    #interference_grid = np.zeros_like(X)
    
    # 计算每个接收机位置的总干扰功率
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            receiver_pos = (X[i, j], Y[i, j])
            total_power_mw = 0  # 总干扰功率 (毫瓦)
            
            # 对每个干扰源计算并叠加功率
            for interferer_pos in interferers:
                # 计算干扰源和接收机之间的距离（米）
                distance = np.sqrt((interferer_pos[0] - receiver_pos[0])**2 + 
                       (interferer_pos[1] - receiver_pos[1])**2)
                if distance <= 300:     # 150米以内认为被干扰
                    # 计算单个干扰源的干扰功率 (dBm)
                    power_dBm = calculate_interference(distance, interferer_power, frequency)
                    # 转换为线性值 (毫瓦) 并累加
                    power_mw = 10 ** (power_dBm / 10)
                    total_power_mw += power_mw
            
            # 将总功率转换回dBm并存储
            if total_power_mw > 0:
                interference_grid[i, j] = 10 * np.log10(total_power_mw)
            else:
                interference_grid[i, j] = -np.inf  # 无干扰 
    
    
    # 可视化结果
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, interference_grid, levels=20, cmap='viridis')
    plt.colorbar(contour, label='干扰功率 (dBm)')

    # 标记干扰源位置
    for idx, pos in enumerate(interferers):
        plt.plot(pos[0], pos[1], 'ro', markersize=8)
        plt.text(pos[0]+20, pos[1]+20, f'干扰源{idx+1}', color='white', fontsize=9)

    plt.title('多个干扰源叠加的干扰功率分布')
    plt.xlabel('X坐标 (米)')
    plt.ylabel('Y坐标 (米)')
    plt.grid(alpha=0.3)
    plt.show()
    """
    ri_pw = np.zeros(num_receivers) # 接收的干扰功率
    for i, receiver_pos in enumerate(receivers):
        
        total_power_mw = 0  # 总干扰功率 (毫瓦)
        # 对每个干扰源计算并叠加功率
        for j, interferer_pos in enumerate(interferers):
            
            # 计算干扰源和接收机之间的距离（米）
            distance = np.sqrt((interferer_pos[0] - receiver_pos[0])**2 + 
                    (interferer_pos[1] - receiver_pos[1])**2)
            #if distance <= 150:     # 150米以内认为被干扰
            # 计算单个干扰源的干扰功率 (dBm)
            power_dBm = calculate_interference(distance, interferers_power[j], frequency)
            # 转换为线性值 (毫瓦) 并累加
            power_mw = 10 ** (power_dBm / 10)
            total_power_mw =+ power_mw
        
        # 将总功率转换回dBm并存储
        if total_power_mw > 0:
            ri_pw[i] = 10 * np.log10(total_power_mw)
        else:
            ri_pw[i] = -np.inf  # 无干扰 

    # 统计干扰信息
    
    # 构建接收机数据结构
    dtype = [('x', 'f4'),  ('y', 'f4')
            , ('ri_pw', 'f4')
            ] 

    receivers_with_pw = np.zeros(num_receivers, dtype=dtype)
    receivers_with_pw['x'] = receivers[:, 0]
    receivers_with_pw['y'] = receivers[:, 1]
    #receivers_with_pw['jammer_num'] = num_interferers
    receivers_with_pw['ri_pw'] = ri_pw[:]
    
    
    """ 
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
    
    
    def plot_spoofer_distribution(receivers, spoofers, radius=150):
        
        #绘制每个欺骗源的接收机分布图
        #:param receivers: 接收机坐标数组 (num_receivers, 3) [x,y,z]
        #:param spoofers: 欺骗源坐标数组 (num_spoofers, 2) [x,y]
        #:param radius: 影响半径（米）
        
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
    """
    
    return receivers_with_pw, interferers,interferers_power


    