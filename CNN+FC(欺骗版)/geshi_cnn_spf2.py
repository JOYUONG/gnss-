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

def generation_data( x_min, x_max,y_min, y_max,
                    num_receivers, num_interferers):

    # 生成三维接收机位置（添加高度）
    #np.random.seed(42)
    
    receivers = np.random.uniform(low=[x_min, y_min], 
                                 high=[x_max, y_max],
                                 size=(num_receivers, 2))
    
    # 生成干扰源（二维坐标）
    interferers = np.random.uniform(low=[x_min, y_min],
                               high=[x_max, y_max],
                               size=(num_interferers, 2))
    #spoofers = generate_positions(num_spoofers, x_min, x_max, y_min, y_max)
    
    ri_pw = np.zeros(num_receivers) # 接收的干扰功率
    for i, receiver_pos in enumerate(receivers):

        # 对每个干扰源计算并叠加功率
        for j, interferer_pos in enumerate(interferers):
            
            # 计算干扰源和接收机之间的距离（米）
            distance = np.sqrt((interferer_pos[0] - receiver_pos[0])**2 + 
                    (interferer_pos[1] - receiver_pos[1])**2)
            if distance <= 200:     # 200米以内认为被干扰
                ri_pw[i] = 1
            else:
                ri_pw[i] = 0

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
    
    return receivers_with_pw, interferers
if __name__ == '__main__':
    generation_data(0, 1000, 0, 1000, 100, 100)

    