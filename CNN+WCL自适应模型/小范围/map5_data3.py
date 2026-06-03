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

def read_coordinates(file_path):
    """
    读取txt格式的坐标文件
    格式示例：
    2.64705882352941e-002    5.32142857142857e-001
    4.70588235294118e-002    5.50000000000000e-001
    """
    coordinates = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 跳过空行
                if not line.strip():
                    continue
                
                # 分割行数据
                parts = line.split()
                
                # 确保每行有两个数值
                if len(parts) < 2:
                    print(f"警告: 跳过格式不正确的行: {line.strip()}")
                    continue
                
                try:
                    # 转换科学计数法为浮点数
                    x = float(parts[0])*1000
                    y = float(parts[1])*1000
                    coordinates.append((x, y))
                except ValueError:
                    print(f"警告: 无法转换数值: {line.strip()}")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
    
    return coordinates

def generate_points_on_road(receiver_positions, num_interferers):
    """
    从接收机位置中随机选择两个点，在两点连线上生成n等分点作为干扰源位置
    
    参数:
    receiver_positions - 接收机位置列表 [(x1, y1), (x2, y2), ...]
    num_interferers - 要生成的干扰源数量
    
    返回:
    干扰源位置列表 [(x, y), ...]
    """
    if len(receiver_positions) < 2:
        print("错误: 需要至少2个接收机位置")
        return []
    
    # 随机选择两个不同的接收机位置
    idx1, idx2 = random.sample(range(len(receiver_positions)), 2)
    point1 = receiver_positions[idx1]
    point2 = receiver_positions[idx2]
    
    # 计算两点之间的向量
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # 计算两点间距离
    #distance = math.sqrt(dx**2 + dy**2)
    # 生成等分点
    roindex = np.random.randint(3,8)
    i =np.random.randint(1,roindex-1)
    # 计算比例 (0到1之间)
    t = i / (roindex)
    interferers = []
    # 计算等分点坐标
    x = point1[0] + t * dx
    y = point1[1] + t * dy
    interferers.append((x, y))
    
    return interferers

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
    
    file_path = "road2s.txt"  # 替换为您的文件路径
    coords = read_coordinates(file_path)
    # 接收机位置
    receiver_positions = coords
    #print(f"接收机位置 {len(receiver_positions)} ")
    num_receivers = len(receiver_positions)


    # 生成1个干扰源位置
    #num_jammers = 1
    interferers_postion = generate_points_on_road(receiver_positions, num_interferers)
    interferers = np.array(interferers_postion[0])

    interferers_power = np.random.uniform(low=-50, high=-45,size=(num_interferers, 1))
    #spoofers = generate_positions(num_spoofers, x_min, x_max, y_min, y_max)
    # 统计干扰信息
    # 构建接收机数据结构
    dtype = [('x', 'f4'),  ('y', 'f4'), ('ri_pw', 'f4')]
    receivers_with_pw = np.zeros(num_receivers, dtype=dtype)
    ri_pw = np.zeros(num_receivers) # 接收的干扰功率
    for i, receiver_pos in enumerate(receiver_positions):
        
        total_power_mw = 0  # 总干扰功率 (毫瓦)
        # 对每个干扰源计算并叠加功率
            
        # 计算干扰源和接收机之间的距离（米）
        distance = np.sqrt((interferers[0] - receiver_pos[0])**2 + 
                (interferers[1] - receiver_pos[1])**2)
        if distance >= 10:     
            # 计算单个干扰源的干扰功率 (dBm)
            power_dBm = calculate_interference(distance, interferers_power, frequency)
            # 转换为线性值 (毫瓦) 并累加
            power_mw = 10 ** (power_dBm / 10)
            total_power_mw =+ power_mw
            
            # 将总功率转换回dBm并存储，处理数值稳定性
            if total_power_mw > 1e-12:  # 设置最小阈值避免log(0)
                ri_pw[i] = 10 * np.log10(max(total_power_mw, 1e-12))
            else:
                ri_pw[i] = -150.0  # 使用合理的最小值代替-inf

            # 限制功率范围防止数值溢出
            ri_pw[i] = np.clip(ri_pw[i], -150.0, 50.0)
            
            receivers_with_pw[i]['x'] = float(receiver_positions[i][0])
            receivers_with_pw[i]['y'] = float(receiver_positions[i][1])
            receivers_with_pw[i]['ri_pw'] = float(ri_pw[i])
        else:
            receivers_with_pw[i]['x'] = float(receiver_positions[i][0])
            receivers_with_pw[i]['y'] = float(receiver_positions[i][1])
            receivers_with_pw[i]['ri_pw'] = -150.0
    """ 
    plt.figure(figsize=(24, 16))
    plt.scatter(receivers[:,0], receivers[:,1],
            s=50, alpha=0.7, label='Receivers')
    #plt.scatter(jammers['x'], jammers['y'], marker='x', c='red',
    #        s=100, label='Jammers')
    plt.scatter(interferers[:,0], interferers[:,1], marker='x', c='blue',
            s=100)  # 添加半径说明
    #plt.colorbar(label='CNR (dB)')
    plt.title("Receiver Distribution with CNR and Interferers")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnr_distribution.png')
    plt.show() 
    """
    """ 
    plt.figure(figsize=(8, 8))

    # 1. 绘制水平主路（x方向）
    plt.plot([600, 1800], [1200- road_width/2, 1200- road_width/2], linewidth=1)
    plt.plot([600, 1800], [1200+ road_width/2, 1200+ road_width/2], linewidth=1)

    for i, (length, angle) in enumerate(branches):
        draw_branch(start_point[0], start_point[1], length, angle)
    # 支路5-7需要从主路中间点(1200,1200)开始绘制
    branch5_start = (1200, 1200)
    branch5_end = draw_branch(branch5_start[0], branch5_start[1], 600, 75)
    # 支路6从主路中间点(1200,1200)开始绘制
    branch6_end = draw_branch(branch5_start[0], branch5_start[1], 500, -65)
    # 支路7从支路6的3/4开始绘制，方向-115度
    branch7_end = draw_branch(branch6_mid_x, branch6_mid_y, 540, -155)

    # 获取影响状态
    affected = receivers_with_pw['ri_pw'] >= -80
    colors = ['#2ca02c', '#d62728']  # 未受影响/受影响颜色
    # 绘制所有接收机
    plt.scatter(receivers_with_pw['x'], receivers_with_pw['y'], 
            c=[colors[0] if not a else colors[1] for a in affected],
            s=50, alpha=0.6, edgecolors='none'
            , label=f'Affected: {np.sum(affected)}  / {num_receivers}')
    
    # 绘制欺骗源位置
    plt.scatter(interferers[0], interferers[1],marker='*', s=200, 
            c='gold', edgecolors='black', linewidths=1)
    
    # 图例和标注
    plt.xlabel('X (m)', fontsize=9)
    plt.ylabel('Y (m)', fontsize=9)
    #plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    #plt.savefig('spoofer_distribution.png')
    plt.show() 
    """

    """ 
    def plot_spoofer_distribution(receivers, spoofers, radius=150):
        
        #绘制每个欺骗源的接收机分布图
        #:param receivers: 接收机坐标数组 (num_receivers, 3) [x,y,z]
        #:param spoofers: 欺骗源坐标数组 (num_spoofers, 2) [x,y]
        #:param radius: 影响半径（米）
        
        num_spoofers = spoofers.shape[0]
        
        # 计算二维距离矩阵（忽略高度）
        #dists = distance.cdist(receivers[:, :2], spoofers, 'euclidean')  # shape: (num_receivers, num_spoofers)

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
            affected = receivers['ri_pw'] >= -80
            
            # 绘制所有接收机
            ax.scatter(receivers['x'], receivers['y'], 
                    c=[colors[0] if not a else colors[1] for a in affected],
                    s=10, alpha=0.6, edgecolors='none'
                    , label=f'Affected: {np.sum(affected)}  / {num_receivers}')
            
            # 绘制欺骗源位置
            ax.scatter(sp_x, sp_y, marker='*', s=200, 
                    c='gold', edgecolors='black', linewidths=1,
                    label=f'Spoofer {j+1}\n({sp_x:.0f}, {sp_y:.0f})')
            
            # 绘制影响范围
            # circle = plt.Circle((sp_x, sp_y), radius, 
            #                 color='red', alpha=0.2, linestyle='--', fill=False)
            # ax.add_patch(circle)
            
            # 图例和标注
            ax.set_xlabel('X (m)', fontsize=9)
            if j == 0:
                ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_title(f'Spoofer #{j+1} Coverage (Radius={radius}m)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置显示范围（以欺骗源为中心显示2倍半径范围）
            # display_radius = radius * 2
            # ax.set_xlim(sp_x - display_radius, sp_x + display_radius)
            # ax.set_ylim(sp_y - display_radius, sp_y + display_radius)

        plt.tight_layout()
        plt.savefig('spoofer_distribution.png')
        plt.show()
    plot_spoofer_distribution(receivers_with_pw, interferers)
     """
    
    return receivers_with_pw, interferers,interferers_power

# receivers, jammers,jam_pw = generation_data(
#                     x_min=0, x_max=2000, y_min=0, y_max=2000,
#                     num_receivers=100, num_interferers=1)

