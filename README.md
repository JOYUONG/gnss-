# GNSS 干扰源智能定位算法库

<div align="center">

**基于深度学习的 GNSS 欺骗式干扰源定位系统**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

## 📖 项目简介

本项目是一个基于深度学习的 GNSS（全球导航卫星系统）干扰源定位算法库，旨在通过受干扰接收机的空间分布特征，实现对欺骗式干扰源的高精度定位和功率估计。项目包含多种模型架构，从基础的全连接神经网络到 CNN+WCL 自适应融合模型，逐步提升定位精度和鲁棒性。相关论文DOI: 10.1109/IEEECONF65522.2025.11136973

### 核心技术原理

欺骗式干扰源会对其周围一定范围内的接收机产生影响，受影响接收机的空间分布特征（如方差、长宽比、分布密度等）与干扰源的实际位置存在特定关系。通过深度学习方法，可以学习到这种映射关系，并利用它从受干扰接收机的分布特征预测干扰源的位置。

---

## 📁 项目结构

```
定位算法/
├── 全连接神经网络（FC）模型/          # 基础 FC 模型
│   ├── main_spf_ann5.py              # 主程序
│   ├── geshi_spf_ann.py              # 数据生成模块
│   ├── geshi_spf_ann2.py             # 数据生成模块（扩展版）
│   ├── best_model.pth                # 预训练模型权重
│   └── main_spf_ann5说明.md          # 详细说明文档
│
├── CNN+FC模型/                        # CNN+FC 模型
│   ├── main_cnn.py                   # 主程序
│   ├── geshi_ann.py                  # 数据生成模块
│   ├── best_model_stage1.pth         # 阶段1预训练模型
│   ├── best_model_stage2.pth         # 阶段2预训练模型
│   ├── best_model_stage3.pth         # 阶段3预训练模型
│   └── main_cnn说明文档.md           # 详细说明文档
│
├── CNN+WCL自适应模型/                  # CNN+WCL 自适应融合模型
│   ├── 小范围（1km×1km）/            # 小范围场景
│   │   ├── main_cnn4-3-5.py          # 主程序（版本1）
│   │   ├── main_cnn4-3-6.py          # 主程序（版本2）
│   │   ├── map5_data3.py             # 数据生成模块
│   │   ├── road2s.txt                # 道路数据
│   │   ├── best_cnn2model435_stage3.pth  # 预训练模型（版本1）
│   │   ├── best_cnn2model436_stage3.pth  # 预训练模型（版本2）
│   │   ├── main_cnn4-3-5说明文档.md  # 详细说明文档
│   │   └── 4-3-6说明.md              # 版本差异说明
│   │
│   └── 大范围（4.5km×4.5km）/        # 大范围场景
│       ├── main_cnn4-3-11.py         # 主程序
│       ├── main_spf_ann6.py          # 辅助程序
│       ├── map2.txt                  # 地图数据
│       ├── best_cnn2model4311_stage3.pth  # 预训练模型
│       └── main_cnn4-3-11说明.md     # 详细说明文档
│
└── CNN+FC(欺骗版)/                    # CNN+FC 欺骗版模型
    ├── main_cnn_spf.py               # 主程序（版本1）
    ├── main_cnn_spf2.py              # 主程序（版本2）
    ├── geshi_cnn_spf.py              # 数据生成模块（版本1）
    ├── geshi_cnn_spf2.py             # 数据生成模块（版本2）
    ├── best_cnnspfmodel_stage.pth    # 预训练模型（版本1）
    └── best_cnnspfmodel2_stage.pth   # 预训练模型（版本2）
```

---

## 🧠 模型架构

### 1. 全连接神经网络（FC）模型

基础的干扰源定位模型，使用全连接神经网络从受干扰接收机的分布特征中学习干扰源位置。

| 特性 | 说明 |
|------|------|
| **网络结构** | 4层全连接网络（128→64→32→2） |
| **输入特征** | 7维特征向量（X/Y方差、长宽比、4个象限密度比例） |
| **输出** | 干扰源相对于接收机分布中心的偏移量 |
| **聚类算法** | SelfMergeMeanShift（自动带宽调整） |
| **定位精度** | 百米级 |

### 2. CNN+FC 模型

引入卷积神经网络进行特征提取，将接收机分布转换为功率矩阵进行处理。

| 特性 | 说明 |
|------|------|
| **网络结构** | CNN降维（100×100→3×10×10）+ 双分支全连接网络 |
| **输入** | 100×100 干扰功率矩阵 |
| **输出** | 干扰源位置（归一化坐标）+ 功率估计 |
| **训练策略** | 三阶段训练（位置→功率→综合） |
| **定位精度** | 十米级 |

### 3. CNN+WCL 自适应融合模型（小范围 1km×1km）

创新性地将 CNN 与加权质心定位（WCL）进行自适应融合，实现更精准的定位。

| 特性 | 说明 |
|------|------|
| **核心创新** | 置信度学习 + 自适应融合 |
| **融合公式** | X_final = α ⊙ X_CNN + (1 - α) ⊙ X_WCL |
| **网络结构** | CNN降维 + BatchNorm + 双分支预测 + 置信度估计 |
| **输入** | 自适应网格功率矩阵 + WCL位置 |
| **输出** | 融合位置、CNN位置、WCL位置、置信度α、功率 |
| **定位精度** | **10米级** |
| **功率精度** | **1dBm级** |

### 4. CNN+WCL 自适应融合模型（大范围 4.5km×4.5km）

针对大范围场景优化的模型，采用固定网格单元大小、自适应区域选择策略。

| 特性 | 说明 |
|------|------|
| **场景范围** | 4.5km × 4.5km |
| **网格策略** | 固定网格单元大小，自适应区域选择 |
| **区域扩展** | 区域过大时自动扩大网格单元大小 |
| **定位精度** | **10米级** |
| **功率精度** | **1dBm级** |

### 5. CNN+FC 欺骗版模型

专门针对欺骗式干扰场景优化的模型变体。

| 特性 | 说明 |
|------|------|
| **优化方向** | 针对欺骗式干扰特征进行专门优化 |
| **数据生成** | 模拟真实的欺骗式干扰场景 |
| **训练策略** | 分阶段训练，优化欺骗特征提取 |

---

## 📊 性能对比

| 模型 | 定位精度 | 功率精度 | 适用场景 | 计算复杂度 |
|------|----------|----------|----------|------------|
| FC 模型 | 百米级 | - | 快速原型验证 | 低 |
| CNN+FC 模型 | 十米级 | 一般 | 中等精度需求 | 中 |
| CNN+WCL（小范围） | **10米级** | **1dBm级** | 城市密集区域 | 中 |
| CNN+WCL（大范围） | **10米级** | **1dBm级** | 广域覆盖场景 | 中 |
| CNN+FC 欺骗版 | 十米级 | 一般 | 欺骗式干扰 | 中 |

---

## 🛠️ 环境配置

### 依赖库

| 库名称 | 推荐版本 | 用途 |
|--------|----------|------|
| Python | 3.8+ | 运行环境 |
| numpy | 1.23.5+ | 数值计算 |
| pandas | 1.5.3+ | 数据处理 |
| torch | 2.0.1+ | 深度学习框架 |
| h5py | 3.8.0+ | HDF5文件处理 |
| matplotlib | 3.7.1+ | 数据可视化 |
| scipy | 1.10.0+ | 科学计算 |
| tqdm | 4.65.0+ | 进度条显示 |

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/gnss-interference-localization.git
cd gnss-interference-localization

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install numpy pandas torch h5py matplotlib scipy tqdm
```

---

## 🚀 快速开始

### 1. 使用预训练模型进行定位

```python
# 以 CNN+WCL 小范围模型为例
from CNN+WCL自适应模型.小范围 import NeuralLocalizer

# 初始化定位器
locator = NeuralLocalizer("best_cnn2model435_stage3.pth")

# 准备受干扰接收机数据（包含位置和功率信息）
receivers = load_receivers_data()  # 替换为实际数据加载函数

# 执行定位预测
pred_pos_final, pred_pos_cnn, pred_pos_wcl, alpha, pred_pw = locator.predict(receivers)

print(f"融合预测位置: {pred_pos_final}")
print(f"CNN预测位置: {pred_pos_cnn}")
print(f"WCL预测位置: {pred_pos_wcl}")
print(f"置信度权重: {alpha}")
print(f"预测功率: {pred_pw} dBm")
```

### 2. 训练新模型

```python
# 以 CNN+FC 模型为例
from CNN_FC模型.main_cnn import train_model

# 执行三阶段训练
train_model()

# 训练完成后，模型将保存为 best_model_stage3.pth
```

### 3. 生成训练数据

```python
# 以 FC 模型为例
from FC模型.geshi_spf_ann import generate_training_data

# 生成训练数据（保存为HDF5格式）
generate_training_data("training_data.h5", num_samples=10000)
```

---

## 📈 核心技术亮点

### 1. 自适应网格技术
根据受干扰接收机的分布动态调整网格大小和中心位置，使模型能够适应不同场景下的干扰分布。

### 2. 置信度学习机制
模型自动学习置信度 α，动态平衡 CNN 预测和 WCL 结果的权重，在复杂环境中自动增加更可靠方法的权重。

### 3. 分阶段训练策略
- **阶段1**：专注优化位置预测
- **阶段2**：专注优化功率预测
- **阶段3**：综合优化所有任务，学习置信度

### 4. 多目标损失函数
L_CAGE = 0.7 × L_GNN + 0.3 × L_adapt，平衡 CNN 预测和融合结果。

### 5. 匈牙利算法匹配
使用匈牙利算法进行真实位置与估计位置的最优匹配，提高评估准确性。

---

## 📝 输入数据格式

### 接收机数据格式

接收机数据应包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| x | float32 | 接收机X坐标（米） |
| y | float32 | 接收机Y坐标（米） |
| spoofer_count | int32 | 受欺骗次数 |
| spoofer_types | string | 干扰源类型 |
| spoofer_indices | string | 干扰源索引 |

### 训练数据格式

训练数据采用 HDF5 格式存储，包含：
- 接收机位置和受干扰功率
- 干扰源真实位置和功率

---

## 🔧 参数配置

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-3 | 学习率 |
| batch_size | 64 | 批次大小 |
| num_epochs | 100 | 训练轮数 |
| num_receivers | 200 | 每个场景的接收机数量 |
| num_interferers | 1 | 干扰源数量 |
| grid_size | 100 | 网格大小 |

### 聚类参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| bandwidth | 150 | MeanShift聚类带宽 |
| min_cluster_size | 10 | 最小聚类大小 |

---

## 📊 可视化功能

项目包含丰富的可视化功能：

1. **定位结果可视化**
   - 真实干扰源位置
   - 融合预测位置
   - CNN预测位置
   - WCL预测位置

2. **误差分析**
   - 误差分布直方图
   - 累积误差分布曲线
   - 误差统计指标（最小/最大/平均/RMSE）

3. **置信度分析**
   - 置信度α分布图
   - CNN vs WCL 权重对比

4. **功率预测**
   - 功率预测误差分布
   - 真实功率 vs 预测功率散点图

## 🤝 贡献指南

欢迎对本项目进行贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📧 联系方式

如有任何问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至：[3435689767@qq.com]

---

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员和开发者。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！⭐**

</div>
