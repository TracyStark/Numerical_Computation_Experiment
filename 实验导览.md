# 数值计算实验导览

本仓库包含两个主要的机器学习实验项目，分别研究分类问题和回归问题。以下是两个实验的简要介绍和导航。

## 1. 不平衡分类问题研究

### 实验简介
- 研究机器学习中的不平衡分类问题
- 使用银行营销数据集作为示例
- 实现了多种处理不平衡数据的方法

### 主要特点
- 实现了多种采样方法：
  - 欠采样（RandomUnderSampler）
  - 过采样（SMOTE）
  - 组合采样（SMOTEENN）
- 使用多种评估指标：
  - 准确率、精确率、召回率、F1分数
  - ROC曲线和混淆矩阵
- 包含详细的数据可视化分析

### 相关文件
- 主程序：`Classification/imbalanced_classification.py`
- 数据集：`Classification/bank-additional.csv`
- 说明文档：`Classification/README.md`

## 2. 多变量线性回归实验

### 实验简介
- 研究多变量线性回归和岭回归模型
- 在两个不同数据集上进行实验：
  - 空气质量数据集（AirQualityUCI）
  - 加州房价数据集（California Housing）

### 主要特点
- 实现了两种回归模型：
  - 普通线性回归
  - 岭回归（L2正则化）
- 完整的实验流程：
  - 数据预处理
  - 特征选择
  - 模型训练和评估
- 丰富的可视化结果

### 相关文件
- 主程序：`LinearRegression/experiment.py`
- 数据集：
  - `LinearRegression/AirQualityUCI.csv`
  - 加州房价数据集（scikit-learn内置）
- 说明文档：`LinearRegression/README.md`

## 实验环境要求

两个实验都需要以下基本环境：
- Python 3.x
- 核心依赖包：
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

具体依赖请参考各实验目录下的 `requirements.txt` 文件。

## 快速开始

1. 克隆仓库
2. 为每个实验创建独立的虚拟环境
3. 安装相应依赖
4. 运行实验程序

详细步骤请参考各实验目录下的 README.md 文件。

## 注意事项

- 请确保已正确配置Python环境
- 部分数据集文件较大，需要自行下载
- 建议使用虚拟环境运行实验
- 实验过程中生成的可视化结果会保存在相应目录下 