# 不平衡分类问题研究

这个项目主要研究机器学习中的不平衡分类问题，使用银行营销数据集作为示例。

## 项目结构

- `imbalanced_classification.py`: 主程序文件，包含数据预处理、模型训练和评估的完整流程
- `bank-additional-names.txt`: 数据集的特征说明文件
- `requirements.txt`: 项目依赖包列表
- `.gitignore`: Git忽略文件配置

## 环境要求

- Python 3.x
- 依赖包（见requirements.txt）：
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - imbalanced-learn

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/TracyStark/-.git
```

2. 创建并激活虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保已安装所有依赖
2. 运行主程序：
```bash
python imbalanced_classification.py
```

## 项目特点

- 实现了多种处理不平衡数据的方法：
  - 欠采样（RandomUnderSampler）
  - 过采样（SMOTE）
  - 组合采样（SMOTEENN）
- 使用多种评估指标：
  - 准确率
  - 精确率
  - 召回率
  - F1分数
  - ROC曲线
  - 混淆矩阵
- 包含详细的数据可视化分析

## 注意事项

- 数据集文件（.csv）已被添加到.gitignore中，需要自行下载
- 运行程序前请确保已正确配置Python环境 