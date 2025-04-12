import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_mse_comparison(self, alphas, mse_values, dataset_name):
        """绘制不同alpha值下的MSE比较图"""
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, mse_values, 'o-')
        plt.xscale('log')
        plt.xlabel('正则化参数 (alpha)')
        plt.ylabel('均方误差 (MSE)')
        plt.title(f'{dataset_name} - 不同正则化参数下的MSE')
        plt.grid(True)
        plt.savefig(f'mse_comparison_{dataset_name}.png')
        plt.close()
        
    def plot_prediction_vs_actual(self, y_true, y_pred, dataset_name):
        """绘制预测值与实际值的对比图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'{dataset_name} - 预测值 vs 实际值')
        plt.grid(True)
        plt.savefig(f'prediction_vs_actual_{dataset_name}.png')
        plt.close()
        
    def plot_feature_importance(self, feature_names, weights, dataset_name):
        """绘制特征重要性图"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_names, y=np.abs(weights))
        plt.xticks(rotation=45)
        plt.xlabel('特征')
        plt.ylabel('权重绝对值')
        plt.title(f'{dataset_name} - 特征重要性')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{dataset_name}.png')
        plt.close() 