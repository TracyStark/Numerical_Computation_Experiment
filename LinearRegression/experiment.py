import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from data_loader import DataLoader
from linear_regression import LinearRegression
from ridge_regression import RidgeRegression
from visualization import Visualizer

def run_experiment(dataset_name, X, y, feature_names):
    """运行实验"""
    # 初始化
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    visualizer = Visualizer()
    
    # 存储结果
    mse_results = {
        'linear': [],
        'ridge': {alpha: [] for alpha in [0.001, 0.01, 0.1, 1, 10, 100]}
    }
    
    # 5折交叉验证
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 普通线性回归
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        mse_results['linear'].append(mean_squared_error(y_test, y_pred))
        
        # 岭回归（不同alpha值）
        for alpha in mse_results['ridge'].keys():
            ridge = RidgeRegression(alpha=alpha)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            mse_results['ridge'][alpha].append(mean_squared_error(y_test, y_pred))
    
    # 计算平均MSE
    avg_mse_linear = np.mean(mse_results['linear'])
    avg_mse_ridge = {alpha: np.mean(mse_results['ridge'][alpha]) for alpha in mse_results['ridge'].keys()}
    
    # 可视化结果
    alphas = list(avg_mse_ridge.keys())
    mse_values = list(avg_mse_ridge.values())
    visualizer.plot_mse_comparison(alphas, mse_values, dataset_name)
    
    # 使用最佳alpha值进行最终预测
    best_alpha = min(avg_mse_ridge, key=avg_mse_ridge.get)
    ridge = RidgeRegression(alpha=best_alpha)
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    
    # 绘制预测结果
    visualizer.plot_prediction_vs_actual(y, y_pred, dataset_name)
    visualizer.plot_feature_importance(feature_names, ridge.weights, dataset_name)
    
    # 打印结果
    print(f"\n{dataset_name} 数据集结果:")
    print(f"普通线性回归平均MSE: {avg_mse_linear:.4f}")
    print(f"最佳正则化参数 (alpha): {best_alpha}")
    print(f"岭回归最小平均MSE: {avg_mse_ridge[best_alpha]:.4f}")

def main():
    # 加载数据
    data_loader = DataLoader()
    
    # 加州房价数据集
    X_cal, y_cal, feature_names_cal = data_loader.load_california_housing()
    run_experiment("California", X_cal, y_cal, feature_names_cal)

if __name__ == "__main__":
    main() 