import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """使用最小二乘法拟合模型"""
        # 添加偏置项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 计算最优参数
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
    def predict(self, X):
        """预测"""
        if self.weights is None or self.bias is None:
            raise Exception("模型尚未训练")
            
        return X.dot(self.weights) + self.bias
    
    def get_params(self):
        """获取模型参数"""
        return {
            'weights': self.weights,
            'bias': self.bias
        } 