import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 正则化参数
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """使用岭回归拟合模型"""
        # 添加偏置项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 构建正则化矩阵
        n_features = X_b.shape[1]
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0  # 不对偏置项进行正则化
        
        # 计算最优参数
        theta = np.linalg.inv(X_b.T.dot(X_b) + reg_matrix).dot(X_b.T).dot(y)
        
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
            'bias': self.bias,
            'alpha': self.alpha
        } 