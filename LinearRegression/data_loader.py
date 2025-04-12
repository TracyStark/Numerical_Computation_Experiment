import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    def load_air_quality(self):
        print("正在从本地文件加载空气质量数据集...")
        # 读取CSV文件，指定分隔符为分号
        df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
        
        # 删除空列
        df = df.dropna(axis=1, how='all')
        
        # 选择数值型特征（排除日期和时间列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values
        
        # 处理缺失值（将-200替换为NaN）
        X[X == -200] = np.nan
        X = self.imputer.fit_transform(X)
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 使用CO(GT)作为目标变量
        target_col = 'CO(GT)'
        y = df[target_col].values
        y[y == -200] = np.nan
        y = self.imputer.fit_transform(y.reshape(-1, 1)).ravel()
        
        # 从特征中移除目标变量
        feature_cols = [col for col in numeric_cols if col != target_col]
        X = df[feature_cols].values
        X[X == -200] = np.nan
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        
        print(f"数据集加载完成，特征形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y, feature_cols
    
    def load_california_housing(self):
        """加载加州房价数据集"""
        print("正在加载加州房价数据集...")
        # 获取数据集
        california = fetch_california_housing()
        X = california.data
        y = california.target
        feature_names = california.feature_names
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        print(f"数据集加载完成，特征形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y, feature_names
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """准备训练和测试数据集"""
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test 