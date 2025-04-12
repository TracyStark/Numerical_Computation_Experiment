import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def load_datasets():
    """加载多个不平衡数据集"""
    datasets = {}
    
    # 1. 信用卡欺诈检测数据集
    print("\n=== 加载信用卡欺诈检测数据集 ===")
    df_credit = pd.read_csv('creditcard.csv')
    X_credit = df_credit.drop('Class', axis=1)
    y_credit = df_credit['Class']
    datasets['Credit Card Fraud'] = (X_credit, y_credit)
    
    # 2. 银行营销数据集
    print("\n=== 加载银行营销数据集 ===")
    df_bank = pd.read_csv('bank-additional-full.csv', sep=';')
    
    # 数据预处理
    le = LabelEncoder()
    categorical_columns = df_bank.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_bank[col] = le.fit_transform(df_bank[col])
    
    X_bank = df_bank.drop('y', axis=1)
    y_bank = df_bank['y']
    datasets['Bank Marketing'] = (X_bank, y_bank)
    
    # 打印数据集信息
    for name, (X, y) in datasets.items():
        print(f"\n{name}数据集信息:")
        print(f"数据集大小: {X.shape}")
        print(f"正类样本比例: {y.mean():.4f}")
        print(f"特征数量: {X.shape[1]}")
        print(f"正类样本数量: {len(y[y==1])}")
        print(f"负类样本数量: {len(y[y==0])}")
    
    return datasets

def evaluate_models(X, y, dataset_name):
    """评估不同模型在不同参数下的性能"""
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 定义模型和参数
    models = {
        'Decision Tree': DecisionTreeClassifier,
        'Naive Bayes': GaussianNB
    }
    
    dt_params = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    results = []
    
    # 1. 原始数据评估
    print(f"\n=== {dataset_name} - 原始数据评估 ===")
    for model_name, model_class in models.items():
        if model_name == 'Decision Tree':
            for max_depth in dt_params['max_depth']:
                for min_samples in dt_params['min_samples_split']:
                    print(f"训练 {model_name} (max_depth={max_depth}, min_samples_split={min_samples})...")
                    model = model_class(max_depth=max_depth, min_samples_split=min_samples)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Parameters': f'max_depth={max_depth}, min_samples_split={min_samples}',
                        'F1 Score': f1,
                        'Augmented': False
                    })
        else:
            print(f"训练 {model_name}...")
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Parameters': 'default',
                'F1 Score': f1,
                'Augmented': False
            })
    
    # 2. 使用SMOTE扩充数据
    print(f"\n=== {dataset_name} - 数据扩充 ===")
    print("使用SMOTE算法扩充少数类样本...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"扩充前训练集大小: {X_train.shape}")
    print(f"扩充后训练集大小: {X_train_smote.shape}")
    print(f"扩充前正类样本比例: {y_train.mean():.4f}")
    print(f"扩充后正类样本比例: {y_train_smote.mean():.4f}")
    
    # 3. 扩充后数据评估
    print(f"\n=== {dataset_name} - 扩充后数据评估 ===")
    for model_name, model_class in models.items():
        if model_name == 'Decision Tree':
            for max_depth in dt_params['max_depth']:
                for min_samples in dt_params['min_samples_split']:
                    print(f"训练 {model_name} (max_depth={max_depth}, min_samples_split={min_samples})...")
                    model = model_class(max_depth=max_depth, min_samples_split=min_samples)
                    model.fit(X_train_smote, y_train_smote)
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    results.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Parameters': f'max_depth={max_depth}, min_samples_split={min_samples}',
                        'F1 Score': f1,
                        'Augmented': True
                    })
        else:
            print(f"训练 {model_name}...")
            model = model_class()
            model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Parameters': 'default',
                'F1 Score': f1,
                'Augmented': True
            })
    
    return pd.DataFrame(results)

def plot_results(results_df):
    """可视化结果"""
    # 为每个数据集创建单独的图表
    for dataset in results_df['Dataset'].unique():
        plt.figure(figsize=(15, 8))
        dataset_results = results_df[results_df['Dataset'] == dataset]
        sns.barplot(data=dataset_results, x='Model', y='F1 Score', hue='Augmented')
        plt.title(f'{dataset} - Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'model_comparison_{dataset.replace(" ", "_")}.png')
        plt.close()
    
    # 创建所有数据集的综合比较图
    plt.figure(figsize=(15, 8))
    sns.barplot(data=results_df, x='Dataset', y='F1 Score', hue='Model')
    plt.title('Overall Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('overall_comparison.png')
    plt.close()

def main():
    # 1. 加载数据集
    datasets = load_datasets()
    
    # 2. 评估模型
    all_results = []
    for dataset_name, (X, y) in datasets.items():
        results = evaluate_models(X, y, dataset_name)
        all_results.append(results)
    
    # 合并所有结果
    final_results = pd.concat(all_results)
    
    # 3. 保存结果
    final_results.to_csv('classification_results.csv', index=False)
    
    # 4. 绘制结果
    plot_results(final_results)
    
    print("\n实验完成！结果已保存到 classification_results.csv 和相应的图表文件中")

if __name__ == "__main__":
    main() 
    