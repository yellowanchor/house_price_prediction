"""
房价预测模型 - XGBoost + MLP + K折交叉验证
使用RMSE评估模型性能
"""

import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# 下载数据集
print("正在下载房价预测数据集...")
path = kagglehub.dataset_download("cmekta/house-price-prediction-dataset")
print(f"数据集路径: {path}")

# 查找数据文件
train_file = os.path.join(path, "train.csv")
test_file = os.path.join(path, "test.csv")

# 如果没找到，列出目录内容
if not os.path.exists(train_file):
    print("目录内容:", os.listdir(path))
    # 尝试查找CSV文件
    for f in os.listdir(path):
        if 'train' in f.lower() and f.endswith('.csv'):
            train_file = os.path.join(path, f)
        if 'test' in f.lower() and f.endswith('.csv'):
            test_file = os.path.join(path, f)

print(f"训练数据: {train_file}")
print(f"测试数据: {test_file}")

# 加载数据
train_df = pd.read_csv(train_file)
print(f"\n训练集形状: {train_df.shape}")
print(f"测试集形状: {pd.read_csv(test_file).shape if os.path.exists(test_file) else 'N/A'}")

# 显示数据基本信息
print("\n数据前5行:")
print(train_df.head())
print("\n数据类型:")
print(train_df.dtypes)

# 数据预处理
def preprocess_data(df, is_train=True):
    """数据预处理"""
    df = df.copy()
    
    # 保存ID列（如果有）
    id_col = None
    if 'Id' in df.columns:
        id_col = df['Id']
        df = df.drop('Id', axis=1)
    elif 'id' in df.columns:
        id_col = df['id']
        df = df.drop('id', axis=1)
    
    # 分离目标变量（房价通常在最后一列或名为SalePrice的列）
    target = None
    if 'SalePrice' in df.columns:
        target = df['SalePrice']
        df = df.drop('SalePrice', axis=1)
    elif is_train and df.shape[1] > 0:
        # 假设最后一列是目标变量
        target = df.iloc[:, -1]
        df = df.drop(df.columns[-1], axis=1)
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 数值列用中位数填充
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 类别列用众数填充，然后编码
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        df[col] = pd.factorize(df[col])[0]
    
    return df, target, id_col

# 预处理数据
X, y, _ = preprocess_data(train_df, is_train=True)
print(f"\n特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K折交叉验证
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# XGBoost模型
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# MLP模型
mlp_params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1
}

# 存储结果
xgb_rmse_scores = []
mlp_rmse_scores = []
ensemble_rmse_scores = []

print("\n" + "="*60)
print("开始K折交叉验证训练...")
print("="*60)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"\n--- 第 {fold} 折 ---")
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 训练XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_rmse_scores.append(xgb_rmse)
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    
    # 训练MLP
    mlp_model = MLPRegressor(**mlp_params)
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_val)
    mlp_rmse = np.sqrt(mean_squared_error(y_val, mlp_pred))
    mlp_rmse_scores.append(mlp_rmse)
    print(f"MLP RMSE: {mlp_rmse:.4f}")
    
    # 集成预测（简单平均）
    ensemble_pred = (xgb_pred + mlp_pred) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    ensemble_rmse_scores.append(ensemble_rmse)
    print(f"集成模型 RMSE: {ensemble_rmse:.4f}")

# 输出最终结果
print("\n" + "="*60)
print("K折交叉验证结果汇总")
print("="*60)
print(f"\nXGBoost - 平均RMSE: {np.mean(xgb_rmse_scores):.4f} (+/- {np.std(xgb_rmse_scores):.4f})")
print(f"MLP     - 平均RMSE: {np.mean(mlp_rmse_scores):.4f} (+/- {np.std(mlp_rmse_scores):.4f})")
print(f"集成模型 - 平均RMSE: {np.mean(ensemble_rmse_scores):.4f} (+/- {np.std(ensemble_rmse_scores):.4f})")

# 训练最终模型（使用全部数据）
print("\n" + "="*60)
print("训练最终模型...")
print("="*60)

# 训练最终XGBoost模型
final_xgb = xgb.XGBRegressor(**xgb_params)
final_xgb.fit(X_scaled, y)
print("XGBoost 最终模型训练完成")

# 训练最终MLP模型
final_mlp = MLPRegressor(**mlp_params)
final_mlp.fit(X_scaled, y)
print("MLP 最终模型训练完成")

# 如果有测试集，进行预测
if os.path.exists(test_file):
    print("\n" + "="*60)
    print("对测试集进行预测...")
    print("="*60)
    
    test_df = pd.read_csv(test_file)
    X_test, _, test_ids = preprocess_data(test_df, is_train=False)
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    xgb_test_pred = final_xgb.predict(X_test_scaled)
    mlp_test_pred = final_mlp.predict(X_test_scaled)
    ensemble_test_pred = (xgb_test_pred + mlp_test_pred) / 2
    
    print(f"测试集预测完成，共 {len(ensemble_test_pred)} 条记录")
    print(f"预测值范围: [{ensemble_test_pred.min():.2f}, {ensemble_test_pred.max():.2f}]")

print("\n" + "="*60)
print("训练完成！")
print("="*60)

# 保存结果
results = {
    'Model': ['XGBoost', 'MLP', 'Ensemble'],
    'Mean_RMSE': [np.mean(xgb_rmse_scores), np.mean(mlp_rmse_scores), np.mean(ensemble_rmse_scores)],
    'Std_RMSE': [np.std(xgb_rmse_scores), np.std(mlp_rmse_scores), np.std(ensemble_rmse_scores)]
}
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)
print(f"\n结果已保存到 model_results.csv")
