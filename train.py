"""
房价预测模型 - Kaggle竞赛标准版
XGBoost + LightGBM 集成 + 完整特征工程 + K折交叉验证

Kaggle竞赛: House Prices - Advanced Regression Techniques
评估指标: RMSE (log scale)
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    DATA_PATH = os.path.dirname(os.path.abspath(__file__))
    N_FOLDS = 5
    RANDOM_STATE = 42
    TARGET = 'SalePrice'
    
    # 模型参数
    XGB_PARAMS = {
        'n_estimators': 2000,
        'max_depth': 4,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'early_stopping_rounds': 100
    }
    
    LGB_PARAMS = {
        'n_estimators': 2000,
        'max_depth': 4,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }

# ==================== 数据加载 ====================
def load_data():
    """加载训练集和测试集"""
    print("="*60)
    print("房价预测模型 - Kaggle竞赛标准版")
    print("="*60)
    
    # 尝试多个数据源
    train_path = os.path.join(Config.DATA_PATH, 'train.csv')
    test_path = os.path.join(Config.DATA_PATH, 'test.csv')
    
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        print(f"\n本地数据 - 训练集: {train_df.shape}")
        if test_df is not None:
            print(f"测试集: {test_df.shape}")
        return train_df, test_df
    
    # 使用 OpenML 数据集
    print("\n使用 OpenML 数据集...")
    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    return housing.frame, None

# ==================== 特征工程 ====================
def feature_engineering(df, is_train=True):
    """
    完整的特征工程流程
    参考 Kaggle 竞赛高分解决方案
    """
    data = df.copy()
    
    # 保存ID
    ids = None
    if 'Id' in data.columns:
        ids = data['Id']
        data = data.drop('Id', axis=1)
    
    # 保存目标变量
    target = None
    if Config.TARGET in data.columns:
        target = data[Config.TARGET]
        data = data.drop(Config.TARGET, axis=1)
    
    # 分离数值和类别特征
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # ===== 数值特征工程 =====
    
    # 1. 处理缺失值 - 用中位数填充
    for col in numeric_features:
        data[col] = data[col].fillna(data[col].median() if len(data[col].median()) > 0 else 0)
    
    # 2. 添加特征组合
    # 房屋总面积
    if 'GrLivArea' in data.columns:
        data['TotalSF'] = data['GrLivArea']
    elif 'GrLivArea' in data.columns:
        data['TotalSF'] = data['GrLivArea']
    
    # 总面积
    if all(col in data.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    # 总 bathrooms
    if all(col in data.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        data['TotalBath'] = data['FullBath'] + 0.5*data['HalfBath'] + data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
    
    # 房屋年龄相关
    if 'YearBuilt' in data.columns:
        data['HouseAge'] = data['YrSold'] - data['YearBuilt']
        data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
    
    # 车库相关
    if 'GarageArea' in data.columns and 'GarageCars' in data.columns:
        data['GarageAreaPerCar'] = data['GarageArea'] / (data['GarageCars'] + 1)
    
    # ===== 类别特征工程 =====
    
    # 3. 处理缺失值 - 用众数填充
    for col in categorical_features:
        data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'None')
    
    # 4. Label Encoding
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # 5. 独热编码 (可选，对于树模型通常不需要)
    # data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    
    return data, target, ids

# ==================== 模型训练 ====================
class HousePriceModel:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = RobustScaler()
        
    def train_kfold(self, X, y, X_test=None):
        """K折交叉验证训练"""
        n_folds = Config.N_FOLDS
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        # 对数变换目标变量 (Kaggle竞赛标准)
        y_log = np.log1p(y)
        
        # 存储预测结果
        oof_xgb = np.zeros(len(X))
        oof_lgb = np.zeros(len(X))
        test_xgb = np.zeros(len(X_test)) if X_test is not None else None
        test_lgb = np.zeros(len(X_test)) if X_test is not None else None
        
        # RMSE 记录
        xgb_scores = []
        lgb_scores = []
        ensemble_scores = []
        
        print("\n" + "="*60)
        print(f"开始 {n_folds} 折交叉验证...")
        print("="*60)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n--- Fold {fold}/{n_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
            y_val_orig = y.iloc[val_idx]
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(**Config.XGB_PARAMS)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_pred_log = xgb_model.predict(X_val)
            xgb_pred = np.expm1(xgb_pred_log)
            xgb_pred = np.clip(xgb_pred, 0, y.max() * 2)
            oof_xgb[val_idx] = xgb_pred
            
            # 计算 RMSE (log scale) - Kaggle评估标准
            xgb_rmse_log = np.sqrt(mean_squared_error(y_val, xgb_pred_log))
            xgb_scores.append(xgb_rmse_log)
            
            # LightGBM
            lgb_model = LGBMRegressor(**Config.LGB_PARAMS)
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.callback.early_stopping(100, verbose=False)]
            )
            lgb_pred_log = lgb_model.predict(X_val)
            lgb_pred = np.expm1(lgb_pred_log)
            lgb_pred = np.clip(lgb_pred, 0, y.max() * 2)
            oof_lgb[val_idx] = lgb_pred
            
            lgb_rmse_log = np.sqrt(mean_squared_error(y_val, lgb_pred_log))
            lgb_scores.append(lgb_rmse_log)
            
            # 集成预测
            ensemble_pred = (xgb_pred + lgb_pred) / 2
            ensemble_rmse_log = np.sqrt(mean_squared_error(y_val, 
                np.log1p(ensemble_pred)))
            ensemble_scores.append(ensemble_rmse_log)
            
            print(f"XGBoost  RMSE (log): {xgb_rmse_log:.5f}")
            print(f"LightGBM RMSE (log): {lgb_rmse_log:.5f}")
            print(f"集成     RMSE (log): {ensemble_rmse_log:.5f}")
            
            # 测试集预测
            if X_test is not None:
                test_xgb += np.expm1(xgb_model.predict(X_test)) / n_folds
                test_lgb += np.expm1(lgb_model.predict(X_test)) / n_folds
        
        # 保存最终模型
        self.xgb_model = xgb.XGBRegressor(**{k:v for k,v in Config.XGB_PARAMS.items() if k != 'early_stopping_rounds'})
        self.xgb_model.fit(X, y_log)
        
        self.lgb_model = LGBMRegressor(**{k:v for k,v in Config.LGB_PARAMS.items()})
        self.lgb_model.fit(X, y_log)
        
        # 汇总结果
        print("\n" + "="*60)
        print("K折交叉验证结果 (RMSE - log scale)")
        print("="*60)
        print(f"XGBoost   CV RMSE: {np.mean(xgb_scores):.5f} (+/- {np.std(xgb_scores):.5f})")
        print(f"LightGBM  CV RMSE: {np.mean(lgb_scores):.5f} (+/- {np.std(lgb_scores):.5f})")
        print(f"集成      CV RMSE: {np.mean(ensemble_scores):.5f} (+/- {np.std(ensemble_scores):.5f})")
        
        # 计算整体 OOF RMSE
        oof_rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
        oof_rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
        oof_ensemble = (oof_xgb + oof_lgb) / 2
        oof_rmse_ensemble = np.sqrt(mean_squared_error(y, oof_ensemble))
        
        print("\n整体 OOF RMSE (原始尺度):")
        print(f"XGBoost:  {oof_rmse_xgb:,.2f}")
        print(f"LightGBM: {oof_rmse_lgb:,.2f}")
        print(f"集成:     {oof_rmse_ensemble:,.2f}")
        
        # 测试集预测
        if X_test is not None:
            test_pred = (test_xgb + test_lgb) / 2
            test_pred = np.clip(test_pred, 0, None)
            return oof_ensemble, test_pred
        
        return oof_ensemble, None
    
    def save_model(self, path):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'xgb': self.xgb_model,
                'lgb': self.lgb_model,
                'scaler': self.scaler
            }, f)
        print(f"\n模型已保存: {path}")

# ==================== 主程序 ====================
def main():
    # 加载数据
    train_df, test_df = load_data()
    
    # 特征工程
    print("\n进行特征工程...")
    X, y, train_ids = feature_engineering(train_df, is_train=True)
    X_test, _, test_ids = feature_engineering(test_df, is_train=False) if test_df is not None else (None, None, None)
    
    print(f"\n特征数量: {X.shape[1]}")
    print(f"训练样本: {X.shape[0]}")
    if X_test is not None:
        print(f"测试样本: {X_test.shape[0]}")
    
    # 标准化
    X_scaled = Config.RANDOM_STATE
    X_scaled = RobustScaler()
    X_train_scaled = X_scaled.fit_transform(X)
    X_test_scaled = X_scaled.transform(X_test) if X_test is not None else None
    
    # 训练模型
    model = HousePriceModel()
    oof_pred, test_pred = model.train_kfold(X_train_scaled, y, X_test_scaled)
    
    # 保存模型
    model.save_model(os.path.join(Config.DATA_PATH, 'model.pkl'))
    
    # 保存 OOF 预测
    oof_df = pd.DataFrame({
        'Id': train_ids if train_ids is not None else range(len(y)),
        'Actual': y.values,
        'Predicted': oof_pred
    })
    oof_df.to_csv(os.path.join(Config.DATA_PATH, 'oof_predictions.csv'), index=False)
    print(f"\nOOF 预测已保存: oof_predictions.csv")
    
    # 生成 Kaggle 提交文件
    if test_df is not None and test_pred is not None:
        submission = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': test_pred
        })
        submission_path = os.path.join(Config.DATA_PATH, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        print(f"\nKaggle 提交文件已生成: submission.csv")
        print(f"\n预测结果预览:")
        print(submission.head(10))
    
    # 保存评估结果
    results = {
        'Model': ['XGBoost+LightGBM Ensemble'],
        'OOF_RMSE': [np.sqrt(mean_squared_error(y, oof_pred))],
        'CV_RMSE_log': [np.mean([
            np.sqrt(mean_squared_error(np.log1p(y.iloc[val_idx]), np.log1p(oof_pred[val_idx])))
            for _, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X)
        ])]
    }
    pd.DataFrame(results).to_csv(os.path.join(Config.DATA_PATH, 'model_results.csv'), index=False)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()
