"""
房价预测模型 - 完整版
特征工程 + 多模型对比 + SHAP可解释性 + Flask Web应用
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    DATA_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(DATA_PATH, 'models')
    TARGET = 'SalePrice'
    RANDOM_STATE = 42
    N_FOLDS = 5
    
    # 数值特征
    NUMERIC_FEATURES = [
        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
        'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    
    # 类别特征
    CATEGORICAL_FEATURES = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
        'SaleCondition'
    ]

# ==================== 数据加载 ====================
def load_data():
    """加载训练集和测试集"""
    print("="*60)
    print("房价预测模型 - 完整版")
    print("="*60)
    
    train_path = os.path.join(Config.DATA_PATH, 'train.csv')
    test_path = os.path.join(Config.DATA_PATH, 'test.csv')
    
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        print(f"\n本地数据 - 训练集: {train_df.shape}")
        return train_df, test_df
    
    # 使用 OpenML 数据集
    print("\n使用 OpenML 数据集...")
    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    return housing.frame, None

# ==================== 特征工程 ====================
class FeatureEngineering:
    """完整的特征工程流程"""
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.feature_names = []
        self.scaler = None
        
    def identify_features(self, df):
        """识别数值和类别特征"""
        # 获取数据中实际存在的特征
        numeric = []
        categorical = []
        
        for col in Config.NUMERIC_FEATURES:
            if col in df.columns:
                numeric.append(col)
        
        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                categorical.append(col)
        
        self.numeric_features = numeric
        self.categorical_features = categorical
        print(f"\n数值特征: {len(numeric)} 个")
        print(f"类别特征: {len(categorical)} 个")
        
    def fit(self, X):
        """训练特征预处理器"""
        print("\n特征工程 - 训练预处理器...")
        
        # 数值特征: 缺失值用中位数填充 + 标准化
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 类别特征: 缺失值用众数填充 + 独热编码
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 组合预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        # 训练预处理器
        self.preprocessor.fit(X)
        
        # 获取特征名称
        self._get_feature_names()
        print(f"总特征数量: {len(self.feature_names)}")
        
    def _get_feature_names(self):
        """获取处理后的特征名称"""
        self.feature_names = []
        
        # 数值特征名称
        self.feature_names.extend(self.numeric_features)
        
        # 独热编码后的类别特征名称
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        self.feature_names.extend(cat_feature_names.tolist())
        
    def transform(self, X):
        """转换数据"""
        X_transformed = self.preprocessor.transform(X)
        return X_transformed
    
    def fit_transform(self, X):
        """训练并转换"""
        self.fit(X)
        return self.transform(X)
    
    def save(self, path):
        """保存预处理器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'feature_names': self.feature_names
            }, f)
        print(f"预处理器已保存: {path}")
    
    def load(self, path):
        """加载预处理器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.preprocessor = data['preprocessor']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.feature_names = data['feature_names']

# ==================== 模型训练 ====================
class ModelTrainer:
    """多模型对比训练"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self):
        """定义要对比的模型"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=Config.RANDOM_STATE),
            'Lasso Regression': Lasso(alpha=1.0, random_state=Config.RANDOM_STATE),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=Config.RANDOM_STATE),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=Config.RANDOM_STATE),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=Config.RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=Config.RANDOM_STATE),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=Config.RANDOM_STATE, n_jobs=-1,
                early_stopping_rounds=50
            )
        }
    
    def train_all(self, X, y, X_val=None, y_val=None):
        """训练所有模型并对比"""
        print("\n" + "="*60)
        print("多模型对比训练")
        print("="*60)
        
        # 对数变换目标变量
        y_log = np.log1p(y)
        
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        self.models = self.get_models()
        
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            
            # K折交叉验证
            cv_scores = []
            cv_r2_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val_fold = y_log.iloc[train_idx], y_log.iloc[val_idx]
                
                # 克隆模型
                from sklearn.base import clone
                model_clone = clone(model)
                
                # 训练
                if 'XGBoost' in name:
                    model_clone.fit(X_train, y_train, eval_set=[(X_val, y_val_fold)], verbose=False)
                else:
                    model_clone.fit(X_train, y_train)
                
                # 预测
                pred_log = model_clone.predict(X_val)
                pred = np.expm1(pred_log)
                y_val_orig = np.expm1(y_val_fold)
                
                # 计算 RMSE
                rmse = np.sqrt(mean_squared_error(y_val_orig, pred))
                r2 = r2_score(y_val_orig, pred)
                cv_scores.append(rmse)
                cv_r2_scores.append(r2)
            
            # 存储结果
            self.results[name] = {
                'cv_rmse_mean': np.mean(cv_scores),
                'cv_rmse_std': np.std(cv_scores),
                'cv_r2_mean': np.mean(cv_r2_scores),
                'model': model
            }
            
            print(f"  RMSE: {np.mean(cv_scores):,.2f} (+/- {np.std(cv_scores):,.2f})")
            print(f"  R²: {np.mean(cv_r2_scores):.4f}")
        
        # 找出最佳模型
        self._find_best_model()
        
        return self.results
    
    def _find_best_model(self):
        """找出最佳模型"""
        best_rmse = float('inf')
        best_name = None
        
        for name, result in self.results.items():
            if result['cv_rmse_mean'] < best_rmse:
                best_rmse = result['cv_rmse_mean']
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        print(f"\n最佳模型: {best_name} (RMSE: {best_rmse:,.2f})")
    
    def print_results(self):
        """打印结果对比"""
        print("\n" + "="*60)
        print("模型对比结果汇总")
        print("="*60)
        print(f"{'模型':<25} {'RMSE':<15} {'R²':<10}")
        print("-"*50)
        
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['cv_rmse_mean'])
        
        for name, result in sorted_results:
            marker = "⭐" if name == self.best_model_name else "  "
            print(f"{marker}{name:<23} {result['cv_rmse_mean']:>12,.2f}  {result['cv_r2_mean']:.4f}")
    
    def save_results(self, path):
        """保存训练结果"""
        # 移除模型对象，只保存指标
        results_to_save = {}
        for name, result in self.results.items():
            results_to_save[name] = {
                'cv_rmse_mean': float(result['cv_rmse_mean']),
                'cv_rmse_std': float(result['cv_rmse_std']),
                'cv_r2_mean': float(result['cv_r2_mean'])
            }
        
        with open(path, 'w') as f:
            json.dump({
                'results': results_to_save,
                'best_model': self.best_model_name
            }, f, indent=2)
        print(f"\n结果已保存: {path}")
    
    def save_models(self, path):
        """保存训练好的模型"""
        os.makedirs(path, exist_ok=True)
        
        # 保存最佳模型
        import joblib
        best_model_path = os.path.join(path, 'best_model.pkl')
        joblib.dump(self.best_model, best_model_path)
        print(f"最佳模型已保存: {best_model_path}")
        
        # 保存所有模型
        all_models_path = os.path.join(path, 'all_models.pkl')
        with open(all_models_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"所有模型已保存: {all_models_path}")

# ==================== SHAP分析 ====================
class SHAPExplainer:
    """SHAP可解释性分析"""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
    def create_explainer(self, model, X_sample, feature_names):
        """创建SHAP解释器"""
        print("\n" + "="*60)
        print("SHAP 可解释性分析")
        print("="*60)
        
        self.feature_names = feature_names
        
        try:
            import shap
            
            # XGBoost模型
            if 'XGBRegressor' in str(type(model)):
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(X_sample)
            
            # 其他模型使用KernelExplainer
            else:
                def predict_wrapper(x):
                    return model.predict(x)
                self.explainer = shap.KernelExplainer(predict_wrapper, X_sample[:100])
                self.shap_values = self.explainer.shap_values(X_sample[:100])
            
            print("SHAP 分析完成!")
            return True
            
        except ImportError:
            print("SHAP 未安装，跳过可解释性分析")
            print("安装命令: pip install shap")
            return False
    
    def get_feature_importance(self, X_sample=None):
        """获取特征重要性"""
        if self.shap_values is None:
            return None
        
        import shap
        
        # 计算平均绝对SHAP值
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 重要特征:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def save_analysis(self, path, X_sample=None):
        """保存SHAP分析结果"""
        if self.shap_values is None:
            return
        
        import shap
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存特征重要性
        importance_df = self.get_feature_importance()
        importance_df.to_csv(os.path.join(path, 'feature_importance.csv'), index=False)
        
        # 保存SHAP摘要图
        plt = shap.summary_plot(self.shap_values, X_sample, feature_names=self.feature_names, show=False)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(path, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP分析结果已保存: {path}")

# ==================== 主程序 ====================
def main():
    # 创建模型目录
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 分离特征和目标
    X = train_df.drop([Config.TARGET, 'Id'], axis=1) if 'Id' in train_df.columns else train_df.drop(Config.TARGET, axis=1)
    y = train_df[Config.TARGET]
    
    print(f"\n特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 特征工程
    fe = FeatureEngineering()
    fe.identify_features(X)
    X_transformed = fe.fit_transform(X)
    fe.save(os.path.join(Config.MODEL_PATH, 'preprocessor.pkl'))
    
    # 训练模型
    trainer = ModelTrainer()
    trainer.train_all(X_transformed, y)
    trainer.print_results()
    trainer.save_results(os.path.join(Config.MODEL_PATH, 'results.json'))
    trainer.save_models(Config.MODEL_PATH)
    
    # SHAP分析
    shap_analyzer = SHAPExplainer()
    if shap_analyzer.create_explainer(trainer.best_model, X_transformed[:100], fe.feature_names):
        shap_analyzer.save_analysis(os.path.join(Config.MODEL_PATH, 'shap'))
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()
