"""
房价预测模型 - 完整版
特征工程 + 多模型对比 + SHAP可解释性 + Flask Web应用
官方Kaggle数据获取
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

# ==================== 数据加载（官方Kaggle方法）====================
def load_data():
    """使用kagglehub官方方法加载数据"""
    print("="*60)
    print("房价预测模型 - 完整版")
    print("="*60)
    
    train_path = os.path.join(Config.DATA_PATH, 'train.csv')
    test_path = os.path.join(Config.DATA_PATH, 'test.csv')
    
    # 如果本地已有数据，直接加载
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
        print(f"\n加载本地数据 - 训练集: {train_df.shape}")
        if test_df is not None:
            print(f"测试集: {test_df.shape}")
        return train_df, test_df
    
    # 使用 kagglehub 官方方法下载
    print("\n使用 KaggleHub 官方方法下载数据...")
    
    try:
        import kagglehub
        
        # 方法1: 下载竞赛数据
        print("下载 Kaggle House Prices 竞赛数据...")
        path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
        
        # 查找下载的文件
        files = os.listdir(path)
        print(f"下载完成，文件列表: {files}")
        
        # 加载数据
        train_file = os.path.join(path, 'train.csv')
        test_file = os.path.join(path, 'test.csv')
        
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file) if os.path.exists(test_file) else None
            print(f"\n训练集: {train_df.shape}")
            if test_df is not None:
                print(f"测试集: {test_df.shape}")
            return train_df, test_df
        
        raise FileNotFoundError("train.csv not found in downloaded files")
        
    except ImportError:
        print("kagglehub 未安装，安装中...")
        os.system('pip install kagglehub')
        return load_data()
        
    except Exception as e:
        print(f"KaggleHub 下载失败: {e}")
        print("\n" + "="*60)
        print("请确保已完成以下配置：")
        print("1. 安装 kaggle 包: pip install kagglehub")
        print("2. 配置 Kaggle 认证:")
        print("   - 访问 https://www.kaggle.com/account")
        print("   - 点击 'Create New API Token' 下载 kaggle.json")
        print("   - Windows: 放置到 C:\\Users\\<用户名>\\.kaggle\\kaggle.json")
        print("   - Linux/Mac: 放置到 ~/.kaggle/kaggle.json")
        print("="*60)
        
        # 备用方案：使用 OpenML
        print("\n使用备用数据源 OpenML...")
        from sklearn.datasets import fetch_openml
        housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
        return housing.frame, None

# ==================== 特征工程 ====================
class FeatureEngineering:
    """完整的特征工程流程"""
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.feature_names = []
        
    def identify_features(self, df):
        """识别数值和类别特征"""
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除ID和目标变量
        exclude_cols = ['Id', Config.TARGET]
        self.numeric_features = [c for c in self.numeric_features if c not in exclude_cols]
        
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\n数值特征: {len(self.numeric_features)} 个")
        print(f"类别特征: {len(self.categorical_features)} 个")
        
    def fit(self, X):
        """训练特征预处理器"""
        print("\n特征工程 - 训练预处理器...")
        
        # 数值特征: 缺失值用中位数填充 + 标准化
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 类别特征: 缺失值用众数填充 + 独热编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 组合预处理器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor.fit(X)
        self._get_feature_names()
        print(f"总特征数量: {len(self.feature_names)}")
        
    def _get_feature_names(self):
        """获取处理后的特征名称"""
        self.feature_names = list(self.numeric_features)
        
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        self.feature_names.extend(cat_feature_names.tolist())
        
    def transform(self, X):
        """转换数据"""
        return self.preprocessor.transform(X)
    
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
    
    def train_all(self, X, y):
        """训练所有模型并对比"""
        print("\n" + "="*60)
        print("多模型对比训练")
        print("="*60)
        
        y_log = np.log1p(y)
        
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        self.models = self.get_models()
        
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            
            cv_scores = []
            cv_r2_scores = []
            
            from sklearn.base import clone
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val_fold = y_log.iloc[train_idx], y_log.iloc[val_idx]
                
                model_clone = clone(model)
                
                if 'XGBoost' in name:
                    model_clone.fit(X_train, y_train, eval_set=[(X_val, y_val_fold)], verbose=False)
                else:
                    model_clone.fit(X_train, y_train)
                
                pred_log = model_clone.predict(X_val)
                pred = np.expm1(pred_log)
                y_val_orig = np.expm1(y_val_fold)
                
                rmse = np.sqrt(mean_squared_error(y_val_orig, pred))
                r2 = r2_score(y_val_orig, pred)
                cv_scores.append(rmse)
                cv_r2_scores.append(r2)
            
            self.results[name] = {
                'cv_rmse_mean': np.mean(cv_scores),
                'cv_rmse_std': np.std(cv_scores),
                'cv_r2_mean': np.mean(cv_r2_scores),
                'model': model
            }
            
            print(f"  RMSE: {np.mean(cv_scores):,.2f} (+/- {np.std(cv_scores):,.2f})")
            print(f"  R²: {np.mean(cv_r2_scores):.4f}")
        
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
        
        import joblib
        best_model_path = os.path.join(path, 'best_model.pkl')
        joblib.dump(self.best_model, best_model_path)
        print(f"最佳模型已保存: {best_model_path}")

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
            
            if 'XGBRegressor' in str(type(model)):
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(X_sample)
                print("SHAP 分析完成!")
                return True
            
        except ImportError:
            print("SHAP 未安装，跳过可解释性分析")
            print("安装命令: pip install shap")
        
        return False
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.shap_values is None:
            return None
        
        import shap
        
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 重要特征:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df

# ==================== 主程序 ====================
def main():
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
        shap_analyzer.get_feature_importance()
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print("\n下一步:")
    print("1. pip install flask shap")
    print("2. python app.py")
    print("3. 访问 http://127.0.0.1:5000")

if __name__ == '__main__':
    main()
