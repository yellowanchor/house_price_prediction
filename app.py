"""
房价预测 Web 应用
Flask + 交互式预测 + SHAP可视化
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'house-price-prediction-secret-key'
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'models')

# 全局变量
model = None
preprocessor = None
feature_names = None
numeric_features = None
categorical_features = None
shap_explainer = None
shap_values = None
X_sample = None

# ==================== 模型加载 ====================
def load_models():
    """加载训练好的模型"""
    global model, preprocessor, feature_names, numeric_features, categorical_features
    
    import joblib
    
    model_path = os.path.join(app.config['MODEL_PATH'], 'best_model.pkl')
    preprocessor_path = os.path.join(app.config['MODEL_PATH'], 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("模型文件不存在，请先运行 train.py")
        return False
    
    # 使用joblib加载模型
    model = joblib.load(model_path)
    
    # 使用pickle加载预处理器
    with open(preprocessor_path, 'rb') as f:
        data = pickle.load(f)
        preprocessor = data['preprocessor']
        feature_names = data['feature_names']
        numeric_features = data['numeric_features']
        categorical_features = data['categorical_features']
    
    print(f"模型加载成功: {model}")
    return True

# ==================== 路由 ====================
@app.route('/')
def index():
    """主页"""
    # 加载模型对比结果
    results_path = os.path.join(app.config['MODEL_PATH'], 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    # 将results传递给模板
    return render_template('index.html', results=results)

@app.route('/api/results')
def api_results():
    """API: 获取模型对比结果"""
    results_path = os.path.join(app.config['MODEL_PATH'], 'results.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    global shap_explainer, shap_values, X_sample
    
    try:
        # 获取输入数据
        data = request.get_json()
        
        # 构建特征DataFrame
        features = {}
        for col in numeric_features:
            features[col] = float(data.get(col, 0))
        for col in categorical_features:
            features[col] = data.get(col, 'Unknown')
        
        df = pd.DataFrame([features])
        
        # 确保列顺序正确
        all_cols = numeric_features + categorical_features
        df = df.reindex(columns=[c for c in all_cols if c in df.columns or c in features], fill_value=0)
        
        # 转换类别特征（使用训练时的编码）
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # 预处理
        X = preprocessor.transform(df)
        
        # 预测
        pred_log = model.predict(X)
        pred = float(np.expm1(pred_log)[0])  # 转换为Python float
        
        result = {
            'success': True,
            'prediction': round(pred, 2),
            'prediction_formatted': f"${pred:,.2f}"
        }
        
        # SHAP分析
        try:
            if shap_explainer is None:
                # 创建SHAP解释器
                if hasattr(model, 'predict'):
                    X_sample = preprocessor.transform(train_df_sample())
                    shap_explainer = shap.TreeExplainer(model)
                    shap_values = shap_explainer.shap_values(X_sample)
            
            # 获取当前预测的SHAP值
            current_shap = shap_explainer.shap_values(X)[0]
            
            # 获取Top 5正向和负向影响因素
            feature_impact = list(zip(feature_names, current_shap))
            feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            
            result['shap_analysis'] = {
                'top_positive': [
                    {'feature': f, 'value': round(v, 4), 'impact': 'positive'}
                    for f, v in feature_impact[:5] if v > 0
                ],
                'top_negative': [
                    {'feature': f, 'value': round(v, 4), 'impact': 'negative'}
                    for f, v in feature_impact[:5] if v < 0
                ]
            }
            
        except Exception as e:
            result['shap_error'] = str(e)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/shap_plot')
def shap_plot():
    """SHAP可视化"""
    global shap_explainer, shap_values, X_sample
    
    try:
        if X_sample is None:
            X_sample = preprocessor.transform(train_df_sample())
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_sample)
        
        # 生成SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        # 转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{image_base64}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/feature_importance')
def feature_importance():
    """特征重要性"""
    global shap_values, X_sample
    
    try:
        if shap_values is None:
            X_sample = preprocessor.transform(train_df_sample())
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_sample)
        
        # 计算特征重要性
        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [float(x) for x in importance]  # 转换为Python float
        }).sort_values('importance', ascending=False).head(20)
        
        return jsonify({
            'success': True,
            'data': importance_df.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def train_df_sample():
    """返回样本数据用于SHAP分析"""
    data = {
        'LotFrontage': [65], 'LotArea': [8450], 'OverallQual': [6], 'OverallCond': [5],
        'YearBuilt': [2003], 'YearRemodAdd': [2003], 'MasVnrArea': [196], 'BsmtFinSF1': [706],
        'BsmtFinSF2': [0], 'BsmtUnfSF': [150], 'TotalBsmtSF': [856], '1stFlrSF': [856],
        '2ndFlrSF': [854], 'LowQualFinSF': [0], 'GrLivArea': [1710], 'BsmtFullBath': [1],
        'BsmtHalfBath': [0], 'FullBath': [2], 'HalfBath': [1], 'BedroomAbvGr': [3],
        'KitchenAbvGr': [1], 'TotRmsAbvGrd': [7], 'Fireplaces': [1], 'GarageYrBlt': [2003],
        'GarageCars': [2], 'GarageArea': [548], 'WoodDeckSF': [0], 'OpenPorchSF': [61],
        'EnclosedPorch': [0], '3SsnPorch': [0], 'ScreenPorch': [0], 'PoolArea': [0],
        'MiscVal': [0], 'MoSold': [2], 'YrSold': [2008],
        'MSZoning': ['RL'], 'Street': ['Pave'], 'LotShape': ['Reg'], 'LandContour': ['Lvl'],
        'Utilities': ['AllPub'], 'LotConfig': ['Inside'], 'LandSlope': ['Gtl'],
        'Neighborhood': ['CollgCr'], 'Condition1': ['Norm'], 'Condition2': ['Norm'],
        'BldgType': ['1Fam'], 'HouseStyle': ['2Story'], 'RoofStyle': ['Gable'],
        'RoofMatl': ['CompShg'], 'Exterior1st': ['VinylSd'], 'Exterior2nd': ['VinylSd'],
        'MasVnrType': ['None'], 'ExterQual': ['Gd'], 'ExterCond': ['TA'],
        'Foundation': ['Poured'], 'BsmtQual': ['Gd'], 'BsmtCond': ['TA'],
        'BsmtExposure': ['No'], 'BsmtFinType1': ['GLQ'], 'BsmtFinType2': ['Unf'],
        'Heating': ['GasA'], 'HeatingQC': ['Ex'], 'CentralAir': ['Y'], 'Electrical': ['SBrkr'],
        'KitchenQual': ['Gd'], 'Functional': ['Typ'], 'FireplaceQu': ['Gd'],
        'GarageType': ['Attchd'], 'GarageFinish': ['Fin'], 'GarageQual': ['TA'],
        'GarageCond': ['TA'], 'PavedDrive': ['Y'], 'PoolQC': ['None'],
        'Fence': ['None'], 'MiscFeature': ['None'], 'SaleType': ['WD'],
        'SaleCondition': ['Normal']
    }
    return pd.DataFrame(data)

# ==================== 启动 ====================
if __name__ == '__main__':
    if load_models():
        print("\n" + "="*60)
        print("房价预测 Web 应用")
        print("="*60)
        print("访问地址: http://127.0.0.1:5000")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n错误: 无法加载模型")
        print("请先运行: python train.py")
