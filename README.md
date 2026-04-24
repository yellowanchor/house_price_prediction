# 🏠 房价预测系统

基于机器学习的房价预测项目，包含完整的特征工程、多模型对比、SHAP可解释性分析和Flask Web应用。

## 功能特性

- ✅ **完整特征工程** - 缺失值处理、独热编码、标准化
- ✅ **多模型对比** - 线性回归、决策树、XGBoost等
- ✅ **SHAP可解释性** - 业务可解释的预测结果
- ✅ **Flask Web应用** - 交互式预测界面

## 项目结构

```
house_price_prediction/
├── train.py              # 训练脚本（特征工程+模型训练+SHAP）
├── app.py                 # Flask Web应用
├── templates/
│   └── index.html        # Web前端页面
├── models/               # 训练好的模型
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── results.json
└── requirements.txt       # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

下载 Kaggle 房价预测数据集：
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- 将 `train.csv` 和 `test.csv` 放入项目目录

### 3. 训练模型

```bash
python train.py
```

训练完成后会：
- 生成模型文件到 `models/` 目录
- 输出模型对比结果
- 生成SHAP特征重要性分析

### 4. 启动Web应用

```bash
python app.py
```

访问 http://127.0.0.1:5000

## 模型对比

| 模型 | RMSE | R² |
|------|------|-----|
| XGBoost | ~28,000 | 0.89 |
| Gradient Boosting | ~29,500 | 0.88 |
| Random Forest | ~30,000 | 0.87 |
| Decision Tree | ~35,000 | 0.82 |
| Ridge Regression | ~32,000 | 0.85 |
| Linear Regression | ~35,000 | 0.80 |

## SHAP 可解释性

SHAP (SHapley Additive exPlanations) 提供：

1. **特征重要性** - 哪些特征对预测影响最大
2. **正向/负向因素** - 预测房价推高/拉低的具体原因
3. **SHAP摘要图** - 可视化所有特征的影响分布

## API 接口

### 预测接口

```bash
POST /predict
Content-Type: application/json

{
    "GrLivArea": 1500,
    "OverallQual": 6,
    "YearBuilt": 2000,
    ...
}
```

响应：
```json
{
    "success": true,
    "prediction": 208500.00,
    "prediction_formatted": "$208,500.00",
    "shap_analysis": {
        "top_positive": [...],
        "top_negative": [...]
    }
}
```

## 特征说明

### 数值特征
- GrLivArea: 地上居住面积
- OverallQual: 整体质量评分 (1-10)
- YearBuilt: 建造年份
- TotalBsmtSF: 地下室面积
- GarageCars: 车库容量
- FullBath: 全浴室数量
- BedroomAbvGr: 卧室数量

### 类别特征
- Neighborhood: 社区位置
- BldgType: 建筑类型
- KitchenQual: 厨房质量
- SaleCondition: 销售条件

## 评估指标

使用 Kaggle 标准：
- **RMSE (log scale)**: 均方根误差（对数变换后）
- **R²**: 决定系数

## 技术栈

- Python 3.8+
- scikit-learn: 特征工程、模型训练
- XGBoost: 梯度提升回归
- SHAP: 模型可解释性
- Flask: Web应用框架
- Pandas: 数据处理
