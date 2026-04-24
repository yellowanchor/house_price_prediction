# House Price Prediction - Kaggle Competition

房价预测模型 - Kaggle House Prices 竞赛标准实现

## 竞赛信息

- **竞赛名称**: House Prices - Advanced Regression Techniques
- **评估指标**: RMSE (Root Mean Squared Error) on log-transformed target
- **Kaggle链接**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## 项目结构

```
house_price_prediction/
├── train.py              # 主训练脚本
├── requirements.txt      # 依赖包
├── README.md            # 项目说明
├── model.pkl            # 训练好的模型
├── oof_predictions.csv  # 交叉验证预测结果
├── submission.csv       # Kaggle提交文件
└── model_results.csv    # 模型评估结果
```

## 模型架构

### 集成模型
- **XGBoost**: 梯度提升决策树
- **LightGBM**: 基于直方图的梯度提升
- **集成策略**: 简单平均

### 关键参数
- K-Fold: 5折交叉验证
- Early Stopping: 100轮
- Learning Rate: 0.02
- Estimators: 2000
- Max Depth: 4

## 特征工程

1. **缺失值处理**: 数值用中位数，类别用众数
2. **特征组合**:
   - TotalSF: 总面积 (地下室 + 1层 + 2层)
   - TotalBath: 总卫生间数
   - HouseAge/RemodAge: 房龄
   - GarageAreaPerCar: 每车位车库面积
3. **目标变量**: log1p 变换（竞赛标准）

## 评估结果

| 指标 | 值 |
|------|-----|
| CV RMSE (log scale) | ~0.12-0.14 |
| OOF RMSE (原始尺度) | 模型相关 |

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 Kaggle 下载的 `train.csv` 和 `test.csv` 放在项目根目录。

### 3. 训练模型

```bash
python train.py
```

### 4. 生成提交文件

运行后会自动生成 `submission.csv` 文件，直接上传到 Kaggle 即可。

## 文件说明

### train.py
- `load_data()`: 加载训练集和测试集
- `feature_engineering()`: 特征工程
- `HousePriceModel`: 模型训练类
- `main()`: 主程序入口

### submission.csv
Kaggle 提交格式:
```
Id,SalePrice
1461,208500.00
...
```

## 依赖包

- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- xgboost>=2.0.0
- lightgbm>=4.0.0

## 参考

- Kaggle 竞赛讨论区高分方案
- XGBoost 官方文档
- LightGBM 官方文档
