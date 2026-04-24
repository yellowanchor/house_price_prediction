# Kaggle 提交指南

## 步骤 1: 下载数据

1. 访问 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
2. 下载 `train.csv` 和 `test.csv`
3. 将文件放入项目目录

## 步骤 2: 运行训练

```bash
pip install -r requirements.txt
python train.py
```

## 步骤 3: 提交结果

1. 打开 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit
2. 点击 "Submit Predictions"
3. 上传 `submission.csv` 文件
4. 确认格式正确
5. 提交

## 提交文件格式

```csv
Id,SalePrice
1461,208500.0
1462,171500.0
...
```

## 评估标准

Kaggle 使用 **RMSE on log(SalePrice)** 进行评估：

```
RMSE = sqrt(mean((log(pred) - log(actual))^2))
```

这就是为什么我们使用 `np.log1p()` 变换目标变量。

## 提升分数的建议

1. **更多特征工程**:
   - 地区特征编码
   - 时间特征
   - 特征交叉

2. **模型调优**:
   - Optuna/GridSearchCV 超参数搜索
   - 更多模型集成 (CatBoost, Ridge, ElasticNet)

3. **数据清洗**:
   - 处理异常值
   - 特征选择
