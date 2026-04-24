# 房价预测项目

基于机器学习的房价预测模型，使用 XGBoost 和 MLP（多层感知机）进行预测，通过 K 折交叉验证评估模型性能。

## 项目结构

```
house_price_prediction/
├── train.py              # 训练脚本
├── requirements.txt      # 依赖包
├── model_results.csv     # 模型评估结果
└── README.md            # 项目说明
```

## 功能特性

- 使用 KaggleHub 获取房价预测数据集
- XGBoost 梯度提升回归模型
- MLP 多层感知机神经网络
- 5 折交叉验证
- RMSE（均方根误差）评估指标
- 模型集成预测

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python train.py
```

## 模型说明

### XGBoost
- 强大的梯度提升算法
- 自动处理缺失值
- 支持并行训练

### MLP (多层感知机)
- 深度神经网络
- ReLU 激活函数
- Adam 优化器
- 早停策略防止过拟合

### 集成方法
- XGBoost 和 MLP 预测值简单平均
- 结合两种模型的优势

## 评估指标

- **RMSE (Root Mean Square Error)**: 均方根误差，值越小表示模型预测越准确

## 许可证

MIT License
