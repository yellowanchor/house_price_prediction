# House Price Prediction System

基于机器学习的房价预测系统，支持交互式预测和SHAP可解释性分析。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)

## Features

- Complete feature engineering pipeline
- Multiple model comparison (XGBoost, Random Forest, Gradient Boosting, etc.)
- SHAP interpretability analysis
- Interactive Flask Web application
- Bilingual interface (Chinese/English)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download the Kaggle House Prices dataset:
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- Place `train.csv` in the project directory

### 3. Train Model

```bash
python train.py
```

### 4. Start Web Application

```bash
python app.py
```

Visit http://127.0.0.1:5000

## Project Structure

```
house_price_prediction/
├── train.py                 # Training script
├── app.py                   # Flask web application
├── templates/
│   └── index.html           # Frontend page
├── models/                  # Trained models
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── results.json
├── data/
│   └── train.csv            # Training data
├── requirements.txt
└── README.md
```

## Model Performance

| Model | RMSE | R² |
|-------|------|-----|
| XGBoost | 28,191 | 0.8642 |
| Gradient Boosting | 30,583 | 0.8294 |
| Random Forest | 30,960 | 0.8349 |
| Decision Tree | 42,045 | 0.6896 |
| Ridge Regression | 55,253 | -0.2511 |
| Linear Regression | 61,117 | -0.7047 |

## Top Features (SHAP)

1. **OverallQual** - Overall quality rating (1-10)
2. **GrLivArea** - Above ground living area
3. **GarageFinish_Unf** - Unfinished garage
4. **TotalBsmtSF** - Basement area
5. **YearBuilt** - Year of construction

## API

### Predict Endpoint

```bash
POST /predict
Content-Type: application/json

{
    "GrLivArea": 139,
    "OverallQual": 5,
    "YearBuilt": 2000,
    "TotalBsmtSF": 74,
    "GarageCars": 2,
    "FullBath": 1,
    "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr",
    "BldgType": "1Fam",
    "KitchenQual": "Gd"
}
```

## Tech Stack

- Python 3.8+
- scikit-learn - Feature engineering, model training
- XGBoost - Gradient boosting
- SHAP - Model interpretability
- Flask - Web framework
- Pandas/NumPy - Data processing
- Matplotlib - Visualization

## License

MIT License
