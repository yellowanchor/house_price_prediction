#!/bin/bash
# 创建新仓库并推送

REPO_NAME="house_price_prediction"
GITHUB_USER="yellowanchor"
DESCRIPTION="House Price Prediction using XGBoost + LightGBM with K-Fold Cross Validation"

# 创建 GitHub 仓库
echo "创建 GitHub 仓库..."
gh repo create $REPO_NAME --public --description "$DESCRIPTION" --source . --push

echo "完成!"
