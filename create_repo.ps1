# 创建 GitHub 仓库并推送
# 需要先安装 GitHub CLI: https://cli.github.com/

$RepoName = "house_price_prediction"
$User = "yellowanchor"
$Description = "House Price Prediction using XGBoost + LightGBM with K-Fold Cross Validation"

Write-Host "检查 GitHub CLI..."
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "GitHub CLI 未安装，请先安装: https://cli.github.com/" -ForegroundColor Red
    Write-Host "或者手动创建仓库后执行: git push -u origin main" -ForegroundColor Yellow
    exit 1
}

Write-Host "登录状态检查..."
gh auth status

Write-Host "创建仓库..."
gh repo create $RepoName --public --description $Description --source . --push

if ($LASTEXITCODE -eq 0) {
    Write-Host "仓库创建并推送成功!" -ForegroundColor Green
} else {
    Write-Host "创建失败，请手动创建仓库后执行: git push -u origin main" -ForegroundColor Yellow
}
