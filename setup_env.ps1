# 便捷激活虚拟环境脚本
# 运行方法：在PowerShell中输入 .\setup_env.ps1

Write-Host "正在激活项目虚拟环境..." -ForegroundColor Green

# 激活虚拟环境
.venv\Scripts\Activate.ps1

Write-Host "虚拟环境已激活!" -ForegroundColor Green
Write-Host ""
Write-Host "首次使用请运行: pip install -r requirements.txt" -ForegroundColor Yellow
Write-Host ""
Write-Host "运行程序示例: python src/process.py" -ForegroundColor Cyan
