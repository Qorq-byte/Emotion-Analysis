$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$activate = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "未找到虚拟环境。请先运行: python -m venv .venv"
}

& $venvPython --version
if ($LASTEXITCODE -ne 0) {
    throw "虚拟环境指向无效的 Python。请重建 .venv 后运行: python -m pip install -r requirements.txt"
}

. $activate
Write-Host '虚拟环境已激活。' -ForegroundColor Green
Write-Host '训练: python src/train.py | 评估: python src/evaluate.py | 预测: python src/predict.py' -ForegroundColor Cyan
