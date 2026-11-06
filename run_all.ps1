# 一键回归：生成数据集 → 评测 → 单元测试 → 生成插图 → 真实图检测
# 用法： pwsh ./run_all.ps1        （需已安装 requirements.txt 中的依赖）

$ErrorActionPreference = "Stop"
$py = "python"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "== 1/5 生成合成数据集 ==" -ForegroundColor Cyan
& $py tools/make_dataset.py --out dataset --per-condition 40

Write-Host "== 2/5 量化评测 ==" -ForegroundColor Cyan
& $py tools/benchmark.py --dataset dataset --out docs

Write-Host "== 3/5 单元测试 ==" -ForegroundColor Cyan
& $py -m unittest discover -s tests

Write-Host "== 4/5 生成报告插图 ==" -ForegroundColor Cyan
& $py tools/make_figures.py

Write-Host "== 5/5 真实测试图检测 ==" -ForegroundColor Cyan
& $py main.py --dir data --save-dir result --json docs/real_images_result.json --quiet

Write-Host "`n全部完成：docs/benchmark_table.md、docs/figures/、result/" -ForegroundColor Green
