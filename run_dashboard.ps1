# Cybersecurity Network Threat & Intrusion Profiler SOC Launcher
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING CYBERSECURITY THREAT & INTRUSION PROFILER (SOC ENGINE)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot
python -m streamlit run app.py
