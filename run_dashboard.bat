@echo off
title Cybersecurity Network Threat & Intrusion Profiler SOC
echo ===================================================================
echo   LAUNCHING CYBERSECURITY THREAT & INTRUSION PROFILER (SOC ENGINE)
echo ===================================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
