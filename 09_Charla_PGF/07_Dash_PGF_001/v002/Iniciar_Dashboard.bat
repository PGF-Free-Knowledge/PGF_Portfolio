@echo off
cd /d "%~dp0"
echo Iniciando Dashboard PGF...
python -m streamlit run app\app.py
pause
