@echo off
chcp 65001 > nul
title ForexAI Vision - 5-6時間後 Pips予測アプリ

echo ========================================================
echo   ForexAI Vision: USD/JPY & JPY/USD 5-6h Pips Predictor
echo ========================================================
echo.
echo アプリケーションを起動しています...
echo ブラウザが自動的に開きます。しばらくお待ちください。
echo.
echo 停止する場合は、このウィンドウで Ctrl + C を押してください。
echo.

python -m streamlit run app.py --server.headless=false

pause
