@echo off
echo 🚀 Iniciando Sistema MT5 Trading...
echo.

REM Verificar se MT5 está executando
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✅ MT5 já está executando
) else (
    echo ⚠️ MT5 não está executando
    echo Por favor, inicie o MetaTrader 5 manualmente
    pause
    exit /b 1
)

echo.
echo 🔄 Iniciando sistema de trading...
python mt5_main.py --mode gui

pause
