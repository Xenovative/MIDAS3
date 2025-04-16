@echo off
REM Start LLM Chat Interface Platform

REM 1. Check if Conda environment exists
call conda env list | find "midas3-llm" > nul
if %errorlevel% neq 0 (
    echo Creating Conda environment...
    conda env create -f environment.yml
    if %errorlevel% neq 0 (
        echo Error creating conda environment
        pause
        exit /b
    )
)

REM 2. Activate Conda environment
call conda activate midas3-llm
if %errorlevel% neq 0 (
    echo Error activating conda environment
    pause
    exit /b
)

REM 3. Start Ollama service if not running
echo Checking Ollama status...
tasklist | find "ollama" > nul
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    start "" "ollama" serve
    timeout /t 10 > nul  # Wait longer for Ollama to initialize
    
    REM Verify Ollama started
    tasklist | find "ollama" > nul
    if %errorlevel% neq 0 (
        echo Failed to start Ollama
        pause
        exit /b
    )
)

REM 4. Start Flask application in a new console
echo Starting Flask server...
start "FlaskApp" cmd /k "call conda activate midas3-llm && python app.py"

REM 5. Start ComfyUI in its virtual environment in a new console
echo Starting ComfyUI...
start "ComfyUI" cmd /k "call C:\AIapps\MIDAS_standaloneapp\venvs\comfyui_venv\Scripts\activate && cd /d C:\AIapps\MIDAS_standaloneapp\ComfyUI && python main.py"

pause
