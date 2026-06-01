@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem Windows launcher for the benchmark suite.
rem
rem Usage:
rem   run_all.bat
rem   run_all.bat --task hia_hou
rem   run_all.bat --runs 5
rem   run_all.bat --model minimol
rem -----------------------------------------------------------------------------

set "TASK="
set "RUNS=5"
set "MODEL="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--task" (
  set "TASK=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--runs" (
  set "RUNS=%~2"
  shift
  shift
  goto parse_args
)
if /I "%~1"=="--model" (
  set "MODEL=%~2"
  shift
  shift
  goto parse_args
)

echo Unknown argument: %~1
exit /b 1

:args_done

rem Try to locate conda.bat from common Windows install locations.
set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT where conda >nul 2>nul
if not defined CONDA_BAT if errorlevel 1 (
  echo ERROR: Conda was not found on PATH, and no standard install location was detected.
  echo Install Miniconda/Anaconda or open this script from Anaconda Prompt.
  exit /b 1
)
if not defined CONDA_BAT for /f "delims=" %%I in ('where conda') do set "CONDA_BAT=%%I"

set "EXTRA_FLAGS=--runs %RUNS%"
if not "%TASK%"=="" set "EXTRA_FLAGS=!EXTRA_FLAGS! --task !TASK!"

set "PASSED_COUNT=0"
set "FAILED_COUNT=0"

call "%CONDA_BAT%" activate >nul 2>nul
if errorlevel 1 (
  echo ERROR: Unable to initialize Conda from %CONDA_BAT%.
  exit /b 1
)

call :run_model "MiniMol" "minimol_env" "model_assets\MiniMol\code\run_benchmark.py"
call :run_model "DeepMol" "deepmol_env" "model_assets\DeepMol\code\run_benchmark.py"
call :run_model "AttrMasking" "attrmasking_env" "model_assets\AttrMasking\code\run_benchmark.py"
call :run_model "ZairaChem" "zairachem" "model_assets\ZairaChem\code\run_benchmark.py"
call :run_model "MapLight_GNN" "maplight_env" "model_assets\MapLight_GNN\code\run_benchmark.py"

if not "%MODEL%"=="" goto skip_summary

echo.
echo ============================================================
echo   Generating summary reports...
echo ============================================================
call "%CONDA_BAT%" activate minimol_env >nul 2>nul
python results\generate_summary.py
call "%CONDA_BAT%" deactivate >nul 2>nul

:skip_summary

echo.
echo ============================================================
echo   Run Complete
echo   Passed: %PASSED_COUNT%
echo   Failed: %FAILED_COUNT%
echo ============================================================

if "%FAILED_COUNT%"=="0" (
  exit /b 0
) else (
  exit /b 1
)

:run_model
set "MODEL_NAME=%~1"
set "ENV_NAME=%~2"
set "SCRIPT_PATH=%~3"

if not "%MODEL%"=="" (
  if /I not "%MODEL%"=="%MODEL_NAME%" (
    goto :eof
  )
)

call "%CONDA_BAT%" activate "%ENV_NAME%" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Could not activate environment '%ENV_NAME%'.
  set /a FAILED_COUNT+=1
  goto :eof
)

python "%SCRIPT_PATH%" %EXTRA_FLAGS%
if errorlevel 1 (
  echo  ✗ %MODEL_NAME% failed.
  set /a FAILED_COUNT+=1
) else (
  echo  ✓ %MODEL_NAME% completed successfully.
  set /a PASSED_COUNT+=1
)

call "%CONDA_BAT%" deactivate >nul 2>nul
goto :eof
