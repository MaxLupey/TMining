@echo off
set ENV_NAME=textmining
set USER_NAME=%USERNAME%
echo Checking if environment '%ENV_NAME%' exists...

for /f "delims=" %%a in ('where conda') do set CONDA_PATH=%%a
if not "%CONDA_PATH%"=="" (
    echo Conda found at: %CONDA_PATH%
    conda info --envs | findstr /b /c:"%ENV_NAME%" > nul

    if errorlevel 1 (
        echo The environment '%ENV_NAME%' does not exist. Creating a new environment...

        conda env create -n %ENV_NAME% -f env\textmining.yml

        if errorlevel 1 (
            echo There was a problem creating the environment '%ENV_NAME%'. Please check the error messages above.
            pause
        ) else (
            echo The environment '%ENV_NAME%' has been successfully created.
            echo Activate it by running 'conda activate %ENV_NAME%'.
        )
    ) else (
        echo The environment '%ENV_NAME%' already exists. Please activate it by running 'conda activate %ENV_NAME%'.
        pause
    )
) else (
    echo Conda not found.
    pause
)