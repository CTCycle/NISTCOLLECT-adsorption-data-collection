@echo off
rem Use this script to install package dependencies in developer mode

call conda activate NISTADS && cd .. && pip install -e .
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

