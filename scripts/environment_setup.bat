@echo off
rem Use this script to create a new environment called "NISTCOLLECT"

echo STEP 1: Creation of NISTCOLLECT environment
call conda create -n NISTCOLLECT python=3.11 -y
if errorlevel 1 (
    echo Failed to create the environment NISTCOLLECT
    goto :eof
)

rem If present, activate the environment
call conda activate NISTCOLLECT

rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call pip install numpy==1.26.4 pandas==2.1.4 openpyxl==3.1.5 tqdm==4.66.4
call pip install pubchempy==1.0.4 transformers==4.43.3 matplotlib==3.9.0 seaborn==0.13.2 scikit-learn==1.5.1
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

@echo off
rem install packages in editable mode
echo STEP 3: Install utils packages in editable mode
call cd .. && pip install -e .
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul
