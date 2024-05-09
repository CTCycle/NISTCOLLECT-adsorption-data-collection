@echo off
rem Use this script to create a new environment called "NISTADS"

echo STEP 1: Creation of NISTADS environment
call conda create -n NISTADS python=3.10 
if errorlevel 1 (
    echo Failed to create the environment NISTADS
    goto :eof
)

echo Environment NISTADS successfully created!
rem If present, activate the environment
call conda activate NISTADS


rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call pip install numpy pandas tqdm pubchempy transformers matplotlib seaborn scikit-learn
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul
