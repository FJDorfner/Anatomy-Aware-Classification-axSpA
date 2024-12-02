#!/bin/bash

# Convert Jupyter notebooks to Python scripts
jupyter nbconvert --to script Models_02.ipynb
jupyter nbconvert --to script Preprocessing_01.ipynb
jupyter nbconvert --to script Utils_00.ipynb
jupyter nbconvert --to script Training_03.ipynb

# List of converted Python scripts
python_scripts=("Models_02.py" "Preprocessing_01.py" "Utils_00.py" "Training_03.py")

# Move the converted Python scripts to the sister directory
for python_script in "${python_scripts[@]}"; do
    mv "$python_script" ../scripts/
done
