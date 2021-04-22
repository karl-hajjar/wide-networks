source env/bin/activate  # activate the virtual environment

export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

python3 scripts/server_scripts/toy_python_script.py