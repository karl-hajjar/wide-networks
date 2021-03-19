source env/bin/activate  # activate the virtual environment

export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path

python3 "$1"  # run the python script passed as argument