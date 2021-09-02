source env/bin/activate  # activate the virtual environment
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path
python3 pytorch/jobs/abc_parameterizations/standard_fc_ip_run.py --activation="relu" --n_steps=600 --dataset="mnist"