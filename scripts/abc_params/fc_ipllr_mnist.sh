source env/bin/activate  # activate the virtual environment
export PYTHONPATH=$PYTHONPATH:"$PWD"  # add wide-networks library to python path
python3 pytorch/jobs/abc_parameterizations/fc_ipllr_run.py --activation="relu" --n_steps=600 --dataset="mnist"