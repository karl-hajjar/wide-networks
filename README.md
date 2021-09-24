# WIDE-NETWORKS

This repository contains the code of various experiments on wide neural networks.
In particular, we implement classes for abc-parameterizations of NNs as defined by [(Yang & Hu 2021)](https://arxiv.org/pdf/2011.14522.pdf).
Although an equivalent description can be given using only ac-parameterizations, we keep the 3 scales (a, b and c) in the 
code to allow more flexibility depending on how we want to approach the problem of dealing with infinitely wide NNs.

## Structure of the code
All the code related to neural networks is in the directory *pytorch*. The different models we have implemented are in 
this directory along with the base class found in the file *base_model.py* which implements the generic attributes and 
methods all our NNs classes will share. The BaseModel class inherits from the [Pytorch Lightning module](https://pytorch-lightning.readthedocs.io/en/latest/),
and essentially defines the necessary attributes for any NN to work properly, namely the architecture (which is defined 
in the _build_model() method), the activation function (we consider the same activation function at each layer), the loss 
function, the optimizer and the initializer for the parameters of the network. Optionally, the BaseModel class can define
attributes for the normalization (*e.g.* BatchNorm, LayerNorm, etc) and the scheduler, and any of the aforementioned attributes 
(optional or not) can be customized depending on the needs (see examples for the scheduler of ipllr and the initializer 
of abc_param).

All the hyper-parameters which define the model (depth, width, activation function name, loss name, optimizer name, etc)
have to be passed as argument to \__init__() as an object of the class ModelConfig (*pytorch/configs/model.py*). This class
reads from a yaml config file which defines all the necessary objects for a NN (see examples in *pytorch/configs*). Essentially,
the class ModelConfig is here so that one only has to set the yaml config file properly and then the attributes are correctly 
populated in BaseModel via the class ModelConfig.

The code for abc-parameterizations can be found in *pytorch/abc_params*. There, we define the base class for abc-parameterizations,
mainly setting the layer, init and lr scales from the values of *a,b,c*, as well as defining the initial parameters through Gaussians
of appropriate variance depending on the value of *b* and the activation function. All that is architecture specific (fully-connected,
conv, residual, etc) is left out of this base class and has to be implemented in the _build_model() method of the child 
class (see examples in *pytorch/abc_params/fully_connected*). We also define there the base classes for the ntk, muP ([Yang & Hu 2021](https://arxiv.org/pdf/2011.14522.pdf)),
ip and ipllr parameterizations, and there fully-connected implementations in *pytorch/abc_params/fully_connected*.

## Experiment runs

### Setup
Before running any experiment, make sure you first install all the necessary packages:
````bash
pip3 install -r requirements.txt
````
You can optionally create a virtual environment through 
```bash
python3 -m venv your_env_dir
```   
then activate it with 
````bash
source your_env_dir/bin/activate
````
and then install the requirements once the environment is activated. Now, if you haven't installed the **wide-networks**
library in site-packages, before running the command for your experiment, make sure you first add the wide-networks 
library to the PYTHONPATH by running the command
```bash
export PYTHONPATH=$PYTHONPATH:"$PWD"
```
from the main directory (wide-networks/.) of where the wide-networks library is located. 
 
### Python jobs
We define python jobs which can be run with arguments from the command line in the directory *jobs*. Mainly, those jobs 
launch a training / val / test pipeline for a given model using the Lightning module, and the results are collected in a 
dictionary which is saved to a pickle file a the end of training for later examination. Additionally, metrics are logged 
in TensorBoard and can be visualized during training with the command
```bash
tensorboard --logdir=`your_experiment_dir`
```
We have written jobs to launch experiments on MNIST and CIFAR-10 with the fully connected version of different models such as muP ([Yang & Hu 2021](https://arxiv.org/pdf/2011.14522.pdf)),
IP-LLR, Naive-IP which can be found in *jobs/abc_parameterizations*. Arguments can be passed to those Python scripts through 
the command line, but they are optional and the default values will be used if the parameters of the script are not manually 
set. For example, the command
````bash
python3 jobs/abc_parameterizations/fc_muP_run.py --activation="relu" --n_steps=600 --dataset="mnist"
```` 
will launch a training / val / test pipeline with ReLU as the activation function, 600 SGD steps and the MNIST dataset. 
The other parameters of the run (*e.g.* the base learning rate and batch size) will have their default values. The jobs
will automatically create a directory (and potentially subdirectories) for the experiment and save their the python logs,
the tensorboard events and the results dictionary saved to a pickle file as well as the checkpoints saved for the network.

## Visualizing results
To visualize the results after training for a given experiment, one can launch the notebook *experiments-results.ipynb* 
located in *pytorch/notebooks/training/abc_parameterizations*, and simply change the arguments in the "Set variables" cell
to load the results from the corresponding experiment. Then running all the cells will produce (and save) some figures related 
to the training phase (*e.g.* loss *vs.* steps).

    