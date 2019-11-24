# Neural Networks
A Neural Network implementation for Machine Learning Class assignment.

## Usage
```
usage: main.py [-h]
               [-o {batchsize,nlayers,nneurons,regularization,alpha,beta}]
               [-e {breast-cancer,ionosphere,pima,wine}]
               [-d {breast-cancer,ionosphere,pima,wine}]

optional arguments:
  -h, --help            show this help message and exit
  -o {batchsize,nlayers,nneurons,regularization,alpha,beta}, --optimize {batchsize,nlayers,nneurons,regularization,alpha,beta}
                        Optimize the parameters for a given dataset.
  -e {breast-cancer,ionosphere,pima,wine}, --evaluate {breast-cancer,ionosphere,pima,wine}
                        Evaluate a parameterized model for a given dataset.
  -d {breast-cancer,ionosphere,pima,wine}, --dataset {breast-cancer,ionosphere,pima,wine}
                        Dataset for parameter optimization.
``` 

## Numerical checks
To run checks with numerical gradients, just type:
```bash
$ python3 checks.py tests/checks/ex1/network.txt tests/checks/ex1/initial_weights.txt tests/checks/ex1/dataset.txt
```
