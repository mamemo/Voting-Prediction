# Voting Prediction

**Description:** This repository contains the first project of Artificial Intelligence course from Instituto Tecnológico de Costa Rica, imparted by the professor Juan Manuel Esquivel. The project consists on a comparison and analysis between 5 models ([Logistic Regression](#logistic-regression) / [Neural Networks](#neural-network) / [Decision Trees](#decision-tree) / [K-Nearest Neighbors](#k-nearest-neighbors) / [Support Vector Machines](#support-vector-machine)) to predict the president electoral votes for Costa Rica on 2018.


### Content:

* [Installation](#installation)
* [Usage](#usage)
* [Models' Report](#models'-report)
* [Samples Generator](#samples-generator)

## Installation:

Before using the project, first you have to install all the project dependencies.

* Python 3.5 or greater, and it has to be 64-bit.
* Tensorflow:
	* Install it with pip:
    ```pip install --upgrade tensorflow```
* Keras:
	* For installing Keras you must have a installation of Tensorflow, Theano or CNTK.
	* Install it  with pip:
    ```pip install keras```
    * For storing the models you can install h5py:
    ```pip install h5py```
    * For visualizing the model's graph:
    ```pip install pydot```
 * Scikit:
 	* For installing Scikit you must have Python 3.3+, Numpy 1.8.2+ and Scipy 0.13.3+.
 	* Install it  with pip:
    ```pip install -U scikit-learn```

Now that all dependencies are installed. You can install the project using:

```pip install -U tec.ic.ia.p1.g07```

Or you can clone the repository by:

```git clone https://github.com/mamemo/Voting-Prediction.git```

## Usage:

When you have the directory with the repository, you have to search the repository on a console and execute the instruction:

```python -m tec.ic.ia.p1.g07.py -h```

This will display all the flags that you can work with:

```
  -h, --help            show this help message and exit
  --regresion-logistica
                        Logistic Regression Model.
  --l1                  L1 regularization.
  --l2                  L2 regularization.
  --red-neuronal        Neural Network Model.
  --numero-capas NUMERO_CAPAS
                        Number of Layers.
  --unidades-por-capa UNIDADES_POR_CAPA
                        Number of Units per Layer.
  --funcion-activacion {softmax,elu,selu,softplus,softsign,relu,tanh,sigmoid,hard_sigmoid,linear}
                        Activation Function.
  --arbol               Decision Tree Model.
  --umbral-poda UMBRAL_PODA
                        Minimum information gain required to do a partition.
  --knn                 K Nearest Neighbors Model.
  --k K                 Number of Layers.
  --svm                 Support Vector Machine Model.
  --prefijo PREFIJO     Prefix of all generated files.
  --poblacion POBLACION
                        Number of Samples.
  --porcentaje-pruebas PORCENTAJE_PRUEBAS
                        Percentage of samples to use on final validation.
  --muestras {PAIS,SAN_JOSE,ALAJUELA,CARTAGO,HEREDIA,GUANACASTE,PUNTARENAS,LIMON}
                        The function to called when generating samples.
```

Each model uses different flags, but they have four in common, you can see the description next to each flag:

```
  --prefijo PREFIJO     Prefix of all generated files.
  --poblacion POBLACION
                        Number of Samples.
  --porcentaje-pruebas PORCENTAJE_PRUEBAS
                        Percentage of samples to use on final validation.
  --muestras {PAIS,SAN_JOSE,ALAJUELA,CARTAGO,HEREDIA,GUANACASTE,PUNTARENAS,LIMON}
                        The function to called when generating samples.
```

To run logistic regression you will need:

```
  --regresion-logistica
                        Logistic Regression Model.
  --l1                  L1 regularization.
  --l2                  L2 regularization.
```

To run neural network you will need:

```
  --red-neuronal        Neural Network Model.
  --numero-capas NUMERO_CAPAS
                        Number of Layers.
  --unidades-por-capa UNIDADES_POR_CAPA
                        Number of Units per Layer.
  --funcion-activacion {softmax,elu,selu,softplus,softsign,relu,tanh,sigmoid,hard_sigmoid,linear}
                        Activation Function.
```

To run decision tree you will need:

```
  --arbol               Decision Tree Model.
  --umbral-poda UMBRAL_PODA
                        Minimum information gain required to do a partition.
```

To run k-nearest neighbors you will need:

```
  --knn                 K Nearest Neighbors Model.
  --k K                 Number of Layers.
```

To run support vector machine you will need:

```
  --svm                 Support Vector Machine Model.
```


## Models' Report:

This section contains the analysis of using each model and how well it performs with different parameters.

### Logistic Regression

For logistic regression we had to compare how it performs with regularization L1 and L2. All the experiment combinations were ran 10 times and the value in the table is the mean. This algorithm uses the normalized samples NOMBRE. In these tests we used the next hyper-parameters to get the best results:
* Learning rate = 0.01
* Training epochs = 5000
* Batch size = 1000
* Regularization Coefficient = 0.01

The results are:

<table>
    <thead>
        <tr>
            <th>Prediction</th>
            <th colspan=4>L1</th>
            <th colspan=4>L2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td colspan=2 align="center">Acurracy</td>
            <td colspan=2 align="center">Loss</td>
            <td colspan=2 align="center">Acurracy</td>
            <td colspan=2 align="center">Loss</td>
        </tr>
        <tr>
            <td></td>
            <td align="center">Train</td>
            <td align="center">Test</td>
            <td align="center">Train</td>
            <td align="center">Test</td>
            <td align="center">Train</td>
            <td align="center">Test</td>
            <td align="center">Train</td>
            <td align="center">Test</td>
        </tr>
        <tr>
            <td>r1</td>
            <td></td>
            <td></td>
            <td>2.59636</td>
            <td>2.58917</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>r2</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>r2 with r1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>

HABLADA DICIENDO PORQUE LOS RESULTADOS DIERON ASI

### Neural Network

### Decision Tree

### K-Nearest Neighbors

### Support Vector Machines


## Samples Generator
