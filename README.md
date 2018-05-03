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
* Numpy:
    * Install it with pip:
    ```python -m pip install --user numpy```
* Scipy:
    * Install it with pip:
  ```python -m pip install --user scipy```
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

For the decision tree we had to compare how it performs with different thresholds, different amounts of attributes (r1, r2 and r2 with r1) and other combinations. All the experiment combinations were ran 10 times and the value in the table is the mean. This algorithm uses the normalized samples NOMBRE. 

First we compared the accuracy of tree without pruning with different thresholds, with the country results. Including the classification r1, r2 and r2 with r1. This is to see the behavior of the accuracy as it goes down the threshold, comparing the set of training with test.

The threshold is in the range of 0 to 1, where 1 is 100%. It is important to mention that as the node of a tree classifies the data, how closer to 1 is its deviation (value of the chi square), the classification will be worse.

The results are:

<table>
    <thead>
        <tr>
            <th>Round 1</th>
            <th align="center">Original</th>
            <th align="center">0.80</th>
            <th align="center">0.60</th>
            <th align="center">0.40</th>
            <th align="center">0.20</th>
            <th align="center">0.10</th>
            <th align="center">0.05</th>
            <th align="center">0.02</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>99.814%</td>
            <td>78,849%</td>
            <td>42,965%</td>
            <td>34,932%</td>
            <td>28,290%</td>
            <td>27,717%</td>
            <td>27,482%</td>
            <td>27,289%</td>
        </tr>
        <tr>
            <td>Test</td>
            <td>18,880%</td>
            <td>20,530%</td>
            <td>23,864%</td>
            <td>25,755%</td>
            <td>26,715%</td>
            <td>27,040%</td>
            <td>26,385%</td>
            <td>27,155%</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Round 2</th>
            <th align="center">Original</th>
            <th align="center">0.80</th>
            <th align="center">0.60</th>
            <th align="center">0.40</th>
            <th align="center">0.20</th>
            <th align="center">0.10</th>
            <th align="center">0.05</th>
            <th align="center">0.02</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>99,911%</td>
            <td>95,178%</td>
            <td>83,981%</td>
            <td>72,201%</td>
            <td>65,539%</td>
            <td>63,788%</td>
            <td>63,394%</td>
            <td>62,916%</td>
        </tr>
        <tr>
            <td>Test</td>
            <td>53,316%</td>
            <td>54,505%</td>
            <td>57,860%</td>
            <td>60,045%</td>
            <td>61,740%</td>
            <td>61,880%</td>
            <td>61,995%</td>
            <td>62,510%</td>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <tr>
            <th>R2 with R1</th>
            <th align="center">Original</th>
            <th align="center">0.80</th>
            <th align="center">0.60</th>
            <th align="center">0.40</th>
            <th align="center">0.20</th>
            <th align="center">0.10</th>
            <th align="center">0.05</th>
            <th align="center">0.02</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Training</td>
            <td>99,983%</td>
            <td>90,336%</td>
            <td>79,884%</td>
            <td>71,436%</td>
            <td>65,473%</td>
            <td>63,694%</td>
            <td>63,415%</td>
            <td>62,975%</td>
        </tr>
        <tr>
            <td>Test</td>
            <td>53,975%</td>
            <td>56,239%</td>
            <td>58,720%</td>
            <td>60,085%</td>
            <td>61,870%</td>
            <td>61,715%</td>
            <td>62,195%</td>
            <td>62,555%</td>
        </tr>
    </tbody>
</table>


[HABLADA DICIENDO PORQUE LOS RESULTADOS DIERON ASI]

Ahora vemos comportamientos particulares que se han encontrado en el modelo. Primeramente se verá el comportamiento por provincias particulares, en concreto CARTAGO y PUNTARENAS las cuales tienen un sesgo más grande hacia un partido político particular. Seguidamente se verá el comportamiento del rendimiento cuando es entrenado, pero con la restricción de que no puede repitir atributos en la totalidad del árbol.

De acuerdo a las tablas de comparación de diferentes umbrales, se escoge el umbral 0.02 debido a que es el que retorna un mejor accuracy al podar el árbol. No se muestra el resultado del árbol sin poda.



### K-Nearest Neighbors

### Support Vector Machines


## Samples Generator
