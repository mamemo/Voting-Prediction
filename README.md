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

First we compared the accuracy of the tree without pruning with different thresholds, with the country results. Including the classification r1, r2 and r2 with r1. This is to see the behavior of the accuracy as it goes down the threshold, comparing the set of training with test.

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

According to the results obtained with the threshold change and without pruning, we can conclude that:

* The accuracy of the tree without pruning, with the training set is greater than 99.8%, which indicates that there is an overfitting in the data, the accuracy of the test set is well below 99%. For this reason, an analysis of different values of thresholds for tree pruning is included.

* As the threshold is decreased, the performance of the training set is reduced, while the performance of the test set increases gradually.

* It can be seen that in each vote estimate, if the threshold value is close to 0, the performance of the training test and the test test is reasonably similar.

* With a threshold of 0.02, the performance of the model increases almost ten percent of its original accuracy with the tree without pruning.

* We can observe that the r2 and r2_with_r1 have a similar behavior, the accuracy of the tree without pruning is 53% and with a threshold of 0.02, 62.5%. Including the vote of the first round to estimate the vote of the second round has no direct effect, the classification of the second round that does not take into consideration the first round behaves practically the same.

* Although we can observe that there are two performance decreases (r1 with 0.05 and r2_with_r1 with 0.10) as the threshold increases, the highest accuracy results can always be observed at the 0.02 threshold.

Now we see some particular behaviors that have been found in the model. First, we will see the behavior by provinces, specifically Cartago and Puntarenas, to analyze if there are differences in accuracy. Next we will see the behavior of the accuracy when it is trained, but with the restriction that it can not repeat attributes in the whole tree.

In the following table the threshold is chosen 0.02 because it is the one that returns a better accuracy when pruning the tree according to the tables analyzed previously. The result of the tree without pruning is not analyzed.


<table>
    <thead>
        <tr>
            <th>Province</th>
            <th colspan=3>Cartago</th>
            <th colspan=3>Puntarenas</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Round</td>
            <td align="center">r1</td>
            <td align="center">r2</td>
            <td align="center">r2 with r1</td>
            <td align="center">r1</td>
            <td align="center">r2</td>
            <td align="center">r2 with r1</td>
        </tr>
        <tr>
            <td align="center">Training</td>
            <td>26.450%</td>
            <td>73.903%</td>
            <td>73.894%</td>
            <td>35.012%</td>
            <td>55.992%</td>
            <td>56.211%</td>
        </tr>
        <tr>
            <td align="center">Test</td>
            <td>26.070%</td>
            <td>73.855%</td>
            <td>73.800%</td>
            <td>35.336%</td>
            <td>55.745%</td>
            <td>55.735%</td>
        </tr>
    </tbody>
</table>

In the table of provinces we can notice some behaviors different to the estimation behavior by country, but first it is important to mention what differentiates the provinces to understand the results:

* Cartago was the province that in its two rounds of voting had the lowest proportion of abstinence, while Puntarenas was one of the provinces with the highest proportion of abstinence.  

How does that difference affect? By taking only the people who voted, Cartago is more accurate because there is more data from the entire province, but in Puntarenas you have data from a smaller sector, so the data contains noise when you match the indicators of the entire population of Puntarenas. The indicators used include the population that did not vote, which also affect the model.

The following table also uses the 0.02 threshold because it is the one that returns the best accuracy when pruning the tree according to the tables analyzed previously. In this case the accuracy of the unpruned tree is shown, because the results are important to mention.

<table>
    <thead>
        <tr>
            <th>Threshold</th>
            <th colspan=3>Without Pruning</th>
            <th colspan=3>0.02</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Round</td>
            <td align="center">r1</td>
            <td align="center">r2</td>
            <td align="center">r2 with r1</td>
            <td align="center">r1</td>
            <td align="center">r2</td>
            <td align="center">r2 with r1</td>
        </tr>
        <tr>
            <td align="center">Training</td>
            <td>27.905%</td>
            <td>62.752%</td>
            <td>62.801%</td>
            <td>27.261%</td>
            <td>62.396%</td>
            <td>62.412%</td>
        </tr>
        <tr>
            <td align="center">Test</td>
            <td>26.705%</td>
            <td>62.085%</td>
            <td>62.160%</td>
            <td>26.980%</td>
            <td>62.315%</td>
            <td>62.320%</td>
        </tr>
    </tbody>
</table>

We can see that training a tree with a restriction can cause the accuracy to increase considerably, to the point that by applying the 0.02 pruning (when before it was the threshold that caused the highest accuracy) the accuracy can decrease instead of increase.
The restriction is that an attribute can be used only once, not repeatedly as in the previous iterations. With only the training, the performance of the training and testing set is similar, which indicates that there is no overfitting as it exists when the tree is trained allowing repeating attributes.
It can be concluded that, including the restriction, there is no increase in the overall performance of the predictions, but there is a clear decrease in overfitting in their initial training.

### K-Nearest Neighbors

### Support Vector Machines


## Samples Generator
