import argparse

from tec.ic.ac.p1.models.Decision_Tree import DecisionTree
from tec.ic.ac.p1.models.K_Nearest_Neighbors import KNearestNeighbors
from tec.ic.ac.p1.models.Logistic_Regression import LogisticRegression
from tec.ic.ac.p1.models.Neural_Network import NeuralNetwork
from tec.ic.ac.p1.models.Support_Vector_Machine import SupportVectorMachine


from tec.ic.ia.pc1.g07 import generar_muestra_pais


parser = argparse.ArgumentParser(
    description='This program allows to train a model of your choice based on Costa Rica\'s elections.')

# Logistic Regression
parser.add_argument("--regresion-logistica", action="store_true", help="Logistic Regression Model.")
parser.add_argument("--l1", action="store_true", help="L1 regularization.")
parser.add_argument("--l2", action="store_true", help="L2 regularization.")

# Neuronal Network
parser.add_argument("--red-neuronal", action="store_true", help="Neural Network Model.")
parser.add_argument("--numero-capas", type=int, help="Number of Layers.")
parser.add_argument("--unidades-por-capa", type=list, help="Number of Units per Layer.")
parser.add_argument("--funcion-activacion",
                    choices=["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
                             "hard_sigmoid", "linear"], help="Activation Function.")

# Decision Tree
parser.add_argument("--arbol", action="store_true", help="Decision Tree Model.")
parser.add_argument("--umbral-poda", type=float, help="Minimum information gain required to do a partition.")

# KNN
parser.add_argument("--knn", action="store_true", help="K Nearest Neighbors Model.")
parser.add_argument("--k", type=int, help="Number of Layers.")

# SVM
parser.add_argument("--svm", action="store_true", help="Support Vector Machine Model.")

# Main Program
parser.add_argument("--prefijo", required=True, help="Prefix of all generated files.")
parser.add_argument("--poblacion", required=True, type=int, help="Number of Samples.")
parser.add_argument("--porcentaje-pruebas", required=True, type=float,
                    help="Percentage of samples to use on final validation.")
parser.add_argument("--prediccion", required=True, choices=["prediccion_r1","prediccion_r2","prediccion_r2_con_r1"], help="The model prediction.")

args = parser.parse_args()

# Checks if just one model is selected
cont_unique_flag = 0
unique_flags = ["regresion_logistica", "red_neuronal", "arbol", "knn", "svm"]
for flag in unique_flags:
    if args.__dict__[flag]:
        cont_unique_flag += 1

if cont_unique_flag > 1:
    parser.error("The application only allows one model per execution.")

# Removes non-wanted attributes depending on prediction type and creates samples
samples = []
pre_samples = generar_muestra_pais(args.poblacion)
indexes = [1,2,3,4,5,9,10,15,19,21,24,25,27,30,31,32,37,38,40,44]
if args.prediccion == "prediccion_r1":
    indexes.extend([7,56])
elif args.prediccion == "prediccion_r2":
    indexes.extend([6,55])
for i in range(0,args.poblacion):
    sample = []
    for j in range(0,57):
        if j not in indexes:
            sample.append(pre_samples[i][j])
    samples.append(sample)

# Instantiates the Model class and call its execute method
model = None
if args.regresion_logistica:
    regularization = None
    if not args.l1 and not args.l2:
        parser.error("The logistic regression model have to know which regularization will apply. [L1,L2]")
        exit(-1)
    elif args.l1 and args.l2:
        parser.error("The logistic regression model only supports one regularization. [L1,L2]")
        exit(-1)
    elif args.l1:
        regularization = "l1"
    elif args.l2:
        regularization = "l2"
    model = LogisticRegression(samples=samples, prefix=args.prefijo, regularization=regularization)
elif args.red_neuronal:
    if args.numero_capas is None or args.unidades_por_capa is None or args.funcion_activacion is None:
        parser.error("The neural network model have to know amount of layers, units per layer and activation function.")
        exit(-1)
    model = NeuralNetwork(samples=samples, prefix=args.prefijo, layers=args.numero_capas,
                          units_per_layer=args.unidades_por_capa, activation_function=args.funcion_activacion)
elif args.arbol:
    if args.umbral_poda is None:
        parser.error("The decision tree model have to know the pruning threshold.")
        exit(-1)
    model = DecisionTree(samples=samples, prefix=args.prefijo, pruning_threshold=args.umbral_poda)
elif args.knn:
    if args.umbral_poda is None:
        parser.error("The k nearest neighbors model have to know the k neighbors.")
        exit(-1)
    model = KNearestNeighbors(samples=samples, prefix=args.prefijo, k=args.k)
elif args.svm:
    model = SupportVectorMachine(samples=samples, prefix=args.prefijo)
else:
    parser.print_help()

model.execute()
