import argparse
import csv

from tec.ic.ia.p1.models.Decision_Tree import DecisionTree
from tec.ic.ia.p1.models.K_Nearest_Neighbors import KNearestNeighbors
from tec.ic.ia.p1.models.Logistic_Regression import LogisticRegression
from tec.ic.ia.p1.models.Neural_Network import NeuralNetwork
from tec.ic.ia.p1.models.Support_Vector_Machine import SupportVectorMachine


from tec.ic.ia.p1.samples import generar_muestras


parser = argparse.ArgumentParser(
    description='This program allows to train a model of your choice based on Costa Rica\'s elections.')

# Logistic Regression
parser.add_argument("--regresion-logistica",
                    action="store_true", help="Logistic Regression Model.")
parser.add_argument("--l1", action="store_true", help="L1 regularization.")
parser.add_argument("--l2", action="store_true", help="L2 regularization.")

# Neuronal Network
parser.add_argument("--red-neuronal", action="store_true",
                    help="Neural Network Model.")
parser.add_argument("--numero-capas", type=int, help="Number of Layers.")
parser.add_argument("--unidades-por-capa",
                    help="Number of Units per Layer.")
parser.add_argument("--funcion-activacion",
                    choices=["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
                             "hard_sigmoid", "linear"], help="Activation Function.")

# Decision Tree
parser.add_argument("--arbol", action="store_true", help="Decision Tree Model.")
parser.add_argument("--umbral-poda", type=float,
                    help="Minimum information gain required to do a partition.")

# KNN
parser.add_argument("--knn", action="store_true",
                    help="K Nearest Neighbors Model.")
parser.add_argument("--k", type=int, help="Number of Layers.")

# SVM
parser.add_argument("--svm", action="store_true",
                    help="Support Vector Machine Model.")

# Main Program
parser.add_argument("--prefijo", required=True,
                    help="Prefix of all generated files.")
parser.add_argument("--poblacion", required=True,
                    type=int, help="Number of Samples.")
parser.add_argument("--porcentaje-pruebas", required=True, type=float,
                    help="Percentage of samples to use on final validation.")
parser.add_argument("--muestras", required=True, choices=["PAIS", 'SAN_JOSE',
                                                          'ALAJUELA',
                                                          'CARTAGO',
                                                          'HEREDIA',
                                                          'GUANACASTE',
                                                          'PUNTARENAS',
                                                          'LIMON'], help="The function to called when generating samples.")

args = parser.parse_args()

# Checks if just one model is selected
cont_unique_flag = 0
unique_flags = ["regresion_logistica", "red_neuronal", "arbol", "knn", "svm"]
for flag in unique_flags:
    if args.__dict__[flag]:
        cont_unique_flag += 1

if cont_unique_flag > 1:
    parser.error("The application only allows one model per execution.")

# Instantiates the Model class and call its execute method
model = None
normalization = None
if args.regresion_logistica:
    regularization = None
    if not args.l1 and not args.l2:
        parser.error(
            "The logistic regression model have to know which regularization will apply. [L1,L2]")
        exit(-1)
    elif args.l1 and args.l2:
        parser.error(
            "The logistic regression model only supports one regularization. [L1,L2]")
        exit(-1)
    elif args.l1:
        regularization = "l1"
    elif args.l2:
        regularization = "l2"
    model = LogisticRegression(samples_train=None, samples_test=None,prefix = args.prefijo, regularization=regularization)
    normalization = "total_con_one_hot_encoding"
elif args.red_neuronal:
    if args.numero_capas is None or args.unidades_por_capa is None or args.funcion_activacion is None:
        parser.error(
            "The neural network model have to know amount of layers, units per layer and activation function.")
        exit(-1)
    model = NeuralNetwork(samples_train=None, samples_test=None,prefix = args.prefijo,layers=args.numero_capas,
                          units_per_layer=args.unidades_por_capa, activation_function=args.funcion_activacion)
    normalization = "total_con_one_hot_encoding"
elif args.arbol:
    if args.umbral_poda is None:
        parser.error(
            "The decision tree model have to know the pruning threshold.")
        exit(-1)
    model = DecisionTree(samples_train=None, samples_test=None,prefix = args.prefijo,pruning_threshold=args.umbral_poda)
    normalization = "semi"
elif args.knn:
    if args.k is None:
        parser.error(
            "The k nearest neighbors model have to know the k neighbors.")
        exit(-1)
    model = KNearestNeighbors(samples_train=None, samples_test=None,prefix = args.prefijo, k=args.k)
    normalization = "total_sin_one_hot_encoding"
elif args.svm:
    model = SupportVectorMachine(samples_train=None, samples_test=None,prefix = args.prefijo)
    normalization = "total_con_one_hot_encoding"
else:
    parser.print_help()
    exit(0)

print("\n\nGenerating samples")
# Removes non-wanted attributes depending on prediction type and creates samples
configured_sets = generar_muestras(args.muestras, args.poblacion, args.porcentaje_pruebas, normalization)

first_round = ['ACCESIBILIDAD SIN EXCLUSION', 'ACCION CIUDADANA', 'ALIANZA DEMOCRATA CRISTIANA', 'DE LOS TRABAJADORES', 'FRENTE AMPLIO', 'INTEGRACION NACIONAL', 'LIBERACION NACIONAL',
                        'MOVIMIENTO LIBERTARIO', 'NUEVA GENERACION', 'RENOVACION COSTARRICENSE', 'REPUBLICANO SOCIAL CRISTIANO', 'RESTAURACION NACIONAL', 'UNIDAD SOCIAL CRISTIANA', 'NULOS', 'BLANCOS']
second_round = ['ACCION CIUDADANA', 'RESTAURACION NACIONAL', 'NULOS', 'BLANCOS']

#Executes the model for each type of prediction
overall_results = []
training_set_len = len(configured_sets[0][0])
testing_set_len = len(configured_sets[0][2])
for i in range(0,3):
    model.samples_train = [configured_sets[i][0],configured_sets[i][1]]
    model.samples_test = [configured_sets[i][2],configured_sets[i][3]]

    print("\nStart execution ",i+1)
    votes = model.execute()
    #Changes indexes for party names
    for j in range(0,len(votes)):
        if i == 0:
            votes[j] = first_round[int(votes[j])]
        else:
            votes[j] = second_round[int(votes[j])]
    overall_results.append(votes)
#Prepares rows with samples and results for csv
rows = []
for i in range (0,len(configured_sets[2][4])):
    es_entrenamiento = "False"
    row = configured_sets[2][4][i]
    if i < training_set_len:
        es_entrenamiento = "True"    
        row.extend([es_entrenamiento,'','',''])
    else:
        row.extend([es_entrenamiento,overall_results[0][i-training_set_len],overall_results[1][i-training_set_len],overall_results[2][i-training_set_len]])
    rows.append(row)
#Creates output csv
with open('output.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        filewriter.writerow(row)
  

