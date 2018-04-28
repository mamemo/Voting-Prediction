from math import log
from numpy import argmax

entropia_general = 0
lista_atributos = []
valores_atributos = []
valores_answers = []
lista_answers = []
total = []
valores = []
minimos = []
maximos = []
N = 0
atributos_oficiales = []
valores_atributos_oficiales = []

#Funcion responsable de sacar la ganancia a un atributo especifico
def gain(atributo , datos, resultado):
    global N, total, atributos_oficiales, valores_atributos_oficiales, entropia_general, lista_atributos, valores_atributos, valores_answers, lista_answers
    N = len(datos)
    sacar_datos(atributo, datos, resultado)
    sacar_entropia_general(datos)
    final = (gain_total(atributo, datos, resultado))
    if(atributo not in atributos_oficiales):
        atributos_oficiales += [atributo]
        valores_atributos_oficiales += [lista_atributos]
    entropia_general = False
    lista_atributos = []
    valores_atributos = []
    valores_answers = []
    lista_answers = []
    total = []
    N = 0
    return (final)
    
#Recorre la lista de datos para sacar la cantidad de valores por vk.
def sacar_datos(atributo, datos, resultado):
    global lista_atributos, valores_atributos, valores_answers, lista_answers, total, lista
    for i in datos:
        if (lista_atributos.count(lista[i][atributo]) == 0):
            lista_atributos += [lista[i][atributo]]
            valores_atributos += [1]
            lista_2 = []
            for j in lista_answers:
                lista_2 += [0]
            total += [lista_2]
        else:
            valores_atributos[lista_atributos.index(lista[i][atributo])] += 1
        if (lista_answers.count(lista[i][resultado]) == 0):
            lista_answers += [lista[i][resultado]]
            valores_answers += [1]
            for j in total:
                j += [0]
        else:
            valores_answers[lista_answers.index(lista[i][resultado])] += 1
        total[lista_atributos.index(lista[i][atributo])][lista_answers.index(lista[i][resultado])] += 1

#Saca la entropia del atributo.
def sacar_entropia_general(datos):
    global entropia_general, N
    for i in valores_answers:
        entropia_general += (i/N)*log((i/N),2)
    entropia_general *= -1

#Saca la entropia de los hijos del atributo y lo resta a la entropia del atributo.
def gain_total(atributo, datos, resultado):
    global valores_atributos, N, total, valores_answers, entropia_general
    resultado = 0
    for i in range(len(valores_atributos)):
        resultado_parcial = 0
        for j in range(len(valores_answers)):
            if(total[i][j]!=0):
                resultado_parcial += (total[i][j]/valores_atributos[i])*log(total[i][j]/valores_atributos[i],2)
        resultado += (valores_atributos[i]/N) * (-1) * (resultado_parcial)
    return(entropia_general - resultado)

#Establece rangos a los valores numericos.
def generar_rangos():
    global minimos, maximos, valores, lista
    for i in range(len(lista)):
        if(i>0 and valores==[]):
            break
        for j in range(len(lista[i])):
            if(i==0 and type(lista[i][j])!=type("")):
                valores += [j]
                minimos += [lista[i][j]]
                maximos += [lista[i][j]]
            elif(j in valores and type(lista[i][j])!=type("") and lista[i][j]<minimos[valores.index(j)]):
                minimos[valores.index(j)] = lista[i][j]
            elif(j in valores and type(lista[i][j])!=type("") and lista[i][j]>maximos[valores.index(j)]):
                maximos[valores.index(j)] = lista[i][j]+1      
    for i in lista:
        for j in valores:
            suma = (maximos[valores.index(j)]-minimos[valores.index(j)])/4
            for k in range(4):
                 if(i[j] >= (minimos[valores.index(j)]+((suma)*(k))) and  i[j] < (minimos[valores.index(j)]+((suma)*(k+1)))):
                     i[j] = str(minimos[valores.index(j)]+((suma)*(k)))+"-"+str(minimos[valores.index(j)]+((suma)*(k+1)))
                     break
                 elif (i[j] <minimos[valores.index(j)]):
                     i[j] = "< "+str(minimos[valores.index(j)])
                     break
                 elif (i[j] > maximos[valores.index(j)]):
                     i[j] = "> "+str(maximos[valores.index(j)])
                     break

#Clase arbol.
class Tree(object):
    def __init__(self):
        self.hijos = []
        self.condicion = []
        self.atributo = None

#Funcion encargada de realizar el arbol de decision.
def decision_tree_learning(examples, attributes, parent_examples):
    global lista, atributos_oficiales, valores_atributos_oficiales
    print("---------------------------------------------")
    if (examples == []):
        print("Hoja: ",plurality_value(parent_examples))
        return plurality_value(parent_examples)
    elif (clasificacion(examples) == True):
        print("Hoja: ", lista[examples[0]][-1])
        return lista[examples[0]][-1]
    elif (attributes == []):
        print("Hoja: ",plurality_value(examples))
        return plurality_value(examples)
    else:
        lista_r=[gain(i, examples, -1) for i in attributes]
        print(lista_r)
        tree = Tree()
        tree.atributo = attributes[argmax(lista_r)]
        ejemplos = [[] for i in range(len(valores_atributos_oficiales[tree.atributo]))]
        v = []
        for i in examples:
            v = valores_atributos_oficiales[atributos_oficiales.index(tree.atributo)]
            ejemplos[v.index(lista[i][tree.atributo])] += [i]
        print("Atributos :", v, "\nEjemplos: ",ejemplos)
        print("Atributos: ",attributes, "\nAtributo actual: ", tree.atributo)
        attributes.remove(tree.atributo)
        for k in range(len(v)):
            tree.condicion += [v[k]]
            tree.hijos += [decision_tree_learning(ejemplos[k],attributes, examples)]
        return (tree)

#Saca la clasificacion de los datos.
def clasificacion(examples):
    global lista
    iguales = True
    valor = lista[examples[0]][-1]
    for i in examples:
        if(lista[i][-1]!=valor):
            iguales = False
            break
    return iguales

#Saca el valor mas comun de los datos.
def plurality_value(examples):
    global lista
    salidas = []
    valores = []
    for i in examples:
        if(lista[i][-1] not in salidas):
            salidas += [lista[i][-1]]
            valores += [1]
        else:
            valores[salidas.index(lista[i][-1])] += 1
    return (salidas[argmax(valores)])


#Lista de ejemplo.
lista = [["yes" , "no"  , "no"  , "yes" , "some" , "$$$" , "no"  , "yes" , "french" , "0-10"  , "yes"],
         ["yes" , "no"  , "no"  , "yes" , "full" , "$"   , "no"  , "no"  , "thai"   , "30-60" , "no" ],
         ["no"  , "no"  , "no"  , "no"  , "some" , "$"   , "no"  , "no"  , "burger" , "0-10"  , "yes"],
         ["yes" , "yes" , "yes" , "yes" , "full" , "$"   , "yes" , "no"  , "thai"   , "10-30" , "yes"],
         ["yes" , "yes" , "yes" , "no"  , "full" , "$$$" , "no"  , "yes" , "french" , ">60"   , "no" ],
         ["no"  , "no"  , "no"  , "yes" , "some" , "$$"  , "yes" , "yes" , "italian", "0-10"  , "yes"],
         ["no"  , "no"  , "no"  , "no"  , "none" , "$"   , "yes" , "no"  , "burger" , "0-10"  , "no" ],
         ["no"  , "no"  , "no"  , "yes" , "some" , "$$"  , "yes" , "yes" , "thai"   , "0-10"  , "yes"],
         ["no"  , "yes" , "yes" , "no"  , "full" , "$"   , "yes" , "no"  , "burger" , ">60"   , "no" ],
         ["yes" , "yes" , "yes" , "yes" , "full" , "$$$" , "no"  , "yes" , "italian", "10-30" , "no" ],
         ["no"  , "no"  , "no"  , "no"  , "none" , "$"   , "no"  , "no"  , "thai"   , "0-10"  , "no" ],
         ["yes" , "yes" , "yes" , "yes" , "full" , "$"   , "no"  , "no"  , "burger" , "30-60" , "yes"]]

#Ejemplo de corrida del arbol.
att = [i for i in range(len(lista[0])-1)]
datos = [i for i in range(len(lista))]
generar_rangos()
decision_tree_learning(datos, att, datos)
