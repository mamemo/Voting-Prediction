from tec.ic.ia.p1.models.Model import Model
from math import log
from numpy import argmax

#Clase arbol.
class Tree(object):
    def __init__(self):
        self.hijos = []
        self.condicion = []
        self.atributo = None

class DecisionTree(Model):
    def __init__(self, samples_train, samples_test, prefix, pruning_threshold):
        super().__init__(samples_train, samples_test, prefix)
        self.pruning_threshold = pruning_threshold
        self.entropia_general = 0
        self.lista_atributos = []
        self.valores_atributos = []
        self.valores_answers = []
        self.lista_answers = []
        self.total = []
        self.valores = []
        self.minimos = []
        self.maximos = []
        self.N = 0
        self.atributos_oficiales = []
        self.valores_atributos_oficiales = []

    #Funcion responsable de sacar la ganancia a un atributo especifico
    def gain(self, atributo , datos):
        self.N = len(datos)
        self.sacar_datos(atributo, datos)
        self.sacar_entropia_general(datos)
        final = (self.gain_total(atributo, datos))
        if(atributo not in self.atributos_oficiales):
            self.atributos_oficiales += [atributo]
            self.valores_atributos_oficiales += [self.lista_atributos]
        self.entropia_general = False
        self.lista_atributos = []
        self.valores_atributos = []
        self.valores_answers = []
        self.lista_answers = []
        self.total = []
        self.N = 0
        return (final)
        
    #Recorre la lista de datos para sacar la cantidad de valores por vk.
    def sacar_datos(self, atributo, datos):
        for i in datos:
            if (self.lista_atributos.count(self.samples_train[0][i][atributo]) == 0):
                self.lista_atributos += [self.samples_train[0][i][atributo]]
                self.valores_atributos += [1]
                lista_2 = []
                for j in self.lista_answers:
                    lista_2 += [0]
                self.total += [lista_2]
            else:
                self.valores_atributos[self.lista_atributos.index(self.samples_train[0][i][atributo])] += 1
            if (self.lista_answers.count(self.samples_train[1][i]) == 0):
                self.lista_answers += [self.samples_train[1][i]]
                self.valores_answers += [1]
                for j in self.total:
                    j += [0]
            else:
                self.valores_answers[self.lista_answers.index(self.samples_train[1][i])] += 1
            self.total[self.lista_atributos.index(self.samples_train[0][i][atributo])][self.lista_answers.index(self.samples_train[1][i])] += 1

    #Saca la entropia del atributo.
    def sacar_entropia_general(self, datos):
        for i in self.valores_answers:
            self.entropia_general += (i/self.N)*log((i/self.N),2)
        self.entropia_general *= -1

    #Saca la entropia de los hijos del atributo y lo resta a la entropia del atributo.
    def gain_total(self, atributo, datos):
        resultado = 0
        for i in range(len(self.valores_atributos)):
            resultado_parcial = 0
            for j in range(len(self.valores_answers)):
                if(self.total[i][j]!=0):
                    resultado_parcial += (self.total[i][j]/self.valores_atributos[i])*log(self.total[i][j]/self.valores_atributos[i],2)
            resultado += (self.valores_atributos[i]/self.N) * (-1) * (resultado_parcial)
        return(self.entropia_general - resultado)

    #Establece rangos a los valores numericos.
    def generar_rangos(self):
        for i in range(len(self.samples_train[0])):
            if(i>0 and self.valores==[]):
                break
            for j in range(len(self.samples_train[0][i])):
                if(i==0 and type(self.samples_train[0][i][j])!=type("")):
                    self.valores += [j]
                    self.minimos += [self.samples_train[0][i][j]]
                    self.maximos += [self.samples_train[0][i][j]]
                elif(j in self.valores and type(self.samples_train[0][i][j])!=type("") and self.samples_train[0][i][j]<self.minimos[self.valores.index(j)]):
                    self.minimos[self.valores.index(j)] = self.samples_train[0][i][j]
                elif(j in self.valores and type(self.samples_train[0][i][j])!=type("") and self.samples_train[0][i][j]>self.maximos[self.valores.index(j)]):
                    self.maximos[self.valores.index(j)] = self.samples_train[0][i][j]+1      
        for i in self.samples_train[0]:
            for j in self.valores:
                suma = (self.maximos[self.valores.index(j)]-self.minimos[self.valores.index(j)])/4
                for k in range(4):
                     if(i[j] >= (self.minimos[self.valores.index(j)]+((suma)*(k))) and  i[j] < (self.minimos[self.valores.index(j)]+((suma)*(k+1)))):
                         i[j] = str(self.minimos[self.valores.index(j)]+((suma)*(k)))+"-"+str(self.minimos[self.valores.index(j)]+((suma)*(k+1)))
                         break
                     elif (i[j] <self.minimos[self.valores.index(j)]):
                         i[j] = "< "+str(self.minimos[self.valores.index(j)])
                         break
                     elif (i[j] > self.maximos[self.valores.index(j)]):
                         i[j] = "> "+str(self.maximos[self.valores.index(j)])
                         break

    #Funcion encargada de realizar el arbol de decision.
    def decision_tree_learning(self, examples, attributes, parent_examples):
        print("---------------------------------------------")
        if (examples == []):
            print("Hoja: ",self.plurality_value(parent_examples))
            return self.plurality_value(parent_examples)
        elif (self.clasificacion(examples) == True):
            print("Hoja: ", self.samples_train[1][examples[0]])
            return self.samples_train[1][examples[0]]
        elif (attributes == []):
            print("Hoja: ",self.plurality_value(examples))
            return self.plurality_value(examples)
        else:
            lista_r=[self.gain(i, examples) for i in attributes]
            print(lista_r)
            tree = Tree()
            tree.atributo = attributes[argmax(lista_r)]
            ejemplos = [[] for i in range(len(self.valores_atributos_oficiales[tree.atributo]))]
            v = []
            for i in examples:
                v = self.valores_atributos_oficiales[self.atributos_oficiales.index(tree.atributo)]
                ejemplos[v.index(self.samples_train[0][i][tree.atributo])] += [i]
            print("Atributos :", v, "\nEjemplos: ",ejemplos)
            print("Atributos: ",attributes, "\nAtributo actual: ", tree.atributo)
            attributes.remove(tree.atributo)
            for k in range(len(v)):
                tree.condicion += [v[k]]
                tree.hijos += [self.decision_tree_learning(ejemplos[k],attributes, examples)]
            return (tree)

    #Saca la clasificacion de los datos.
    def clasificacion(self, examples):
        iguales = True
        valor = self.samples_train[1][examples[0]]
        for i in examples:
            if(self.samples_train[1][i]!=valor):
                iguales = False
                break
        return iguales

    #Saca el valor mas comun de los datos.
    def plurality_value(self, examples):
        salidas = []
        valores = []
        for i in examples:
            if(self.samples_train[1][i] not in salidas):
                salidas += [self.samples_train[1][i]]
                valores += [1]
            else:
                valores[salidas.index(self.samples_train[1][i])] += 1
        return (salidas[argmax(valores)])

    def execute(self):
        print("\n\n----------entro al arbol de decision-----------\n\n")

        print("Cantidad ejemplos training: ", len(self.samples_train[0]))
        print("Cantidad ejemplos testing: ", len(self.samples_test[0]),"\n\n")

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
        self.samples_train = [[],[]]
        print("LISTA: ", self.samples_train)
        for i in lista:
            self.samples_train[0] += [i[:-1]]
            self.samples_train[1] += [i[-1]]
        #self.generar_rangos()
        print("LISTA: ", self.samples_train)
        self.decision_tree_learning(datos, att, datos)

        print("\n\n----------salio del arbol de decision-----------\n\n")
        