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
        #Variables para Gain()
        self.entropia_general = 0
        self.lista_atributos = []
        self.valores_atributos = []
        self.valores_answers = []
        self.lista_answers = []
        #Se usa para funcion de importancia
        self.total = []
        #Cantidad de ejemplos
        self.N = 0
        #Guarda los hijos de todos los atributos
        self.atributos_oficiales = []
        self.valores_atributos_oficiales = []
        #guarda el arbol
        self.main_tree = None

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
            numericos = []
            for i in range(len(self.samples_train[0][0])):
                if(type(self.samples_train[0][0][i])!=type("")):
                    numericos += [i]
            if (numericos != []):
                for i in self.samples_train[0]:
                    for j in numericos:
                        if(i[j] >= 0 and i[j]<0.20):
                            i[j] = "[0 , 0.20["
                        elif(i[j] >= 0.20 and i[j]<0.40):
                            i[j] = "[0.20 , 0.40["
                        elif(i[j] >= 0.40 and i[j]<0.60):
                            i[j] = "[0.40 , 0.60["
                        elif(i[j] >= 0.60 and i[j]<0.80):
                            i[j] = "[0.60 , 0.80["
                        elif(i[j] >= 0.80 and i[j]<=1):
                            i[j] = "[0.80 , 1]"
                for i in self.samples_test[0]:
                    for j in numericos:
                        if(i[j] >= 0 and i[j]<0.20):
                            i[j] = "[0 , 0.20["
                        elif(i[j] >= 0.20 and i[j]<0.40):
                            i[j] = "[0.20 , 0.40["
                        elif(i[j] >= 0.40 and i[j]<0.60):
                            i[j] = "[0.40 , 0.60["
                        elif(i[j] >= 0.60 and i[j]<0.80):
                            i[j] = "[0.60 , 0.80["
                        elif(i[j] >= 0.80 and i[j]<=1):
                            i[j] = "[0.80 , 1]"
            

    #Funcion encargada de realizar el arbol de decision.
    def decision_tree_learning(self, examples, attributes, parent_examples):
        #print("---------------------------------------------")
        if (examples == []):
            #print("Hoja: ",self.plurality_value(parent_examples))
            return self.plurality_value(parent_examples)
        elif (self.clasificacion(examples) == True):
            #print("Hoja: ", self.samples_train[1][examples[0]])
            return self.samples_train[1][examples[0]]
        elif (attributes == []):
            #print("Hoja: ",self.plurality_value(examples))
            return self.plurality_value(examples)
        else:
            lista_r=[self.gain(i, examples) for i in attributes]
            #print(lista_r)
            tree = Tree()
            tree.atributo = attributes[argmax(lista_r)]
            ejemplos = [[] for i in range(len(self.valores_atributos_oficiales[tree.atributo]))]
            v = []
            for i in examples:
                v = self.valores_atributos_oficiales[self.atributos_oficiales.index(tree.atributo)]
                ejemplos[v.index(self.samples_train[0][i][tree.atributo])] += [i]
            #print("Atributos :", v)
            #print("Atributos: ",attributes, "\nAtributo actual: ", tree.atributo)
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

    def validar_datos(self):
        count = 0
        partidos = []
        a_x_p = []
        for i in range(len(self.samples_test[0])):
            hijo = False
            arbol = self.main_tree
            while (hijo == False):
                arbol = arbol.hijos[arbol.condicion.index(self.samples_test[0][i][arbol.atributo])]
                if(type(arbol) == type("")):
                    hijo = True
            #print("Era: ", self.samples_test[1][i], " Salio: ", arbol)
            if (self.samples_test[1][i] == arbol):
                count += 1
                if(self.samples_test[1][i] not in partidos):
                    partidos += [self.samples_test[1][i]]
                    a_x_p += [[1,1]]
                else:
                    a_x_p[partidos.index(self.samples_test[1][i])][0] += 1
                    a_x_p[partidos.index(self.samples_test[1][i])][1] += 1
            else:
                if(self.samples_test[1][i] not in partidos):
                    partidos += [self.samples_test[1][i]]
                    a_x_p += [[1,0]]
                else:
                    a_x_p[partidos.index(self.samples_test[1][i])][0] += 1

        print("Aciertos: ", count," De: ", len(self.samples_test[0]), " Accuracy: ", (count/len(self.samples_test[0])*100))
        for i in range(len(a_x_p)):
            print(partidos[i])
            print(a_x_p[i])

    def execute(self):
        #Ejemplo de corrida del arbol.
        att = [i for i in range(len(self.samples_train[0][0]))]
        datos = [i for i in range(len(self.samples_train[0]))]
        self.generar_rangos()
        for i in self.samples_train[0]:
            for j in i:
                if(type(j)!=type("")):
                    print("HABIA UN VALORRRRRRRRR: ", j)
        self.main_tree = self.decision_tree_learning(datos, att, datos)
        self.validar_datos()
        