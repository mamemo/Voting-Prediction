from tec.ic.ia.p1.models.Model import Model
from math import log
from numpy import argmax
import scipy.stats as stats
import datetime

#Clase arbol.
class DTree(object):
    def __init__(self):
        self.hijos = []
        self.condicion = []
        self.atributo = None

class DecisionTree(Model):
    def __init__(self, samples_train, samples_test, prefix, pruning_threshold):
        super().__init__(samples_train, samples_test, prefix)
        self.votos = [['ACCESIBILIDAD SIN EXCLUSION', 'ACCION CIUDADANA', 'ALIANZA DEMOCRATA CRISTIANA', 'DE LOS TRABAJADORES', 'FRENTE AMPLIO', 'INTEGRACION NACIONAL', 'LIBERACION NACIONAL','MOVIMIENTO LIBERTARIO', 'NUEVA GENERACION', 'RENOVACION COSTARRICENSE', 'REPUBLICANO SOCIAL CRISTIANO', 'RESTAURACION NACIONAL', 'UNIDAD SOCIAL CRISTIANA', 'NULOS', 'BLANCOS'],['ACCION CIUDADANA', 'RESTAURACION NACIONAL', 'NULOS', 'BLANCOS']]
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
        self.prediccion = 0
        #guarda el arbol
        self.main_tree = None
        self.oficial_outputs = []

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
                    if(i[j] >= 0 and i[j]<0.25):
                        i[j] = "[0 , 0.25["
                    elif(i[j] >= 0.25 and i[j]<0.50):
                        i[j] = "[0.25 , 0.50["
                    elif(i[j] >= 0.50 and i[j]<0.75):
                        i[j] = "[0.50 , 0.75["
                    elif(i[j] >= 0.75 and i[j]<=1):
                        i[j] = "[0.75 , 1]"
            for i in self.samples_test[0]:
                for j in numericos:
                    if(i[j] >= 0 and i[j]<0.25):
                        i[j] = "[0 , 0.25["
                    elif(i[j] >= 0.25 and i[j]<0.50):
                        i[j] = "[0.25 , 0.50["
                    elif(i[j] >= 0.50 and i[j]<0.75):
                        i[j] = "[0.50 , 0.75["
                    elif(i[j] >= 0.75 and i[j]<=1):
                        i[j] = "[0.75 , 1]"
            

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
            tree = DTree()
            tree.atributo = attributes[argmax(lista_r)]
            ejemplos = [[] for i in range(len(self.valores_atributos_oficiales[tree.atributo]))]
            v = []
            for i in examples:
                v = self.valores_atributos_oficiales[self.atributos_oficiales.index(tree.atributo)]
                ejemplos[v.index(self.samples_train[0][i][tree.atributo])] += [i]
            #print("Atributos :", v)
            #print("Atributos: ",attributes, "\nAtributo actual: ", tree.atributo)
            at = attributes[0:]      #con repetir atributos
            at.remove(tree.atributo) #con repetir atributos
            #attributes.remove(tree.atributo) #sin repetir atributos
            for k in range(len(v)):
                tree.condicion += [v[k]]
                #con repetir atributos
                tree.hijos += [self.decision_tree_learning(ejemplos[k],at, examples)]
                #con repetir atributos
                #tree.hijos += [self.decision_tree_learning(ejemplos[k],attributes, examples)]
            return (tree)

    #Saca la clasificacion de los datos.
    def clasificacion(self, examples):
        iguales = True
        valor = self.samples_train[1][examples[0]]
        for i in examples:
            if(self.samples_train[1][i] not in self.oficial_outputs):
                self.oficial_outputs += [self.samples_train[1][i]]
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

    def pruning_tree(self, examples, tree):
        attributes = []
        values = []
        for i in examples:
            if(self.samples_train[0][i][tree.atributo] not in attributes):
                attributes += [self.samples_train[0][i][tree.atributo]]
                values += [[i]]
            else:
                values[attributes.index(self.samples_train[0][i][tree.atributo])] += [i]
        cant_hojas = 0
        for k in range(len(tree.hijos)):
            if (type(tree.hijos[k]) != type("")):
                #atributo = tree.hijos[k].atributo
                tree.hijos[k] = self.pruning_tree(values[attributes.index(tree.condicion[k])] , tree.hijos[k])
                if (type(tree.hijos[k])==type("")):
                    cant_hojas += 1
                    #print("Elimino atributo: ", atributo, " ahora es: ", tree.hijos[k])
            else:
                cant_hojas += 1
        if(cant_hojas > 0):
            desviation = self.total_desviation(attributes, values)
            if (desviation > self.pruning_threshold):
                #print(" + Desviacion: ", desviation, " para nodo: ", tree.atributo)
                return self.plurality_value(examples)
            else:
                return tree
        else:
            return tree

    def total_desviation(self, attributes, examples):
        outputs = []
        total_outputs = []
        total_examples = 0
        outputs_x_attributes = []
        result = 0
        for i in range(len(examples)):
            outputs_x_attributes += [[]]
            outputs_x_attributes[-1] = [0 for i in range((len(outputs)))]
            for j in examples[i]:
                if(self.samples_train[1][j] not in outputs):
                    outputs += [self.samples_train[1][j]]
                    total_outputs += [1]
                    total_examples += 1
                    outputs_x_attributes[i] += [1]
                else:
                    total_examples += 1
                    total_outputs[outputs.index(self.samples_train[1][j])] += 1
                    outputs_x_attributes[i][outputs.index(self.samples_train[1][j])] += 1
        for i in range(len(outputs_x_attributes)):
            partial_result = 0
            for j in range(len(outputs_x_attributes[i])):
                true_irrelevance = total_outputs[j] * (len(examples[i])/total_examples)
                partial_result += ((outputs_x_attributes[i][j] - true_irrelevance)**2)/true_irrelevance
            result += partial_result
        return(1 - stats.chi2.cdf(x=result, df=((len(outputs)-1)*(len(attributes)-1))))

    def validar_datos(self,data):
        count = 0
        partidos = []
        a_x_p = []
        result = []
        for i in range(len(data[0])):
            hijo = False
            arbol = self.main_tree
            while (hijo == False):
                if(arbol.condicion.count(data[0][i][arbol.atributo]) != 0):
                    arbol = arbol.hijos[arbol.condicion.index(data[0][i][arbol.atributo])]
                    if(type(arbol) == type("")):
                        hijo = True
                else:
                    print("Atributo: \"",data[0][i][arbol.atributo], "\" no se encuentra en arbol.")
                    break
            #print("Era: ", data[1][i], " Salio: ", arbol)
            if (data[1][i] == arbol):
                result += [self.votos[self.prediccion].index(data[1][i])]
                count += 1
                if(data[1][i] not in partidos):
                    partidos += [data[1][i]]
                    a_x_p += [[1,1]]
                else:
                    a_x_p[partidos.index(data[1][i])][0] += 1
                    a_x_p[partidos.index(data[1][i])][1] += 1
            else:
                result += [self.votos[self.prediccion].index(data[1][i])]
                if(data[1][i] not in partidos):
                    partidos += [data[1][i]]
                    a_x_p += [[1,0]]
                else:
                    a_x_p[partidos.index(data[1][i])][0] += 1

        print("Aciertos: ", count," De: ", len(data[0]), " Accuracy: ", (count/len(data[0])*100),"%")
        #for i in range(len(a_x_p)):
        #    print(partidos[i])
        #    print(a_x_p[i])
        return result

    def execute(self):
        result = []
        self.atributos_oficiales = []
        self.valores_atributos_oficiales = []
        self.main_tree = None
        self.oficial_outputs = []
        att = [i for i in range(len(self.samples_train[0][0]))]
        datos = [i for i in range(len(self.samples_train[0]))]
        self.generar_rangos()
        
        #Se crea el arbol
        self.main_tree = self.decision_tree_learning(datos, att, datos)
        if(len(self.oficial_outputs) > 4):
            self.prediccion = 0
        else:
            self.prediccion = 1

        print("\n-------Before pruning--------")
        #Se validan los datos con arbol sin podar
        print("-> Training Set")
        self.validar_datos(self.samples_train)
        print("-> Test Set")
        self.validar_datos(self.samples_test)
        
        #Se poda el arbol
        self.main_tree = self.pruning_tree(datos, self.main_tree)
        print("\n-------After pruning--------")
        if(type(self.main_tree) == type("")):
            print("Tree ended as a single leaf: ", self.main_tree)
            print("-> Training Set")
            print("Aciertos: ", self.samples_train[1].count(self.main_tree)," De: ", len(self.samples_train[0]), " Accuracy: ", (self.samples_train[1].count(self.main_tree)/len(self.samples_train[0]))*100,"%")
            print("-> Test Set")
            print("Aciertos: ", self.samples_test[1].count(self.main_tree)," De: ", len(self.samples_test[0]), " Accuracy: ", (self.samples_test[1].count(self.main_tree)/len(self.samples_test[0]))*100,"%")
            return([self.votos[self.prediccion].index(self.main_tree)]*((len(self.samples_train[0])+len(self.samples_test[0]))))
        else:
            #Se validan los datos con arbol podado
            print("-> Training Set")
            result += self.validar_datos(self.samples_train)
            print("-> Test Set")
            result += self.validar_datos(self.samples_test)
            #print(result)
            print("\nEnding execution\n")
            return result
