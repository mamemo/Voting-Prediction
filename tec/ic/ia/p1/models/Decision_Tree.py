from tec.ic.ia.p1.models.Model import Model
from math import log
from numpy import argmax
import scipy.stats as stats
from ete3 import Tree

#Clase arbol.
class DTree(object):
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
        #print(outputs)
        #print(total_outputs)
        #print(total_examples)
        #print(outputs_x_attributes)
        for i in range(len(outputs_x_attributes)):
            partial_result = 0
            for j in range(len(outputs_x_attributes[i])):
                true_irrelevance = total_outputs[j] * (len(examples[i])/total_examples)
                #print("ti "+str(j)+": ",true_irrelevance)
                partial_result += ((outputs_x_attributes[i][j] - true_irrelevance)**2)/true_irrelevance
                #print("rp "+str(j)+": ",((outputs_x_attributes[i][j] - true_irrelevance)**2)/true_irrelevance)
            result += partial_result
        #print("Desviacion: ",result)
        return(1 - stats.chi2.cdf(x=result, df=((len(outputs)-1)*(len(attributes)-1))))

    def validar_datos(self):
        count = 0
        partidos = []
        a_x_p = []
        for i in range(len(self.samples_test[0])):
            hijo = False
            arbol = self.main_tree
            while (hijo == False):
                if(arbol.condicion.count(self.samples_test[0][i][arbol.atributo]) != 0):
                    arbol = arbol.hijos[arbol.condicion.index(self.samples_test[0][i][arbol.atributo])]
                    if(type(arbol) == type("")):
                        hijo = True
                else:
                    print("Atributo: \"",self.samples_test[0][i][arbol.atributo], "\" no se encuentra en arbol.")
                    break
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

    def recorrer_arbol(self, tree):
        arbol = ""
        for k in range(len(tree.hijos)):
            if (type(tree.hijos[k]) != type("")):
                arbol += self.recorrer_arbol(tree.hijos[k])+" <"+tree.condicion[k]+">"
            else:
                #arbol += "<"+tree.condicion[k]+"> "+tree.hijos[k]
                arbol += "<"+tree.condicion[k]+"> "+"PAC"
            if(k < len(tree.hijos)-1):
                arbol+=","
        return ("("+arbol+")"+str(tree.atributo))
        
        

    def prueba_2(self):
        self.samples_train = self.samples_test = [[
                                    ["rainy"    , "hot"  , "high"   , "false" ],
                                    ["rainy"    , "hot"  , "high"   , "true"  ],
                                    ["overoast" , "hot"  , "high"   , "false" ],
                                    ["sunny"    , "mild" , "high"   , "false" ],
                                    ["sunny"    , "cool" , "normal" , "false" ],
                                    ["sunny"    , "cool" , "normal" , "true"  ],
                                    ["overoast" , "cool" , "normal" , "true"  ],
                                    ["rainy"    , "mild" , "high"   , "false" ],
                                    ["rainy"    , "cool" , "normal" , "false" ],
                                    ["sunny"    , "mild" , "normal" , "false" ],
                                    ["rainy"    , "mild" , "normal" , "true"  ],
                                    ["overoast" , "mild" , "high"   , "true"  ],
                                    ["overoast" , "hot"  , "normal" , "false" ],
                                    ["sunny"    , "mild" , "high"   , "true"  ]], 
                                    ["no", "no", "yes", "yes","yes","no","yes","no","yes","yes","yes","yes","yes","no"]]
        self.samples_train = self.samples_test = [[
                 ["yes" , "no"  , "no"  , "yes" , "some" , "$$$" , "no"  , "yes" , "french" , "0-10"  ],
                 ["yes" , "no"  , "no"  , "yes" , "full" , "$"   , "no"  , "no"  , "thai"   , "30-60" ],
                 ["no"  , "no"  , "no"  , "no"  , "some" , "$"   , "no"  , "no"  , "burger" , "0-10"  ],
                 ["yes" , "yes" , "yes" , "yes" , "full" , "$"   , "yes" , "no"  , "thai"   , "10-30" ],
                 ["yes" , "yes" , "yes" , "no"  , "full" , "$$$" , "no"  , "yes" , "french" , ">60"   ],
                 ["no"  , "no"  , "no"  , "yes" , "some" , "$$"  , "yes" , "yes" , "italian", "0-10"  ],
                 ["no"  , "no"  , "no"  , "no"  , "none" , "$"   , "yes" , "no"  , "burger" , "0-10"  ],
                 ["no"  , "no"  , "no"  , "yes" , "some" , "$$"  , "yes" , "yes" , "thai"   , "0-10"  ],
                 ["no"  , "yes" , "yes" , "no"  , "full" , "$"   , "yes" , "no"  , "burger" , ">60"   ],
                 ["yes" , "yes" , "yes" , "yes" , "full" , "$$$" , "no"  , "yes" , "italian", "10-30" ],
                 ["no"  , "no"  , "no"  , "no"  , "none" , "$"   , "no"  , "no"  , "thai"   , "0-10"  ],
                 ["yes" , "yes" , "yes" , "yes" , "full" , "$"   , "no"  , "no"  , "burger" , "30-60" ]],
                 ["yes", "no", "yes" , "yes", "no" , "yes", "no", "yes", "no", "no" , "no", "yes"]]

    def execute(self):
        #Ejemplo de corrida del arbol.
        #self.prueba_2()
        att = [i for i in range(len(self.samples_train[0][0]))]
        datos = [i for i in range(len(self.samples_train[0]))]
        self.generar_rangos()
        #print("-------Hacer arbol--------")
        self.main_tree = self.decision_tree_learning(datos, att, datos)

        #t = Tree(self.recorrer_arbol(self.main_tree)+";", format=1)
        #print (t.get_ascii(attributes=["name", "label"])+"\n")

        #print("-------Validar datos--------")
        self.validar_datos()
        print("-------Hacer poda con umbral ",self.pruning_threshold,"--------")
        self.main_tree = self.pruning_tree(datos, self.main_tree)
        #print(self.main_tree.hijos)

        #t = Tree(self.recorrer_arbol(self.main_tree)+";", format=1)
        #print (t.get_ascii(attributes=["name", "label"])+"\n")


        """for i in range(len(self.main_tree.hijos)):
            print("("+str(self.main_tree.atributo)+") "+self.main_tree.condicion[i],":")
            if(type(self.main_tree.hijos[i])!=type("")):
                t = Tree(self.recorrer_arbol(self.main_tree.hijos[i])+";", format=1)
                print (t.get_ascii(attributes=["name", "label"])+"\n")
            else:
                print(self.main_tree.hijos[i]+"\n")"""
        print("-------Validar datos again--------")
        self.validar_datos()



        """self.samples_train = [[],["si","si","si","si","si","si","si","si","si","si","si","si","si","si","si",
                                  "no","no","no","no","no","no","no","no","no","no","no","no","no","no","no"]]
        print("")
        print("Porcentaje: ",self.total_desviation(["female","male"],[[0,1,15,16,17,18,19,20,21,22],[2,3,4,5,6,7,8,9,10,11,12,13,14,23,24,25,26,27,28,29]]))
        print("")
        print("Porcentaje: ",self.total_desviation(["class ix","class x"],[[0,0,0,0,0,0,15,15,15,15,15,15,15,15],[0,0,0,0,0,0,0,0,0,15,15,15,15,15,15,15]]))
        print("")
        print("Porcentaje: ",self.total_desviation(["t","f"],[[15,15,15,15,15],[0]]))
        print("")"""
