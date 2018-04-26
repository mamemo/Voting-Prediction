from tec.ic.ac.p1.models.Model import Model
import numpy as np

class KNearestNeighbors(Model):
    def __init__(self, samples, prefix, k):
        super().__init__(samples, prefix)
        self.k = k

    def execute(self):
        pass

# Configures sample values (normalizes them and changes them to numbers if they are discrete)    
def configure_samples(samples):
    discrete_values_indexes = [0,5,7,8,11,12,13,14,19,20,21,22,24,27,28,29,30,31,32,33,34,35,36]
    np_samples_by_attr = np.array(samples).T.tolist()
    for i in range(0,len(np_samples_by_attr)):
        if i in discrete_values_indexes:
            for j in range(0,len(np_samples_by_attr[i])):
               np_samples_by_attr[i][j] = convert_discrete_value(i,np_samples_by_attr[i][j])
        np_samples_by_attr[i] = [float(i) for i in np_samples_by_attr[i]]
        np_samples_by_attr[i] = normalize(np_samples_by_attr[i])
    np_samples_by_attr = np.array(np_samples_by_attr)
    return np_samples_by_attr.T.tolist()           

# Receives a list of values and normalizes them
def normalize(values):
    np_values = np.array(values)
    standard_deviation = np.std(np_values).tolist()
    mean = np.mean(np_values).tolist()
    for i in range(0,len(values)):
        if standard_deviation == 0:
            values[i] = values[i] - mean
        else:
            values[i] = (values[i] - mean)/standard_deviation
    return values

# Changes a value to int if it is discrete
def convert_discrete_value(index,value):
    discrete_value_list = []
    if index == 0:
        discrete_value_list = ['SAN JOSE','ALAJUELA','CARTAGO','HEREDIA','GUANACASTE','PUNTARENAS','LIMON']
    elif index == 5:
        discrete_value_list = ['urbana','no urbana']
    elif index == 7:
        discrete_value_list = ['mujer','hombre']
    elif index == 8:
        discrete_value_list = ['15 a 19','20 a 24','25 a 29','30 a 34','35 a 39','40 a 44','45 a 49','50 a 54','55 a 59','60 a 64','65 a 69','70 a 74','75 a 79','80 a 84','85 y más']
    elif index == 11:
        discrete_value_list = ['vivienda en buen estado','vivienda en mal estado']
    elif index == 12:
        discrete_value_list = ['vivienda hacinada','vivienda no hacinada']
    elif index == 13:
        discrete_value_list = ['ningun año','primaria incompleta','primaria completa','secundaria incompleta','secundaria completa','superior']
    elif index == 14:
        discrete_value_list = ['alfabeta','no alfabeta']
    elif index == 19:
        discrete_value_list = ['dentro de fuerza','fuera de fuerza']
    elif index == 20:
        discrete_value_list = ['sector primario','sector secundario','sector terciario','pensionado','rentista','estudia','oficios domesticos','otros']
    elif index == 21:
        discrete_value_list = ['nacido en extranjero','no nacido en extranjero']
    elif index == 22:
        discrete_value_list = ['discapacidad','sin discapacidad']
    elif index == 24:
        discrete_value_list = ['asegurado','no asegurado'] 
    elif index == 27:
        discrete_value_list = ['jefatura femenina','jefatura masculina','jefatura compartida']	
    elif index == 28:
        discrete_value_list = ['no telefono celular','telefono celular']
    elif index == 29:
        discrete_value_list = ['no telefono residencial','telefono residencial']
    elif index == 30:
        discrete_value_list = ['no computadora','computadora']	
    elif index == 31:
        discrete_value_list = ['no internet','internet']	
    elif index == 32:
        discrete_value_list = ['no electricidad','electricidad']
    elif index == 33:
        discrete_value_list = ['no servicio sanitario','servicio sanitario']
    elif index == 34:
        discrete_value_list = ['no agua','agua']
    elif index == 35:
        discrete_value_list = ['ACCESIBILIDAD SIN EXCLUSION','ACCION CIUDADANA','ALIANZA DEMOCRATA CRISTIANA','DE LOS TRABAJADORES','FRENTE AMPLIO','INTEGRACION NACIONAL','LIBERACION NACIONAL','MOVIMIENTO LIBERTARIO','NUEVA GENERACION','RENOVACION COSTARRICENSE','REPUBLICANO SOCIAL CRISTIANO','RESTAURACION NACIONAL','UNIDAD SOCIAL CRISTIANA','NULOS','BLANCOS']
    elif index == 36:
        discrete_value_list = ['ACCION CIUDADANA','RESTAURACION NACIONAL','NULOS','BLANCOS']
    return discrete_value_list.index(value) + 1


