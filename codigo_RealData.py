# -*- coding: utf-8 with BOM -*-
"""
Script para analisis de Datos

@author: REALDATA TEAM
"""

team="""
 /$$$$$$$  /$$$$$$$$  /$$$$$$  /$$       /$$$$$$$   /$$$$$$  /$$$$$$$$/$$$$$$ 
| $$__  $$| $$_____/ /$$__  $$| $$      | $$__  $$ /$$__  $$|__  $$__/$$__  $$
| $$  \ $$| $$      | $$  \ $$| $$      | $$  \ $$| $$  \ $$   | $$ | $$  \ $$
| $$$$$$$/| $$$$$   | $$$$$$$$| $$      | $$  | $$| $$$$$$$$   | $$ | $$$$$$$$
| $$__  $$| $$__/   | $$__  $$| $$      | $$  | $$| $$__  $$   | $$ | $$__  $$
| $$  \ $$| $$      | $$  | $$| $$      | $$  | $$| $$  | $$   | $$ | $$  | $$
| $$  | $$| $$$$$$$$| $$  | $$| $$$$$$$$| $$$$$$$/| $$  | $$   | $$ | $$  | $$
|__/  |__/|________/|__/  |__/|________/|_______/ |__/  |__/   |__/ |__/  |__/
"""
print(team)

# Se importan todas las librerías que vamos a usar posteriormente
import pandas as pd
from sklearn.model_selection import cross_val_predict
# from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error


# Definición de un método a través del cual haremos todos los pasos de preprocesamiento de los datos
# de forma automática, teniendo como entrada la direccion del archivo de datos.
def preparedata(file):
# CARGA DE LOS DATOS
    # Lecutura del fichero csv
	data = pd.read_csv(file)
    
# ANÁLISIS INICIAL DE LAS VARIABLES
	# Eliminación de la variable Socio_Demo_01, ya que no es representativa y tiene muchos valores distintos (922)
	data = data.drop(['Socio_Demo_01'], axis=1)
    
# TRATAMIENTO DE VARIABLES NUMÉRICAS
    # Agregación sobre las variables 'Cons', 'Sal' y 'Op' ya que nos permitiría reducir la dimensionalidad
    # de los datos y a efectos prácticos, obtenemos la misma información que por separado.
    # Nomalizacion de variables numericas
	col_list_numeric_variables = list(data[['Imp_Cons_01','Imp_Cons_02','Imp_Cons_03',
        										'Imp_Cons_04','Imp_Cons_05','Imp_Cons_06',
        										'Imp_Cons_07','Imp_Cons_08','Imp_Cons_09',
        										'Imp_Cons_10','Imp_Cons_11','Imp_Cons_12',
        										'Imp_Cons_13','Imp_Cons_14','Imp_Cons_15',
        										'Imp_Cons_16','Imp_Cons_17','Imp_Sal_01',
        										'Imp_Sal_02','Imp_Sal_03','Imp_Sal_04',
        										'Imp_Sal_05','Imp_Sal_06','Imp_Sal_07',
        										'Imp_Sal_08','Imp_Sal_09','Imp_Sal_10',
        										'Imp_Sal_11','Imp_Sal_12','Imp_Sal_13',
        										'Imp_Sal_14','Imp_Sal_15','Imp_Sal_16',
        										'Imp_Sal_17','Imp_Sal_18','Imp_Sal_19',
        										'Imp_Sal_20','Imp_Sal_21','Num_Oper_01',
        										'Num_Oper_02','Num_Oper_03','Num_Oper_04',
        										'Num_Oper_05','Num_Oper_06','Num_Oper_07',
        										'Num_Oper_08','Num_Oper_09','Num_Oper_10',
        										'Num_Oper_11','Num_Oper_12','Num_Oper_13',
        										'Num_Oper_14','Num_Oper_15','Num_Oper_16',
        										'Num_Oper_17','Num_Oper_18','Num_Oper_19',
        										'Num_Oper_20','Socio_Demo_03','Socio_Demo_05']])
	# Estandarizacion de variables numericas
	data[col_list_numeric_variables] = (data[col_list_numeric_variables] - data[col_list_numeric_variables].mean(axis=0)) / data[col_list_numeric_variables].std(axis=0)
	# Se cambian los valores NaN por 0, ya que la normalización puede producirlo al dividir cualquier valor entre 0
	data[col_list_numeric_variables] = data[col_list_numeric_variables].fillna(0)

# TRATAMIENTO DE VARIABLES CATEGÓRICAS
	# Según el estudio realizado para Socio_Demo_04 en el que se observaban 2 claros grupos
	# Agrupamos el conjunto de valores para Socio_Demo_04 para los valores menores a 5000 y mayores o iguales a 5000.
	# Reducimos de 24 a 2 valores unicos
	thresh = 5000
	data['Socio_Demo_04'][(data['Socio_Demo_04'] < thresh)] = 0
	data['Socio_Demo_04'][(data['Socio_Demo_04'] >= thresh)] = 1
    
	col_list_categorized_variables = list(data[['Ind_Prod_01','Ind_Prod_02','Ind_Prod_03',
												'Ind_Prod_04','Ind_Prod_05','Ind_Prod_06',
												'Ind_Prod_07','Ind_Prod_08','Ind_Prod_09',
												'Ind_Prod_10','Ind_Prod_11','Ind_Prod_12',
												'Ind_Prod_13','Ind_Prod_14','Ind_Prod_15',
												'Ind_Prod_16','Ind_Prod_17','Ind_Prod_18',
												'Ind_Prod_19','Ind_Prod_20','Ind_Prod_21',
												'Ind_Prod_22','Ind_Prod_23','Ind_Prod_24']])
    
	# Adiccion de 3 filas con los valores por defecto 0, 1 y 2 para que siempre existan las 3 casuisticas
	data.loc[len(data)] = [0 for n in range(len(data.columns))]
	data.loc[len(data)] = [1 for n in range(len(data.columns))]
	data.loc[len(data)] = [2 for n in range(len(data.columns))]

	# Aplicación del método one-hot encoding sobre el primer conjunto de categoricas
	data = pd.get_dummies(data,columns=col_list_categorized_variables)

	# Eliminacion de las filas 3 ultimas filas introducidas anteriormente
	data.drop(data.index[len(data) - 1], inplace=True)
	data.drop(data.index[len(data) - 1], inplace=True)
	data.drop(data.index[len(data) - 1], inplace=True)

	# Turno ahora de aplicar one-hot encoding sobre las otras dos variables categoricas que nos quedan
	col_list_categorized_variables = list(data[['Socio_Demo_02']])

	# Aplicación del método one-hot encoding sobre dichas variables
	data = pd.get_dummies(data,columns=col_list_categorized_variables)

# SALIDA DEL DATASET
	# Datos pre-procesados como salida a la llamada del método
	return data


# Se hacen las transformaciones definidas en el metodo sobre los datos del fichero de train
################################################ Añada su ruta para el fichero de TRAIN ################################################
train_data = preparedata('C:/Users/Carlos/Desktop/DATATHON REAL_DATA/Dataset_Salesforce_Predictive_Modelling_TRAIN.csv')

# Borrado de la variable ID_Customer por ser un identificador del cliente
train_data = train_data.drop(['ID_Customer'], axis=1)

# Obtención del objetivo al que se pretende llegar, en este caso el Poder_Adquisitivo
target_data = train_data['Poder_Adquisitivo']
# Se elimina el Poder_Adquisitivo de nuestro set de análisis
train_data = train_data.drop(['Poder_Adquisitivo'],axis=1)

# Seleccion del ALGORITMO a aplicar
regression_algorithm = MLPRegressor(hidden_layer_sizes=(34),max_iter=750)
# Aplicacion del algoritmo mediante cross validation
scores = cross_val_predict(regression_algorithm, train_data, target_data, cv=10)

# Obtención de las métricas: MSE y RMSE
print('MSE: %0.2f' % (mean_squared_error(target_data,scores)))
print('RMSE: %0.2f' % (sqrt(mean_squared_error(target_data,scores))))

# Aplicacion del algoritmo sobre todo el conjunto de datos
regression_algorithm.fit(train_data, target_data)

# Pre-procesamiento para los datos del fichero de test
################################################ Añada su ruta para el fichero de TEST ################################################
test_data = preparedata('C:/Users/Carlos/Desktop/DATATHON REAL_DATA/Dataset_Salesforce_Predictive_Modelling_TEST.csv')

# Extracción de la informacion de ID_Customer para construir el posterior resultado
test_mission = test_data[['ID_Customer']]
# Eliminacion de la variable ID_Customer por ser un identificador del cliente
test_data = test_data.drop(['ID_Customer'], axis=1)

# Se aplica el modelo obtenido del entrenamiento
test_prediction = regression_algorithm.predict(test_data)

# Creaccion del test_mission que contendra los ID_Customer y el poder adquisitivo asociado
test_mission['Poder_Adquisitivo'] = test_prediction

# Exportacion del resultado a csv
################################################ Añada su ruta para indicar donde guardar el fichero de predicción generado ################################################
test_mission.to_csv('C:/Users/Carlos/Desktop/DATATHON REAL_DATA/Python/final/Test_Mission.txt', index=False)

print(team)