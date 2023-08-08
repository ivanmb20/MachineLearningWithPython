# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 20:04:11 2023

@author: IVAN MARTINEZ BRAVO
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 

#%%

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses\Machine_Learning23")
mi_data = pd.read_csv("cancer_mama.csv")

#%%

mi_data.shape

mi_data.columns

#%%
#La columna de diagnostico consta de 0 y 1
#El group_by agrupa en dos grupos para la columna de diagnosis
#Agrupará las filas que tengan 0 y 1
#summarize(conteo_objetivo=n(_)) lo que hace es contar cuantos renglones hay en 
#la subtabla para 0 y para 1
#porcentaje_objetivo=n(_) lo que hace es decirme que porcentaje de filas representan las filas del 0 y el 1
mi_data >> group_by(_.diagnosis) >> summarize(conteo_objetivo = n(_), porcentaje_objetivo = n(_)/569)

#En cuestiones de salud, el 0 representa la negacion del objetivo que se busca
#El 1 el objetivo, en este caso, 1 DEBERÍA ser cancer maligno y 0 benigno
#Pero en este caso ESTÁN AL REVÉS
#%%
#En la siguiente linea vamos a remediar este problema
#En esta linea cambiamos los valores de las columnas de cancer benigno a maligno
#con la funcion mutate nos permite cambiar los valores de columnas de un dataframe
mi_data = mi_data >> mutate(diagnosis = _.diagnosis.replace({0:1,1:0}))

#%%

from sklearn.linear_model import LinearRegression

#Aqui creamos una subtabla de mi_tabla que consistirá de todas las filas
#pero con la función select(_.,_.) le pedimos a python que solo tome las dos columnas que deseamos
data_peor_area = mi_data >> select(_.worst_area,_.diagnosis)

(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red")
 )

modelo_reg_lineal = LinearRegression()
#Selecciono quienes son mis X y Y en mi modelo
modelo_reg_lineal.fit(X=data_peor_area >> select(_.worst_area),
                      y=data_peor_area >> select(_.diagnosis))

#En esta linea le estoy pidiendo que a la subtabla data_peor_area le agregue
#la columna de predicciones
data_peor_area = data_peor_area >> mutate(predicciones_reg_lineal = modelo_reg_lineal.predict(data_peor_area >> select(_.worst_area)))

#Graficamos la regresión lineal 
(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red") +
    geom_line(mapping = aes(x="worst_area", y="predicciones_reg_lineal"),color="blue")
 )
#Al graficar vemos una linea recta, que no se ajusta a los valores de 0 y 1 de 
#la gráfica, esto es debido a que en el ajuste lineal queremos valores numericos
# de la alpha y beta
#pero en este caso tenemos un caso dicotómico, es decir que buscamos valores
#CATEGÓRICOS (que representan el 0 y el 1), para ellos usaremos la regresión LOGISTICA
#%%

from sklearn.linear_model import LogisticRegression

modelo_reg_logis = LogisticRegression() 
#En la sintaxis del .fit de la regresión logistica en lugar de escribirlo
#como en siuba, lo escribimos como Python clásico
#es decir, le ponemos corchete de la columna que nos interesa
modelo_reg_logis.fit(X=data_peor_area >> select(_.worst_area), y=data_peor_area["diagnosis"])

#Hasta este momento la tabla data_peor_area tiene tres columnas (worst area,diagnosis,predicciones)
#y ahora le vamos a agregar una cuarta columna que corresponderá a las probabilidades
# atravpes de la funcion modelo_reg_logis.predict_proba
#en la ultima parte, "[:,1]" nos dice que solo nos quedemos con la columna que representa los puntos malignos
data_peor_area = data_peor_area >> mutate(probabilidades_reg_logis = (modelo_reg_logis.predict_proba(data_peor_area >> select(_.worst_area)))[:,1])
                         

(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red") +
    geom_line(mapping = aes(x="worst_area", y="predicciones_reg_lineal"),color="blue") +
    geom_line(mapping = aes(x="worst_area", y="probabilidades_reg_logis"),color="darkgreen")
 )

data_peor_area = data_peor_area >> mutate(prediccion = modelo_reg_logis.predict(data_peor_area >> select(_.worst_area)))


data_peor_area
#%%

'''
###############################################################################
################                                          #####################
################           Regreseión logística           #####################
################                                          #####################
###############################################################################
'''


import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses\Machine_Learning23")
mi_data = pd.read_csv("cancer_mama.csv")

mi_data = mi_data >> mutate(diagnosis = _.diagnosis.replace({0:1,1:0}))

#%%

'''separación entrenamiento y prueba'''
#Ahora si vamos a tomar todas las columnas, no solo worst area como en lo anterior
#Tomamos las variables independientes
variables_independientes = mi_data >> select(-_.diagnosis)
#Seleccionamos la variable objetivo
objetivo = mi_data >> select(_.diagnosis)

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                  objetivo,
                                                                                                  test_size=0.3,
                                                                                                  random_state=42)

#Le digo que resolvedor quiero que use
#Aqui especifíco el resolvedor porque tengo filas>>columnas
modelo_rl = LogisticRegression(solver = "liblinear")
#el values.ravel lo que hace es convertir mi tabla objetivo_entrenamiento 398x1 a una lista
#esto debido a que la coordenada y del fit en la regresion lineal no acepta tablas
#objetivo_entrenamiento.shape
modelo_rl.fit(indepen_entrenamiento,objetivo_entrenamiento.values.ravel())

predicciones = modelo_rl.predict(indepen_prueba)
predicciones_probabilidades = modelo_rl.predict_proba(indepen_prueba)
objetivos_reales = objetivo_prueba.values.ravel()

#%%
#Esta funcion me va a devolver el valor del diagnóstico verdadero (el dado en la tabla)
#y el predicho por el modelo de la computadora 
def tupla_clase_prediccion(y_real, y_pred):
    return list(zip(y_real, y_pred)) #zip lo que hace es juntamelos en parejas y devuelvemelos en forma de lista

tupla_clase_prediccion(objetivos_reales, predicciones)[:20]


#%%¿Cómo se valida una regresión?

def VP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])

def VN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
    
def FP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])

def FN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==0])


print("""
Verdaderos Positivos: {}
Verdaderos Negativos: {}
Falsos Positivos: {}
Falsos Negativos: {}
""".format(
    VP(objetivos_reales, predicciones),
    VN(objetivos_reales, predicciones),
    FP(objetivos_reales, predicciones),
    FN(objetivos_reales, predicciones)    
))

#%%

'''
###############################################################################
################                                          #####################
################          Métricas de evaluación          #####################
################                                          #####################
###############################################################################
''' 


'''Exactitud (accuracy)'''
metrics.accuracy_score(objetivos_reales, predicciones)

'''Precisión'''

def precision(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fp = FP(clases_reales, predicciones)
    return vp / (vp+fp)

precision(objetivos_reales, predicciones)

'''Sensibilidad'''

metrics.recall_score(objetivos_reales, predicciones)

'''Puntuación F1'''

metrics.f1_score(objetivos_reales, predicciones)

#%%