# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:20:56 2023

@author: IVAN MARTINEZ BRAVO
"""

'''
###############################################################################
################                                          #####################
################         Regresión lineal completa        #####################
################                                          #####################
###############################################################################
'''

import pandas as pd
import numpy as np
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#%%

import os
ruta = "C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses\Machine_Learning23"
os.chdir(ruta) #Quiero que te vayas a esta carpeta
mi_tabla = pd.read_csv("casas_boston.csv")


#%%
#Esta forma es para bajarla directo desde Github
ruta = "https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/casas_boston.csv"
mi_tabla_2 = pd.read_csv(ruta)

#%%
#La columna que queremos predecir es MEDV
#Al escribir select(-_.MEDV) le estoy diciendo "seleccioname todas las columnas excepto MEDV"
#Quitamos también RAD porque no nos da información numerica para el modelo
variables_independientes = mi_tabla >> select(-_.MEDV,-_.RAD)
#Si tuviera que quitar otras columnas escribiría una comma y las demás,select(-_.MEDV,-_.AGE) si es que AGE no tuviera valores numéricos
objetivo = mi_tabla >> select(_.MEDV)

#Ahora le pido las dimensiones de la tabla variables independientes
variables_independientes.shape
#%%

modelo_regresion = LinearRegression()
modelo_regresion.fit(X=variables_independientes,y=objetivo)
mi_tabla = mi_tabla >> mutate(predicciones = modelo_regresion.predict(variables_independientes)) >> select(-_.RAD)
'''
MUY IMPORTANTE
En este momento mi_tabla es la tabla original PERO CON UNA COLUMNA EXTRA: LA DE LAS PREDICCIONES DE LA COMPUTADORA
'''


#%%

'''Función para evaluar el modelo. Sus argumentos son:
    - independientes: tabla de columnas predictoras (es la tabla azul)
    - nombre_columna_objetivo: es el nombre de la columna objetivo de la tabla original
    - tabla_full: es la tabla completa del comentario anterior'''

def evaluar_regresion(independientes,nco,tabla_full):
    #n representa las filas correspondientes a las variables independientes (las filas son .shape[0])
    #k es el numero de columnas de las variables independientes (zona azul) y el indice .shape[1] representa que son las filas
    n = independientes.shape[0]
    k = independientes.shape[1]
    mae = metrics.mean_absolute_error(tabla_full[nco],tabla_full["predicciones"])
    rmse = np.sqrt(metrics.mean_squared_error(tabla_full[nco],tabla_full["predicciones"]))
    r2 = metrics.r2_score(tabla_full[nco],tabla_full["predicciones"])
    r2_adj = 1-(1-r2)*(n-1)/(n-k-1)
    return {"r2_adj":r2_adj,"mae":mae,"rmse":rmse}
    
#%%
    
evaluar_regresion(variables_independientes,"MEDV",mi_tabla)

#%%

'''
###############################################################################
################                                          #####################
################ SEPARACION EN ENTRENAMIENTO Y PRUEBA     #####################
################                                          #####################
###############################################################################
'''

from sklearn.model_selection import train_test_split

#%%

'''Dividiemos en entrenamiento y prueba. El 33% de los datos es para prueba y
utilizamos una semilla igual a 13'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=13)
# A la función train_test_split hay que darle de comer las variables independientes
# La variable objetivo ie la que queremos predecir
# test_size es el porcentaje de datos que queremos tomar
# random_state es la semilla o el estado aleatorio que va a tomar del 33% de los datos que tomó

#Dicha tabla 
#%%
indepen_entrenamiento.shape[0] #Este es el 66% de 506
objetivo_entrenamiento.shape[0] 
indepen_prueba.shape[0] #Este es el 33% de 506
objetivo_prueba.shape[0]

#%%
#Ahora con la función mutate uno la zona azul oscuro y naranja oscuro que representa la zona de entrenamiento del modelo
mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
#Hago lo mismo para la zona de prueba
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

#%%

modelo_entrenamiento = LinearRegression()

modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)

mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))

#%%
mi_tabla_entrenamiento.columns

evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)

#%%
#mutate siempr epega verticalmente
#
mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))
evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)

#%%
#Quiero que me cree un diccionario con los valores siguientes
Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)

Resultados = pd.DataFrame(Resultados)
Resultados

#%%
'''Cambiando random_state a 42'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=42)

mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

modelo_entrenamiento = LinearRegression()
modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)
mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))
mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))

Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)


Resultados = pd.DataFrame(Resultados)
Resultados

#Cuando cambiamos el state a 42 aparece que el rmse es menor en la prueba
#que el del entrenamiento, en un buen modelo eso está mal
# Es decir que la precisión del método depende de la partición que tome
#lo cual es muy arbitrario 
#%%
'''
###############################################################################
################                                          #####################
#########################   Validación cruzada      ###########################
################                                          #####################
###############################################################################
'''

from sklearn.model_selection import cross_val_score

#%%

modelo_regresion_validacion = LinearRegression()

#En esta parte simplemente selecciono la parte azul (juntando entrenamiento y prueba)
variables_independientes = mi_tabla >> select(-_.MEDV,-_.predicciones)
#En esta parte simplemente selecciono la parte naranja (juntando entrenamiento y prueba)
objetivo = mi_tabla >> select(_.MEDV)

cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error", #scoring es que medida de evalaución quiero tomar
                cv=10) #cv indica cuantas veces quiero tomar el % de mis datos (en este caso 10 veces tomaré el 33% cada una con su semilla inicial )

rmse_validacion = [-cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=x).mean() for x in range(10,100) 
]

#Hacer de tarea para 150

evaluacion_cruzada = {"particiones":list(range(10,100)),
                      "rmse_validacion":rmse_validacion}

evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)

(ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )
