# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 20:30:28 2023

@author: IVAN MARTINEZ BRAVO
"""

###############################################################################
################                                          #####################
################              Algoritmos knn              #####################
################                                          #####################
###############################################################################

#Este es un método de clasificación que consiste en detectar los puntos
#cercanos al punto nuevo y en función de las características de sus vecinos 
#asigna la clasificación al nuevo punto

#Este metodo es malo si tienes varias columnas categóricas
#No funciona para variables que no sean numéricas
#Es muy sensible a valores atípicos
import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *


#%%

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23")
mi_data = pd.read_csv("datos_peliculas.csv")

#%%

mi_data.head()
mi_data.columns
mi_data.shape

#Las fechas suelen considerarse como variable categorica discreta
#Entonces podemos eliminarla de nuestra data las secuelas porque no aporta mucha
#información para calcular el genero 


#Otra forma de seleccionar una columna es tabla["nombre de columna"]
mi_data["año"]

#Pido el numero de columnas de mi data
len(mi_data.columns)

#creamos la variable peliculas seleccionando su columna
peliculas = mi_data >> select(_.pelicula) #Variable objetivo
mi_data = mi_data >> select(-_.pelicula,-_.secuela) #Variable independiente

#%%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#%%
'''
CLASIFICACIÓN

Probamos KNN para clasificación; en concreto vamos a suponer que queremos 
predecir el género de una película en función de las otras columnas
'''
#Los ordeno para notar fácilmente qué numeros faltan
sorted(mi_data["genero"].unique())

mi_data.shape

#Selecciono mi variable objetivo
variable_objetivo_clasificacion = mi_data >> select(_.genero)
#Selecciono mis variable independientes
variables_independientes_clasificacion = mi_data >> select(-_.genero)
#Selecciono mis variables de entrenamiento y prueba y selecciono el 20% de los datos
X_train, X_test, y_train, y_test = train_test_split(
    variables_independientes_clasificacion,
    variable_objetivo_clasificacion, test_size=0.20,random_state=2023)

sorted(y_train["genero"].unique())

'''Utilizando pesos uniformes'''
#mando a llamar al clasificador diciendole que se fije en 10 vecinos y el 
#weights="uniform" es aquel enfoque en el que clasifica de acuerdo a los mas cercanos
clasificador_knn_uniforme = KNeighborsClassifier(n_neighbors=3, weights="uniform")
#Para ajustarlo le doy de comer la variable independiente de entrenamiento y la 
#variable objetivo de entrenamiento
clasificador_knn_uniforme.fit(X_train, y_train["genero"])
#Ahora predecimos aplicando el predict a las variables de prueba independientes 
preds_uniforme = clasificador_knn_uniforme.predict(X_test)
#Medimos que tan bueno es (el micro lo que hace es que tome encuenta que no es multivariable)
f1_score(y_test, preds_uniforme, average="micro")

'''Utilizando pesos = "distancias" '''
#Aqui el weights="distance " clasifica en funcón de la decisión
clasificador_knn_distancias = KNeighborsClassifier(n_neighbors=100, weights="distance")
clasificador_knn_distancias.fit(X_train, y_train["genero"])

preds_distancias = clasificador_knn_distancias.predict(X_test)
f1_score(y_test, preds_distancias, average="micro")

#%%
'''Selección de k'''

def clasificadores_knn(k):
    #ALas primeras dos lineas mandan a llamar el clasificacion segun el modelo
    knn_uniforme = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_distancias = KNeighborsClassifier(n_neighbors=k, weights="distance")
    #Los ajusto
    knn_uniforme.fit(X_train, y_train["genero"])
    knn_distancias.fit(X_train, y_train["genero"])
    #Los predizco
    preds_uniforme = knn_uniforme.predict(X_test)
    preds_distancias = knn_distancias.predict(X_test)
    #Calculo la metrica f1
    f1_uniforme = f1_score(y_test, preds_uniforme, average="micro")
    f1_distancias = f1_score(y_test, preds_distancias, average="micro")
    #Devuelveme k, y los f1 para cada modelo 
    return (k,f1_uniforme,f1_distancias)
#creame una lista donde contenga los clasificadores para cada k y su f1 en los dos metodos
#Elijo k impares para que el algoritmo no tenga empates a la hora de seleccionar
#cual es la clasificacion de mi nuevo dato en funcion de los k impares mas cercanos
clasificacion_evaluaciones =[ clasificadores_knn(k) for k in range(1,151,2)]

clasificacion_evaluaciones = pd.DataFrame(clasificacion_evaluaciones,
                                          columns = ["k","F1_uniforme","F1_distancias"])
#%%
#Reacomodamos la tabla, primero ponemos la columna F1_uniforme y abajo acomodamos 
#F1_distancias y le quitamos el k para que después del F1_distancias no me ponga las k debajo
#todo esto se hace con la función gather
#Este reacomodo se hace por comodidad de la interpeetación de la computador<
clasificaciones_evaluaciones_tidy = clasificacion_evaluaciones >> gather("F1_tipo",
                                                                         "F1_score",
                                                                         -_.k)

(ggplot(data = clasificaciones_evaluaciones_tidy) +
    geom_point(mapping=aes(x="k",y="F1_score",color="F1_tipo")) +
    geom_line(mapping=aes(x="k",y="F1_score",color="F1_tipo"))
)


(ggplot(data = clasificacion_evaluaciones) +
    geom_point(mapping=aes(x="k",y="F1_uniforme"),color = "red") +
    geom_line(mapping=aes(x="k",y="F1_uniforme"),color = "red") +
    geom_point(mapping=aes(x="k",y="F1_distancias"),color = "blue") +
    geom_line(mapping=aes(x="k",y="F1_distancias"),color = "blue")
)

#Raiz del numero de datos
mi_data.shape[0]**0.5

#Me meto en la tabla de clasificacion_evaluaciones y veo el F1 máximo entre los dos modelos
(clasificacion_evaluaciones >> 
    filter((_.F1_uniforme == _.F1_uniforme.max()) | (_.F1_distancias == _.F1_distancias.max()))
)

#%%

'''Utilizando pesos uniformes'''
#pongo el mejor clasificador de acuerdo al F1 maximo
mejor_clasificador_knn_uniforme = KNeighborsClassifier(n_neighbors=15, weights="uniform")
mejor_clasificador_knn_uniforme.fit(X_train, y_train["genero"])

mejor_preds_uniforme = mejor_clasificador_knn_uniforme.predict(X_test)
f1_score(y_test, mejor_preds_uniforme, average="micro")

'''Utilizando pesos = "distancias" '''
mejor_clasificador_knn_distancias = KNeighborsClassifier(n_neighbors=7, weights="distance")
mejor_clasificador_knn_distancias.fit(X_train, y_train["genero"])

mejor_preds_distancias = mejor_clasificador_knn_distancias.predict(X_test)
f1_score(y_test, mejor_preds_distancias, average="micro")

#%%