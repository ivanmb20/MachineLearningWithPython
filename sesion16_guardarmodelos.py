# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:21:43 2023

@author: IVAN MARTINEZ BRAVO
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23")

mi_data_pixeles = pd.read_csv("mnist_pixeles.csv",header=None)
mi_data_clases = pd.read_csv("mnist_clases.csv",header=None)

#%%

mi_data_clases.shape
mi_data_pixeles.shape

#%%
#.iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
primer_digito = mi_data_pixeles.iloc[0]
#Me agarro la primera fila  en la linea de arriba
primer_digito.to_numpy()
#Le decimos que tome primer digito, lo acomode en tablas de 28x28 y el greys es el color que va a interpretar los numeros de las tablas
# en este caso, son grises
plt.imshow(primer_digito.to_numpy().reshape(28,28), cmap="Greys")
#plt.imshow(mi_data_pixeles.iloc[2534].to_numpy().reshape(28,28), cmap="Greys")
#la tabla de clases nos pide 
mi_data_clases.iloc[0]

#%%
'''Analizar balanceo'''
mi_data_clases.value_counts()*100/mi_data_clases.shape[0]

#%%
#Aplicaré componentes principales para reducir la dimensionalidad
from sklearn.decomposition import PCA
#El 0.8 significa que haga el analisis por componentes y que se quede con el 80%
#de la variabilidad original
pca = PCA(0.8)
#calculo con .fit
mnist_pca = pca.fit_transform(mi_data_pixeles)
#dime con cuantas columnas se quedo
mnist_pca.shape
#Se queda con 43 columnas, las cuales son las combinaciones de columnas de la original
#para quedarse con solo 43
#%%

from scipy.stats import randint as sp_randint
#Mando a llamar el clasificador
clf = KNeighborsClassifier()
#Haré una busqueda aleatoria, le pido numero de vecinos, el parámetro p y el parámetro de pesos
busqueda_dist_parametros = {
    "n_neighbors": sp_randint(2,10),#aqui le paso una funcion de distribucion de probabilidad
    "p": sp_randint(1,3),#Aqui tambien
    "weights": ["uniform", "distance"]
}
#"weights": ["uniform", "distance"] lo que dice es que el algoritmo es que él decida entre
#los dos metodos uniform y distance a la hora de seleccionar cual es el más óptimo de tomar
#sp_randint(numero, numero) lo que hace es generar numeros aleatorios entre 2 y 10 en este ejempl
from sklearn.model_selection import RandomizedSearchCV
#aqui hago una busqueda aleatoria
#le paso el estimador, le paso el diccionario donde tiene que buscar los números
#
busqueda = RandomizedSearchCV(estimator=clf,
                             param_distributions=busqueda_dist_parametros,
                             n_iter=3,#hazme el numero de iteraciones que vamos a usar
                             cv=3,#cv=3 es hazme 3 validaciones cruzadas
                             n_jobs=-1,#número de núcleos que quieres que utilice tu compu para hacer los cálculos
                             scoring="f1_micro") #para cada una hazme una metrica f1
                            #como mis clases están balanceadas, utilizo el micro en vez de macro
#Aqui ya le damos de comer el pca de 43 columnas y converimos los valores de y a array
busqueda.fit(X=mnist_pca, y=mi_data_clases.values.ravel())
#Dame el mejor puntaje y los valores de los parametros con los que se obtienen dicho puntaje
busqueda.best_score_
busqueda.best_params_

#%%

mejores_params = {'n_neighbors': 4, 'p': 2, 'weights': 'distance'}

mejor_knn = KNeighborsClassifier(**mejores_params)
mejor_knn.fit(mnist_pca, mi_data_clases.values.ravel())

#Ya reducimos la dimensionalidad

#%%

mi_numero = pd.read_csv("mi_numero.csv",header = None)
mi_numero.iloc[0].to_numpy()
plt.imshow(mi_numero.iloc[1].to_numpy().reshape(28,28), cmap="Greys")

#en el transform queda guardada la receta de las componentes 
nuevos_pca = pca.transform(mi_numero)
mejor_knn.predict(nuevos_pca)


#%% 
'''usar el modulo pickle para guardar y cargar'''
#pickle se utiliza para guardar grandes cantidades de datos
import pickle
#wb es binario
#en esas dos lineas es que me escriba l
with open("pca.pickle", "wb") as file:
    pickle.dump(pca, file)
    
with open("mejor_knn.pickle", "wb") as file:
    pickle.dump(mejor_knn, file)

#%%

import pickle
with open('pca.pickle', "rb") as file:
    mi_pca = pickle.load(file)
    
with open('mejor_knn.pickle', "rb") as file:
    mejor_knn = pickle.load(file)

#%%

nuevos_numeros = pd.read_csv("nuevos_numeros.csv",header = None)
nuevos_numeros_pca = mi_pca.transform(nuevos_numeros)
mejor_knn.predict(nuevos_numeros_pca)


plt.imshow(nuevos_numeros.iloc[0].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[1].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[2].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[3].to_numpy().reshape(28,28), cmap="Greys")