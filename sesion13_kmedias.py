# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:16:15 2023

@author: IVAN MARTINEZ BRAVO
"""

'''
###############################################################################
################                                          #####################
################                k medias                  #####################
################                                          #####################
###############################################################################
'''

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
import plotly.express as px
from plotly.offline import plot

from sklearn import preprocessing
#Este es para hacer la gráfica de codo
from yellowbrick.cluster import KElbowVisualizer

#%%

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23")
mi_data = pd.read_csv("datos_clientes.csv")

#%%
#Remplazamos la columna categorica de género por una numérica asignando 0->mujer y 1 ->hombre
mi_data = mi_data >> select(-_.Id_cliente) >> mutate(Genero = _.Genero.replace({"Female":1,"Male":0})) 

#Es necesario reescalar los datos, se puede hacer con cualquier método, en este caso elegimos normalizados
#Siempre que tengamos un algoritmo basado en distancias aplicamos un reescalado
escalador = preprocessing.normalize(mi_data)
mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=mi_data.index, 
                                      columns=mi_data.columns)

#%%

from sklearn.cluster import KMeans
#Nombramos al método
modelos = KMeans()
#Nombramos 
visualizer = KElbowVisualizer(modelos, k=(1,12))
visualizer.fit(mi_data_normalizado_df)
visualizer.show()

#%%
#Aplicamos el método
#n_cluster es el numero de clusters que quieres hacer
#El innit sirve para mejorar los mejores centroides iniciales, tambien está el
#random que toma los centroides aleatoriamente, o bien darle un arreglo con los puntos que
#representen tus centroides iniciales de acuerdo a como tu quieras
#n_init es el numero de veces que va a tomar los centroides y calcula la inercia
#y eso lo repite el numero n_init que tú digas y al final se queda con el mejor
#max_iter es el numero maximo de iteraciones que tú vas a permitir
#tol es la tolerancia es la distancia entre centros nuevos y antiguos como 
#parametro para parar al algoritmo
k_medias = KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
k_medias.fit(mi_data_normalizado_df)
#Aqui le preguntamos cuales son los clusters que hizo
k_medias.labels_
#Aqui nos da las coordenadas de los centros de cada cluster
k_medias.cluster_centers_
#k_medias.predict()

'''
parámetros:
    n_clusters: número de clusters
    init: 'k-means++', 'random' o arreglo de tamaño (n_clusters,n_características)
    n_init: 'auto' o entero. Para correr diferentes inicializaciones (1 si init=k-means++ o 10 si 
                                                                      init = random o arreglo)
    max_iter: entero (default = 300); máximo número de iteraciones    
'''

#A mi tabla original le añado las etiquetas pensadas como strings para que me la tome comom variable discreta
mi_data = mi_data >> mutate(Etiquetas = k_medias.labels_.astype(str))

fig = px.scatter_3d(mi_data, x='Edad', y='Puntuacion_gasto', z='Ingreso_anual',
              color='Etiquetas')
plot(fig)