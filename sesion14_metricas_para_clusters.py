# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 20:34:47 2023

@author: IVAN MARTINEZ BRAVO
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import *
from plotnine import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23")

mi_data = pd.read_csv("datos_iris.csv")
#Solo me quedo con las columnas de Sepal
mi_data = mi_data >> select(_.startswith("Sepal")) 

#%%

(ggplot(data = mi_data) +
    geom_point(mapping=aes(x="Sepal_Length",y="Sepal_Width"))
)

#%%

from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples

mi_data = pd.read_csv("datos_iris.csv")
mi_data = mi_data >> select(_.startswith("Sepal")) 
#Elijo un escalador
escalador = preprocessing.normalize(mi_data)
mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=mi_data.index, 
                                      columns=mi_data.columns)
#Utilizamos el método de k medias para cluterizar 
k_medias = KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
#calculamos
k_medias.fit(mi_data_normalizado_df)
#Le pido que me de las etiquetas
Etiquetas = k_medias.labels_

#%%
#Calculamos la silueta total
silhouette_score(mi_data_normalizado_df,Etiquetas)

silhouette_samples(mi_data_normalizado_df,Etiquetas)

#%%
#Agrego una columna en la que agregue las siluetas a mi tabla de dato original
(mi_data >> mutate(siluetas = silhouette_samples(mi_data_normalizado_df,Etiquetas),
                  etiquetas = Etiquetas.astype(str)) >>
    ggplot() +
        geom_point(mapping=aes(x="Sepal_Length",y="Sepal_Width",color = "siluetas",shape="etiquetas"))
)

#%%
#Construimos una funcion 
def constructor_clusters(data,k):
    #aplico un escalador a la tabla original
    escalador = preprocessing.normalize(data)
    #lo transformo a dataframe
    mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=data.index, 
                                      columns=data.columns)
    #Calculamos k medias
    k_medias = KMeans(n_clusters = k ,init='k-means++')
    #Aplicamos k medias
    k_medias.fit(mi_data_normalizado_df)
    #Le pedimos las etiquetas
    Etiquetas = k_medias.labels_
    #Calculamos que tan precisa es la silueta
    silueta = silhouette_score(mi_data_normalizado_df,Etiquetas)
    #Aplicamos el criterio de Calinski para calcular los clusters
    #Este algoritmo utiliza el concepto de densidad
    cal_har = calinski_harabasz_score(mi_data_normalizado_df,Etiquetas)
    
    return k, Etiquetas, silueta, cal_har 

#%%
#Ejecutamos la funcion para nuestros clusters
constructor_clusters(mi_data,4)

#%%

modelos_kmedias = [constructor_clusters(mi_data,k) for k in range(2,10)]

#%%
#Aqui lo que le pido es que de modelos_kmedias es que me de las entradas 0,2 y 3
resultados = pd.DataFrame([(x[0],x[2],x[3]) for x in modelos_kmedias],
             columns = ["k","silueta","calinski_harabasz"])

#%%
#Graficamos el k vs silueta para ver que tan buena es la silueta 
(ggplot(data = resultados) +
    geom_point(mapping = aes(x="k",y="silueta"),color = "red") +
    geom_line(mapping = aes(x="k",y="silueta"),color = "red") 
)

(ggplot(data = resultados) +
geom_point(mapping = aes(x="k",y="calinski_harabasz"),color = "red") +
geom_line(mapping = aes(x="k",y="calinski_harabasz"),color = "red")
)

#%%
#Ahora visualizo el elbow del metodo Kmeans con siluetta
modelos = KMeans()

visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "silhouette")
visualizer.fit(mi_data_normalizado_df)
visualizer.show()

#%%
#Ahora visualizo el elbow del metodo Kmeans con calinski
modelos = KMeans()

visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "calinski_harabasz")
visualizer.fit(mi_data_normalizado_df)
visualizer.show()