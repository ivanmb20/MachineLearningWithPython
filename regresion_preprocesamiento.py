# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:59:00 2023

@author: IVAN MARTINEZ BRAVO
"""

'''
###############################################################################
################                                          #####################
################             Preprocesamiento             #####################
################                                          #####################
###############################################################################
'''


import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *

#%%
#¿Qué es el preprocesamiento?
#Es cuando limpias tu tabla para que ya se pueda trabajar con los modelos
os.chdir("C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23")
mi_data = pd.read_csv("datos_procesamiento.csv")

#%%

mi_data.head()
mi_data.columns
mi_data.shape

#%%
#Las variables numericas son las mas fáciles de trabajar
'''
###############################################################################
################                                          #####################
################           Variables numéricas            #####################
################                                          #####################
###############################################################################
'''
#%%
'''Datos faltantes'''
#Importo un imputador
from sklearn.impute import SimpleImputer

#Dame aquella parte de la tabla donde los datos son numericas
var_numericas_df = mi_data.select_dtypes([int, float])
var_numericas_df.columns

#Devuelveme aquellas columnas que tengan valores faltantes
var_numericas_df[var_numericas_df.isnull().any(axis=1)]
#Si lo quisiera para filas cambiamos axis=0

#Si yo quiero asignar al valor faltante una variable le cambio missing_values=x siendo x la variable que escojo
#En strategy="mean" lo que le estamos indicando es que para una columna que contenga
#un valor faltante promediemos sus demas valores y ese promedio lo pongamos
#en esa casilla donde se encuwentra el valor faltante
imputador = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean")
#Ejecuto el imputador
var_numericas_imputadas = imputador.fit_transform(var_numericas_df)

#Lo transformamos a un dataframe
var_numericas_imputadas_df = pd.DataFrame(var_numericas_imputadas,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_df[var_numericas_imputadas_df.isnull().any(axis=1)]
#Ahora si me arroja que el dataframe es vacio en el sentido de que ya no hay valores faltantes
#%%
'''Estandarización'''
""
#Uno puede tener mucha variación en los datos y aqui es donde entra la estandarización
#La estandarización  es restar la media y divides entre la desviación
#Se recomienda usar este método cuando se tienen varianzas altas
#No existe una regla general, en la practica lo que uno hace es
#estandarizar con el modelo y veo las métricas y entonces decido si mis
#metricas mejoran al estandarizar pues lo aplico si no no tiene caso
var_numericas_df.columns

var_numericas_df.mean()
var_numericas_df.std()

#Mandamos a llamar de sklearn a preprocesamiento
from sklearn import preprocessing

#Aqui nombro al escalador como la funcion StandardScaler de sklearn
escalador = preprocessing.StandardScaler() #Lo mando a llamar
#Aqui lo aplico
var_numericas_imputadas_escalado_standard = escalador.fit_transform(var_numericas_imputadas_df)

escalador.mean_
np.sqrt(escalador.var_)

var_numericas_imputadas_escalado_standard.mean(axis=0)
var_numericas_imputadas_escalado_standard.std(axis=0)

var_numericas_imputadas_escalado_standard_df = pd.DataFrame(var_numericas_imputadas_escalado_standard,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_standard_df

#%%

'''Escalado robusto'''

'''
Para aquellos casos en los que los datos tengan muchos valores extremos (atípicos), es 
posible que estandarizar usando la media y la desviacion estandar no funcione 
bien en el modelo. Para esos casos es mejor usar unos estimadores mas robustos 
(menos sensibles a outliers) y emplear un RobustScaler que funciona substrayendo 
la mediana y escalando mediante el rango intercuartil (IQR).
'''
#El rango intercuartil es el cuartil entre 1 y 2
#Es decir escalado=(dato-mediana)/IQR
var_numericas_imputadas_df.columns

var_numericas_imputadas_df.mean(axis=0)


(ggplot(data = var_numericas_imputadas_df) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
#Mando a llamar al escalador robusto
escalador_robusto = preprocessing.RobustScaler()
#Lo aplico
var_numericas_imputadas_escalado_robusto = escalador_robusto.fit_transform(
                                                        var_numericas_imputadas_df)

var_numericas_imputadas_escalado_robusto.mean(axis=0)
var_numericas_imputadas_escalado_robusto.std(axis=0)
#Construyo la tabla 
var_numericas_imputadas_escalado_robusto_df = pd.DataFrame(var_numericas_imputadas_escalado_robusto,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_robusto_df.mean(axis=0)


(ggplot(data = var_numericas_imputadas_escalado_robusto_df) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
#%%
'''Escalado a un rango arbitrario'''

'''
Hay casos en los que en vez de estandardizar queremos escalar los datos a un 
rango (generalmente [-1,1] o [0,1]). Para ello podemos usar MinMaxScaler que 
hace escalado minmax (obviamente) o MaxAbscaler que simplemente divide cada valor 
de una variable por su valor máximo (y por tanto convierte el valor maximo a 1).
'''

var_numericas_imputadas_df.min()
var_numericas_imputadas_df.max()

'''
########################################## Escalador minmax
'''
escalador_minmax = preprocessing.MinMaxScaler()
var_numericas_imputadas_escalado_minmax = escalador_minmax.fit_transform(var_numericas_imputadas_df)

var_numericas_imputadas_escalado_minmax_df = pd.DataFrame(var_numericas_imputadas_escalado_minmax,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_minmax_df.min()
var_numericas_imputadas_escalado_minmax_df.max()

'''
########################################## Escalador maxabs
'''

escalador_maxabs = preprocessing.MaxAbsScaler()
var_numericas_imputadas_escalado_maxabs = escalador_maxabs.fit_transform(var_numericas_imputadas_df)

var_numericas_imputadas_escalado_minmax_df = pd.DataFrame(var_numericas_imputadas_escalado_maxabs,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_maxabs.max()


#%%

'''
###############################################################################
################                                          #####################
################           Variables categóricas          #####################
################                                          #####################
###############################################################################
'''
#%%

'''
Los modelos están diseñados para trabajar con variables numéricas.
Esto implica que para poder entrenar los modelos con variables categóricas
tenemos que convertirlas a números. Este proceso se llama codificación (encoding)
'''

mi_data.head()

var_categoricas = mi_data >> select(_.col_categorica,_.col_ordinal)

var_categoricas.head(10)

var_categoricas >> n_distinct(_.col_ordinal)

var_categoricas["col_ordinal"].unique()

