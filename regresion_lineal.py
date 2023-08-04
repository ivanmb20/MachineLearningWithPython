# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 21:45:39 2023

@author: IVAN MARTINEZ BRAVO
"""

import os
import pandas as pd
import numpy as np
from siuba import * #Esta paquetería adapta la sintaxis de R a Python
from plotnine import * 
#%%

ruta="C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23"
os.chdir(ruta) #Con esta linea indicamos nuestro lugar de trabajo

mi_tabla=pd.read_csv("datos_regresion.csv")
mi_tabla

mi_tabla.columns

#Al poner los parentesis en el inicio y el final al compilar indico que compile todo dentro del paréntesis
#En la primera parte le pido que mis datos (data) los saque de la mi_tabla
#Después del + le estoy diciendo cómo quiero que sea mi geometría (puntual)
#y dentro del mapping va toda la "aestética de la gráfica"
(ggplot(data = mi_tabla) +
geom_point(mapping=aes(x="caracteristica_1",y="valor_real"),color="red")
)
#%%
#Nos filtramos a través de la funcion select(._columna) para obtener la columna que queramos
variables_independientes = mi_tabla >>select(_.caracteristica_1)

objetivo = mi_tabla>>select(_.valor_real)
#Si quisieramos seleccionar muchas más columnas entonces la sintaxis sería
#mi_tabla >>select(_.col1,_.col2,...,_.coln)


#%%
from sklearn.linear_model import LinearRegression

#Preparate porque vas a hacer una regresión lineal
modelo=LinearRegression()

modelo.fit(X=variables_independientes,y=objetivo)

modelo.intercept_ #Esta es la alpha de la regresión

modelo.coef_ #Estas son las b's de la regresión

#%%

mi_tabla["predicciones"]=modelo.predict(variables_independientes)

(ggplot(data = mi_tabla) +
geom_point(mapping=aes(x="caracteristica_1",y="valor_real"),color="blue") +
geom_point(mapping=aes(x="caracteristica_1",y="predicciones"),color="red")+
geom_abline(slope=1.85,intercept=5.711)+
geom_smooth(mapping=aes(x="caracteristica_1",y="valor_real"),color="green")
)

#%%
from sklearn import metrics

#Con esto hallamos el error absoluto medio
metrics.mean_absolute_error(mi_tabla["valor_real"],mi_tabla["predicciones"])
#Con esto hallamos el  error cuadrático medio
metrics.mean_squared_error(mi_tabla["valor_real"],mi_tabla["predicciones"])
#Hallamos la raiz del error cuadratico medio
np.sqrt(metrics.mean_squared_error(mi_tabla["valor_real"],mi_tabla["predicciones"])
)

mi_tabla >>mutate(error=_.valor_real-_.predicciones)

R2 = metrics.r2_score(mi_tabla["valor_real"],mi_tabla["predicciones"])

1-(1-R2)*(50-1)/(50-1-1)