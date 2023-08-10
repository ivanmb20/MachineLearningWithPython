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
from plotnine import *
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
#modelo_reg_logis.predict_proba(data_peor_area>> select(_.worst_area)) es una tabla de dos columnas
#la primera columnas [:,0] es la probabilidad de que esté en 0 (cancer benigno)
#la segunda [:,1] es la probabilidad de pertenecer a 1 (cancer maligno)

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
#Siempre que sea binaria una clasificacion hay que usar regresión logística

#Verdadero positivos son los 1 que clasificamos bien
#Le doy de comer las tablas
def VP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])
#Verdadero negativo son los 0 que clasificamos bien
def VN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
#Es un falso 1    
def FP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])
#Es un falso 0
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

####### Exactitud =(VP+VN)/(VP+VN+FP+FN) ##### 
'''Exactitud (accuracy)'''
metrics.accuracy_score(objetivos_reales, predicciones)
#La exactitud es una medida general de como se comporta el modelo, 
#mide simplemente el porcentaje de casos que se han clasificado correctamente.


###### Precisión = VP/(VP+FP)
# La precisión indica la habilidad del modelo para clasificar como positivos los casos que son positivos.

'''Precisión'''
def precision(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fp = FP(clases_reales, predicciones)
    return vp / (vp+fp)

precision(objetivos_reales, predicciones)

####### Sensibilidad = (VP)/(VP+FN)
'''Sensibilidad'''
#La sensibilidad nos da una medida de la habilidad del modelo para encontrar todos los casos positivos. 
#La sensibilidad se mide en función de una clase.
metrics.recall_score(objetivos_reales, predicciones)

'''Puntuación F1'''
######## La puntuación F1 es una media ponderada entre la sensibilidad (que intenta obtener cuantos
#  mas verdaderos positivos independientemente de los falsos positivos) y la precisión (que intenta
# obtener solo verdaderos positivos que sean casos claros para limitar los falsos positivos).
metrics.f1_score(objetivos_reales, predicciones)

#El umbral lo vas a seleccionar a partir de la metrica que hayas elegido para el modelo
#%%

pd.DataFrame({"exactitud":[metrics.accuracy_score(objetivos_reales, predicciones)],
 "precision":[precision(objetivos_reales, predicciones)],
 "sensibilidad":[metrics.recall_score(objetivos_reales, predicciones)],
 "F1":[metrics.f1_score(objetivos_reales, predicciones)]
})


#%%
#Aqui yo puedo cambiar el umbral a mi gusto
def proba_a_etiqueta(predicciones_probabilidades,umbral=0.5):
    predicciones = np.zeros([len(predicciones_probabilidades), ])
    predicciones[predicciones_probabilidades[:,1]>=umbral] = 1
    return predicciones
#Cuando no das el umbral, automaticamente se entiende que es 0.5
proba_a_etiqueta(predicciones_probabilidades)


#%%

def evaluar_umbral(umbral):
    predicciones_en_umbral = proba_a_etiqueta(predicciones_probabilidades, umbral)
    precision_umbral = precision(objetivos_reales, predicciones_en_umbral)
    sensibilidad_umbral = metrics.recall_score(objetivos_reales, predicciones_en_umbral)
    F1_umbral = metrics.f1_score(objetivos_reales, predicciones_en_umbral)
    return (umbral,precision_umbral, sensibilidad_umbral, F1_umbral)

#%%
umbrales = np.linspace(0., 1., 1000)
#Calculamos los umbrales para cada punto en umbrales
evaluaciones = pd.DataFrame([evaluar_umbral(x) for x in umbrales],
                            columns = ["umbral","precision","sensibilidad","F1"])

#%%

(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="sensibilidad",y="precision",color="umbral"),size=0.1)
)
#El criterio F1 es mas usado porque el F1 te combina la sensibilidad como la precisión
(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="umbral",y="F1"),size=0.1)
)
#filter lo que hace es devolverme filas
#Devuelveme las filas en donde F1 alcance su valor máximo
evaluaciones >> filter(_.F1 == _.F1.max())