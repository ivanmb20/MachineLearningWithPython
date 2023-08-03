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

ruta="C:/Users/IVAN MARTINEZ BRAVO/Desktop/RESPALDO IVAN 1 08 22/ESCRITORIOO/SciData Courses/Machine_Learning23"
os.chdir(ruta) #Con esta linea indicamos nuestro lugar de trabajo

mi_tabla=pd.read_csv("datos_regresion.csv")


