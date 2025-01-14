# Instalación de Pandas
#!pip install pandas
#!pip install pyarrow

import pandas as pd

# Creacion de un objeto Series
s = pd.Series([2, 4, 6, 8, 10])
print(s)                                
'''
0     2
1     4
2     6
3     8
4    10
dtype: int64
'''

# Creación de un objeto Series inicializándolo con un diccionario de Python
altura = {"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}
s = pd.Series(altura)
print(s)
'''
Santiago    187
Pedro       178
Julia       170
Ana         165
dtype: int64
'''


# Creación de un objeto Series inicializándolo con algunos 
# de los elementos de un diccionario de Python
altura = {"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}
s = pd.Series(altura, index = ["Pedro", "Julia"])
print(s)
'''
Pedro    178
Julia    170
dtype: int64
'''


# Creación de un objeto Series inicializandolo con un escalar
s = pd.Series(34, ["test1", "test2", "test3"])
print(s)
'''
test1    34
test2    34
test3    34
dtype: int64
'''

# Creación de un objeto Series
s = pd.Series([2, 4, 6, 8], index=["num1", "num2", "num3", "num4"])
print(s)
'''
num1    2
num2    4
num3    6
num4    8
dtype: int64
'''


s2 = s.copy()
print(s2)
'''
num1    2
num2    4
num3    6
num4    8
dtype: int64
'''


s2.iloc[1] = 133
print(s2)
print(s)
'''
num1      2
num2    133
num3      6
num4      8
dtype: int64
num1    2
num2    4
num3    6
num4    8
dtype: int64
'''

s["num3"]           #6
s[2]                #6
s.loc["num3"]       #6
s.iloc[2]           #6
s.iloc[2:4]         
'''
num3    6
num4    8
dtype: int64
'''


# Creacion de un objeto Series
s = pd.Series([2, 4, 6, 8, 10])
print(s)
'''
0     2
1     4
2     6
3     8
4    10
dtype: int64
'''

# Los objeto Series son similares y compatibles con los Arrays de Numpy
import numpy as np
# Ufunc de Numpy para sumar los elementos de un Array
np.sum(s)                   #30

# El resto de operaciones aritméticas de Numpy sobre Arrays también son posibles
# Más información al respecto en la Introducción a Numpy
s * 2       
'''
0     4
1     8
2    12
3    16
4    20
dtype: int64
'''

# Creación de un objeto Series denominado Temperaturas
temperaturas = [4.4, 5.1, 6.1, 6.2, 6.1, 6.1, 5.7, 5.2, 4.7, 4.1, 3.9]
s = pd.Series(temperaturas, name="Temperaturas")
'''
0     4.4
1     5.1
2     6.1
3     6.2
4     6.1
5     6.1
6     5.7
7     5.2
8     4.7
9     4.1
10    3.9
Name: Temperaturas, dtype: float64
'''

# Representación gráfica del objeto Series
#%matplotlib inline
import matplotlib.pyplot as plt
s.plot()
plt.show()


# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
    	    peso	altura	hijos
Ana	        56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''


# Creación de un DataFrame inicializándolo con algunos elementos de un diccionario
# de objetos Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas, columns = ["altura", "peso"], index = ["Ana", "Julia", "Santiago"])
'''
    	    altura	peso
Ana	        165	    56
Julia	    170	    64
Santiago	187	    84
'''


# Creación de un DataFrame inicializándolo con una lista de listas de Python
# Importante: Deben especificarse las columnas e indices por separado
valores = [
    [185, 4, 76],
    [170, 0, 65],
    [190, 1, 89]
]
df = pd.DataFrame(valores, columns = ["altura", "hijos", "peso"], index = ["Pedro", "Ana", "Juan"])
'''
        altura	hijos	peso
Pedro	185	    4	    76
Ana	    170	    0	    65
Juan	190	    1	    89
'''


# Creación de un DataFrame inicializándolo con un diccionario de Python
personas = {
    "altura": {"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}, 
    "peso": {"Santiago": 87, "Pedro": 78, "Julia": 70, "Ana": 65}}
df = pd.DataFrame(personas)
'''
	        altura	peso
Santiago	187	    87
Pedro	    178	    78
Julia	    170	    70
Ana	        165	    65
'''


# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
            peso	altura	hijos
Ana         56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''


df["peso"]
'''
Ana         56
Julia       64
Pedro       90
Santiago    84
Name: peso, dtype: int64
'''


df[["peso", "altura"]]
'''
        	peso	altura
Ana	        56	    165
Julia	    64	    170
Pedro	    90	    178
Santiago	84	    187
'''


# Pueden combinarse los metodos anteriores con expresiones booleanas
df["peso"] > 80
'''
Ana         False
Julia       False
Pedro        True
Santiago     True
Name: peso, dtype: bool
'''

# Pueden combinarse los metodos anteriores con expresiones booleanas
df[df["peso"] > 80]
'''
            peso	altura	hijos
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''


df.loc["Pedro"]
'''
peso       90.0
altura    178.0
hijos       2.0
Name: Pedro, dtype: float64
'''

df.iloc[2]
'''
peso       90.0
altura    178.0
hijos       2.0
Name: Pedro, dtype: float64
'''

df.iloc[1:3]
'''
peso	altura	hijos
Julia	64	170	3.0
Pedro	90	178	2.0
'''

df.loc["Julia"]
'''
peso       64.0
altura    170.0
hijos       3.0
Name: Julia, dtype: float64
'''

df.query("altura >= 170 and peso > 60")
'''
            peso	altura	hijos
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''

# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
            peso	altura	hijos
Ana	        56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''


# Copia del DataFrame df en df_copy
# Importante: Al modificar un elemento de df_copy no se modifica df
df_copy = df.copy()

# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
            peso	altura	hijos
Ana	        56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''

# Añadir una nueva columna al DataFrame
df["cumpleaños"] = [1990, 1987, 1980, 1994]
'''
            peso	altura	hijos	cumpleaños
Ana	        56	    165	    NaN	    1990
Julia	    64	    170	    3.0	    1987
Pedro	    90	    178	    2.0	    1980
Santiago	84	    187	    NaN	    1994
'''

# Añadir una nueva columna calculada al DataFrame
df["años"] = 2020 - df["cumpleaños"]
'''
	        peso	altura	hijos	cumpleaños	años
Ana	        56	    165	    NaN	    1990	    30
Julia	    64	    170	    3.0	    1987	    33
Pedro	    90	    178	    2.0	    1980	    40
Santiago	84	    187	    NaN	    1994	    26
'''

# Añadir una nueva columna creando un DataFrame nuevo
df_mod = df.assign(mascotas = [1, 3, 0, 0])
'''
            peso	altura	hijos	cumpleaños	años	mascotas
Ana	        56	    165	    NaN	    1990	    30	    1
Julia	    64	    170	    3.0	    1987	    33	    3
Pedro	    90	    178	    2.0	    1980	    40	    0
Santiago	84	    187	    NaN	    1994	    26	    0
'''

del df["peso"]
'''
            altura	hijos	cumpleaños	años
Ana	        165	    NaN	    1990	    30
Julia	    170	    3.0	    1987	    33
Pedro	    178	    2.0	    1980	    40
Santiago	187	    NaN	    1994	    26
'''

# Eliminar una columna existente devolviendo una copia del DataFrame resultante
df_mod = df.drop(["hijos"], axis=1)
'''
            altura	cumpleaños	años
Ana	        165	    1990	    30
Julia	    170	    1987	    33
Pedro	    178	    1980	    40
Santiago	187	    1994	    26
'''


# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
            peso	altura	hijos
Ana	        56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''

# Evaluar una función sobre una columna del DataFrame
df.eval("altura / 2")
'''
Ana         82.5
Julia       85.0
Pedro       89.0
Santiago    93.5
Name: altura, dtype: float64
'''

# Asignar el valor resultante como una nueva columna
df.eval("media_altura = altura / 2", inplace=True)
'''
            peso	altura	hijos	media_altura
Ana	        56	    165	    NaN	    82.5
Julia	    64	    170	    3.0	    85.0
Pedro	    90	    178	    2.0	    89.0
Santiago	84	    187	    NaN	    93.5
'''

# Evaluar una función utilizando una variable local
max_altura = 180
df.eval("altura > @max_altura")
'''
Ana         False
Julia       False
Pedro       False
Santiago     True
Name: altura, dtype: bool
'''

# Aplicar una función externa a una columna del DataFrame
def func(x):
    return x + 2
df["peso"].apply(func)
'''
Ana         58
Julia       66
Pedro       92
Santiago    86
Name: peso, dtype: int64
'''
'''
            peso	altura	hijos	media_altura
Ana	        56	    165	    NaN	    82.5
Julia	    64	    170	    3.0	    85.0
Pedro	    90	    178	    2.0	    89.0
Santiago	84	    187	    NaN	    93.5
'''


# Creación de un DataFrame inicializándolo con un diccionario de objetios Series
personas = {
    "peso": pd.Series([84, 90, 56, 64], ["Santiago","Pedro", "Ana", "Julia"]),
    "altura": pd.Series({"Santiago": 187, "Pedro": 178, "Julia": 170, "Ana": 165}),
    "hijos": pd.Series([2, 3], ["Pedro", "Julia"])
}
df = pd.DataFrame(personas)
'''
            peso	altura	hijos
Ana	        56	    165	    NaN
Julia	    64	    170	    3.0
Pedro	    90	    178	    2.0
Santiago	84	    187	    NaN
'''
# Guardar el DataFrame como CSV, HTML y JSON
df.to_csv("df_personas.csv")
df.to_html("df_personas.html")
df.to_json("df_personas.json")

# Cargar el DataFrame en Jupyter
df2 = pd.read_csv("df_personas.csv")

# Cargar el DataFrame con la primera columna correctamente asignada
df2 = pd.read_csv("df_personas.csv", index_col=0)