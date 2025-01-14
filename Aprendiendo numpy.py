# Instalación de Numpy y Matplotlib
#pip install numpy
#pip install matplotlib

import numpy as np

# Array cuyos valores son todos 0
a = np.zeros((2, 4))
'''Output
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.]])
'''
a.shape     #Devuelve las dimensiones del array (2,4)
a.ndim      #Numero de dimension en este caso 2
a.size      #Devuelve 8 que seria el numero de elementos en su interior

np.zeros((2, 3, 4))
'''Output
array([[[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]],

       [[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])
'''

np.ones((2, 3, 4))
'''
array([[[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]],

       [[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]])
'''

np.full((2, 3, 4), 8)
'''
array([[[8, 8, 8, 8],
        [8, 8, 8, 8],
        [8, 8, 8, 8]],

       [[8, 8, 8, 8],
        [8, 8, 8, 8],
        [8, 8, 8, 8]]])
'''

np.empty((2, 3, 9))         #Guarda valores basura
'''
array([[[1.21533711e-311, 1.21531025e-311, 2.05833592e-312,
         6.79038654e-313, 2.48273508e-312, 2.35541533e-312,
         6.79038654e-313, 2.05833592e-312, 2.35541533e-312],
        [2.14321575e-312, 6.79038654e-313, 2.35541533e-312,
         6.79038654e-313, 2.35541533e-312, 2.35541533e-312,
         6.79038654e-313, 2.29175545e-312, 2.50395503e-312],
        [2.29175545e-312, 2.41907520e-312, 2.22809558e-312,
         2.12199579e-312, 2.10077583e-312, 2.12199579e-312,
         6.79038654e-313, 2.35541533e-312, 2.35541533e-312]],

       [[2.44029516e-312, 2.18565567e-312, 2.33419537e-312,
         2.35541533e-312, 2.37663529e-312, 2.41907520e-312,
         2.31297541e-312, 2.46151512e-312, 2.35541533e-312],
        [2.12199579e-312, 6.79038654e-313, 2.05833592e-312,
         2.16443571e-312, 2.33419537e-312, 2.22809558e-312,
         2.33419537e-312, 2.33419537e-312, 9.76118064e-313],
        [2.48273508e-312, 2.29175545e-312, 8.48798317e-313,
         9.33678148e-313, 1.08221785e-312, 6.79038653e-313,
         8.70018275e-313, 6.79038653e-313, 8.70018275e-313]]])
'''

# Creación del array utilizando una función basada en rangos
# (minimo, maximo, número elementos del array)
print(np.linspace(0, 6, 10))
'''
[0, 0.66666667, 1.33333333, 2, 2.66666667, 3.33333333, 4, 4.66666667, 5.33333333, 6]
'''

# Inicialización del array con valores aleatorios
np.random.rand(2, 3, 4)
'''
array([[[0.4803661 , 0.04791604, 0.6585326 , 0.26318487],
        [0.49215558, 0.91367386, 0.4091404 , 0.50591823],
        [0.16773816, 0.60664607, 0.34644378, 0.46572176]],

       [[0.4716299 , 0.92467181, 0.2884385 , 0.80479927],
        [0.92523426, 0.64093541, 0.84163116, 0.37160562],
        [0.12448548, 0.93544157, 0.25712712, 0.20475179]]])
'''

# Inicialización del array con valores aleatorios conforme a una distribución normal
np.random.randn(2, 4)
'''
array([[ 0.43861388, -1.2510498 ,  0.68775812,  0.60112712],
       [ 0.37631383, -0.24095791, -0.0918386 ,  0.64442362]])
'''

#Mostrar la grafica de valores
#%matplotlib inline
import matplotlib.pyplot as plt
c = np.random.randn(1000000)
plt.hist(c, bins=200)
plt.show()


# Inicialización del Array utilizando una función personalizada
def func(x, y):
    return x + 2 * y

np.fromfunction(func, (3, 5))
'''
array([[ 0.,  2.,  4.,  6.,  8.],
       [ 1.,  3.,  5.,  7.,  9.],
       [ 2.,  4.,  6.,  8., 10.]])
'''


# Creación de un Array unidimensional
array_uni = np.array([1, 3, 5, 7, 9, 11])
print("Shape:", array_uni.shape)    #Shape: (6,)    
print("Array_uni:", array_uni)      #Array_uni: [ 1  3  5  7  9 11]
# Accediendo al quinto elemento del Array
array_uni[4]            #9
# Accediendo al tercer y cuarto elemento del Array
array_uni[2:4]          #array([5, 7])
# Accediendo a los elementos 0, 3 y 5 del Array
array_uni[0::3]         #array([1, 7])


# Creación de un Array multidimensional
array_multi = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Shape:", array_multi.shape)          #Shape: (2, 4)
print("Array_multi:\n", array_multi)        #Array_multi:[[1 2 3 4][5 6 7 8]]
array_multi[0, 3]                           #4
# Accediendo a la segunda fila del Array
array_multi[1, :]                           #array([5, 6, 7, 8])
# Accediendo al tercer elemento de las dos primeras filas del Array
array_multi[0:2, 2]                         #array([3, 7])


# Creación de un Array unidimensional inicializado con el rango de elementos 0-27
array1 = np.arange(28)
print("Shape:", array1.shape)               #Shape: (28,)
print("Array 1:", array1)                   #Array 1: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]
# Cambiar las dimensiones del Array y sus longitudes
array1.shape = (7, 4)
print("Shape:", array1.shape)               #Shape: (7, 4)
print("Array 1:\n", array1)                 #Array 1:[[ 0  1  2  3][ 4  5  6  7][ 8  9 10 11][12 13 14 15][16 17 18 19][20 21 22 23][24 25 26 27]]
# El ejemplo anterior devuelve un nuevo Array que apunta a los mismos datos. 
# Importante: Modificaciones en un Array, modificaran el otro Array
array2 = array1.reshape(4, 7)
print("Shape:", array2.shape)               #Shape: (4, 7)
print("Array 2:\n", array2)                 #Array 2: [[ 0  1  2  3  4  5  6][ 7  8  9 10 11 12 13][14 15 16 17 18 19 20][21 22 23 24 25 26 27]]
# Modificación del nuevo Array devuelto
array2[0, 3] = 20
print("Array 2:\n", array2)                 #Array 2:[[ 0  1  2 20  4  5  6][ 7  8  9 10 11 12 13][14 15 16 17 18 19 20][21 22 23 24 25 26 27]]
print("Array 1:\n", array1)                 #Array 1:[[ 0  1  2 20][ 4  5  6  7][ 8  9 10 11][12 13 14 15][16 17 18 19][20 21 22 23][24 25 26 27]]
# Desenvuelve el Array, devolviendo un nuevo Array de una sola dimension
# Importante: El nuevo array apunta a los mismos datos
print("Array 1:", array1.ravel())           #Array 1: [ 0  1  2 20  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]


# Creación de dos Arrays unidimensionales
array1 = np.arange(2, 18, 2)
array2 = np.arange(8)
print("Array 1:", array1)                   #Array 1: [ 2  4  6  8 10 12 14 16]
print("Array 2:", array2)                   #Array 2: [0 1 2 3 4 5 6 7]
print(array1 + array2)                      #[ 2  5  8 11 14 17 20 23]
print(array1 - array2)                      #[2 3 4 5 6 7 8 9]
print(array1 * array2)                      #[  0   4  12  24  40  60  84 112]
 
 
# Creación de dos Arrays unidimensionales
array1 = np.arange(5)
array2 = np.array([3])
print("Shape Array 1:", array1.shape)       #Shape Array 1: (5,)
print("Array 1:", array1)                   #Array 1: [0 1 2 3 4]
print("Shape Array 2:", array2.shape)       #Shape Array 2: (1,)
print("Array 2:", array2)                   #Array 2: [3]
array1 + array2                             #array([3, 4, 5, 6, 7])


array1 = np.arange(6)
array1.shape = (2, 3)
array2 = np.arange(6, 18, 4)
print("Shape Array 1:", array1.shape)       #Shape Array 1: (2, 3)
print("Array 1:\n", array1)                 #Array 1:[[0 1 2][3 4 5]]
print("Shape Array 2:", array2.shape)       #Shape Array 2: (3,)
print("Array 2:", array2)                   #Array 2: [ 6 10 14]
array1 + array2                             #array([[ 6, 11, 16],[ 9, 14, 19]])


# Creación de un Array unidimensional
array1 = np.arange(1, 20, 2)
print("Array 1:", array1)                   #Array 1: [ 1  3  5  7  9 11 13 15 17 19]
array1.mean()                               #10.0   da la media del valores de los elementos
array1.sum()                                #100    suma todos los elementos del array
np.square(array1)                           #array([  1,   9,  25,  49,  81, 121, 169, 225, 289, 361])
np.sqrt(array1)                             #array([1, 1.73205081, 2.23606798, 2.64575131, 3, 3.31662479, 3.60555128, 3.87298335, 4.12310563, 4.35889894])
np.exp(array1)                              #array([2.71828183e+00, 2.00855369e+01, 1.48413159e+02, 1.09663316e+03, 8.10308393e+03, 5.98741417e+04, 4.42413392e+05, 3.26901737e+06, 2.41549528e+07, 1.78482301e+08])
np.log(array1)                              #array([0, 1.09861229, 1.60943791, 1.94591015, 2.19722458, 2.39789527, 2.56494936, 2.7080502 , 2.83321334, 2.94443898])
 
 
 
 


 
 
 
 
 
 
 
 
 
 
 



