import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el archivo Excel
file_path = 'Matriz.xlsx'  # Reemplaza con la ruta de tu archivo
df = pd.read_excel(file_path, header=None)  # header=None para indicar que no hay fila de títulos

# Extraer los valores reales y predichos
y_true = df.iloc[:, 4]  # Columna 1: valores reales
y_pred = df.iloc[:, 5]  # Columna 2: valores predichos

# Etiquetas de las clases
clases = ['ARRIBA', 'CENTRO', 'DERECHA', 'IZQUIERDA']

# Crear una matriz de confusión vacía de 4x4
num_clases = len(clases)
matriz_confusion = np.zeros((num_clases, num_clases), dtype=int)

# Llenar la matriz de confusión
for true, pred in zip(y_true, y_pred):
    matriz_confusion[true][pred] += 1

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=clases, yticklabels=clases)
plt.title('Matriz de Confusión Red EYEIA')
plt.xlabel('Valores Predichos')
plt.ylabel('Valores Reales')
plt.show()
