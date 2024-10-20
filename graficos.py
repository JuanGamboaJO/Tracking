import pandas as pd
import matplotlib.pyplot as plt


file_path = 'training_results.xlsx'  
df = pd.read_excel(file_path)

epocas = df.iloc[:, 0]  
col2 = df.iloc[:, 1]   
col3 = df.iloc[:, 2]    
col4 = df.iloc[:, 3]    
col5 = df.iloc[:, 4]    

fig, axs = plt.subplots(1, 2, figsize=(14, 6))


axs[0].plot(epocas, col2, label='Loss Trainig', color='blue', marker='o')
axs[0].plot(epocas, col4, label='Loss Val', color='green', marker='o')
axs[0].set_title('Loss')
axs[0].set_xlabel('Épocas')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epocas, col3, label='Accuracy Training', color='red', marker='o')
axs[1].plot(epocas, col5, label='Accuracy Val', color='orange', marker='o')
axs[1].set_title('Accuaracy')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# Mostrar las gráficas
plt.show()