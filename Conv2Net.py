import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Capa 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1)  # Para imágenes en escala de grises
        self.batchnorm1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)  # Batch Normalization
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capa 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)  # Batch Normalization
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Capa completamente conectada
        self.fc1 = nn.Linear(215296, 81)  # Ajusta el tamaño de entrada según la salida del aplanado
        self.dropout = nn.Dropout(0.02)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(81, 4)  # Salida final

        # Capa de Normalización por Capas
        self.layernorm = nn.LayerNorm(81)  # Normaliza las activaciones en la capa completamente conectada

    def forward(self, x1,x2):
        # Procesar imagen del ojo izquierdo
        x1 = self.conv1(x1)
        x1 = self.batchnorm1(x1)  # Batch Normalization
        x1 = self.relu1(x1)
        x1 = self.dropout(x1)
        x1 = self.maxpool1(x1)

        x1 = self.conv2(x1)
        x1 = self.batchnorm2(x1)  # Batch Normalization
        x1 = self.relu2(x1)
        x1 = self.maxpool2(x1)

        x1 = self.conv3(x1)
        x1 = self.batchnorm3(x1)  # Batch Normalization
        x1 = self.relu3(x1)
        x1 = self.maxpool3(x1)

        x1 = x1.view(x1.size(0), -1)  # Aplanar
        
        # Procesar imagen del ojo derecho
        x2 = self.conv1(x2)
        x2 = self.batchnorm1(x2)  # Batch Normalization
        x2 = self.relu1(x2)
        x2 = self.dropout(x2)
        x2 = self.maxpool1(x2)

        x2 = self.conv2(x2)
        x2 = self.batchnorm2(x2)  # Batch Normalization
        x2 = self.relu2(x2)
        x2 = self.maxpool2(x2)

        x2 = self.conv3(x2)
        x2 = self.batchnorm3(x2)  # Batch Normalization
        x2 = self.relu3(x2)
        x2 = self.maxpool3(x2)

        x2 = x2.view(x2.size(0), -1)  # Aplanar
        

        # Concatenar las características de ambos ojos
        x = torch.cat((x1, x2), dim=1)

        # Capas completamente conectadas
        x = self.fc1(x)
        x = self.layernorm(x)  # Layer Normalization
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

# Crear el modelo
model = ConvNet()


# Contar los parámetros totales y entrenables
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Número total de parámetros: {total_params}")
print(f"Número de parámetros entrenables: {trainable_params}")
