import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from Conv2Net import ConvNet  # Asegúrate de que ConvNet tenga dos entradas
from PIL import Image
import seaborn as sns
import copy
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Recorrer las carpetas y guardar las rutas de las imágenes y las etiquetas
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            derecho_dir = os.path.join(class_dir, 'Derecho')
            izquierdo_dir = os.path.join(class_dir, 'Izquierdo')

            # Asegurarse de que hay la misma cantidad de imágenes en ambas carpetas
            derecho_images = sorted(os.listdir(derecho_dir))
            izquierdo_images = sorted(os.listdir(izquierdo_dir))

            for derecho_img, izquierdo_img in zip(derecho_images, izquierdo_images):
                derecho_path = os.path.join(derecho_dir, derecho_img)
                izquierdo_path = os.path.join(izquierdo_dir, izquierdo_img)
                self.image_paths.append((derecho_path, izquierdo_path))
                self.labels.append(label)  # Etiqueta correspondiente a la clase

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        derecho_path, izquierdo_path = self.image_paths[idx]
        derecho_image = Image.open(derecho_path)  
        izquierdo_image = Image.open(izquierdo_path)  
        label = self.labels[idx]

        if self.transform:
            derecho_image = self.transform(derecho_image)
            izquierdo_image = self.transform(izquierdo_image)

        return derecho_image, izquierdo_image, label

def entrenamiento():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # Convertir imágenes a tensores
    ])

    dataset = EyeDataset(root_dir='Ojos', transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    num_epochs = 3
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Diccionario para almacenar los resultados
    resultados = {
        'Epoca': [],
        'Perdida_entrenamiento': [],
        'Exactitud_entrenamiento': [],
        'Perdida_validacion': [],
        'Exactitud_validacion': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for derecho_images, izquierdo_images, labels in train_loader:
            derecho_images = derecho_images.to(device)
            izquierdo_images = izquierdo_images.to(device)
            labels = labels.to(device)

            outputs = model(derecho_images, izquierdo_images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        bestModel = copy.deepcopy(model)

        train_accuracy = correct_train / total_train

        val_accuracy, val_loss = evaluate_accuracy(model, val_loader)
        # Guardar los resultados de la época en el diccionario
        resultados['Epoca'].append(epoch + 1)
        resultados['Perdida_entrenamiento'].append(loss.item())
        resultados['Exactitud_entrenamiento'].append(train_accuracy)
        resultados['Perdida_validacion'].append(val_loss)
        resultados['Exactitud_validacion'].append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Perdida de entrenamiento: {loss.item():.4f}, Exactitud de entrenamiento: {train_accuracy:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Perdida de validación: {val_loss:.4f}, Exactitud de validación: {val_accuracy:.4f}')

    # Guardar los resultados en un archivo Excel
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel('resultados_entrenamiento.xlsx', index=False)

    # Evaluar en el conjunto de prueba
    test_accuracy, loss_test = evaluate_accuracy(model, test_loader)
    print(f'Exactitud en el conjunto de prueba: {test_accuracy:.4f}')
    torch.save(bestModel.state_dict(), "Models/Eye" + str(int(test_accuracy*100)) + ".plt")
    confusion_matrix = plot_confusion_matrix(model, test_loader, num_classes=4)

def evaluate_accuracy(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for derecho_images, izquierdo_images, labels in dataloader:
            derecho_images = derecho_images.to(device)
            izquierdo_images = izquierdo_images.to(device)
            labels = labels.to(device)
            outputs = model(derecho_images, izquierdo_images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f'Exactitud: {accuracy * 100:.2f}%')
    return accuracy , avg_loss

def plot_confusion_matrix(model, dataloader, num_classes):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for derecho_images, izquierdo_images, labels in dataloader:
            derecho_images = derecho_images.to(device)
            izquierdo_images = izquierdo_images.to(device)
            outputs = model(derecho_images, izquierdo_images)
            _, preds = torch.max(outputs, 1)

            for true_label, pred_label in zip(labels, preds):
                confusion_matrix[true_label.long(), pred_label.long()] += 1

    cm = confusion_matrix.numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Verdadera')
    plt.title('Matriz de Confusión')
    plt.show()

    return confusion_matrix

entrenamiento()
