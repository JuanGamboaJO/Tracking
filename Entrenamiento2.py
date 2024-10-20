import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import seaborn as sns
from ResNet import Model
import copy
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(model, dataloader, num_classes):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
           

            outputs = model(images)
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

def evaluate_accuracy(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # Solo se pasa combined_images

            loss = criterion(outputs, labels)
            total_loss += loss.item()  # Acumular las pérdidas
            
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)

    return accuracy,avg_loss

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Recorrer las carpetas y guardar las rutas de las imágenes y las etiquetasp
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
        image1 = Image.open(derecho_path)  # Cargar en escala de grises
        image2  = Image.open(izquierdo_path)  # Cargar en escala de grises
        label = self.labels[idx]

        
        derecho_image = self.transform(image1)
        izquierdo_image = self.transform(image2)

        combined_image = torch.cat([derecho_image, izquierdo_image], dim=2)


        return combined_image, label

def entrenamiento():
    transform = torchvision.transforms.Compose([  
        torchvision.transforms.Resize((224,112)),
        torchvision.transforms.ToTensor(),  # Convertir imágenes a tensores
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizar según el preentrenamiento
    ])

    dataset = EyeDataset(root_dir='Ojos', transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    num_epochs = 3
    model = Model(n_outputs=4)
    model.unfreeze()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
            {'params': model.resnet.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    
    results = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        all_labels = []
        all_preds = []
        for images, labels in train_loader: 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) 
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        bestModel = copy.deepcopy(model)

        train_accuracy = sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)

        val_accuracy, val_loss = evaluate_accuracy(model, val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Exactitud: {val_accuracy:.4f}')

        results['epoch'].append(epoch + 1)
        results['train_loss'].append(loss.item())
        results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accuracy)
    
    df = pd.DataFrame(results)
    df.to_excel('training_results.xlsx', index=False)

    num_classes = 4
    test_accuracy,loss_test = evaluate_accuracy(model, test_loader)
    print(f'Exactitud en el conjunto de prueba: {test_accuracy:.4f}')
    torch.save(bestModel.state_dict(), "Models/Res" + str(int(test_accuracy*100)) + ".plt")
    confusion_matrix = plot_confusion_matrix(model, test_loader, num_classes)



# Las funciones evaluate_accuracy y plot_confusion_matrix permanecen iguales

entrenamiento()
