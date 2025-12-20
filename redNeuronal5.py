######

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ==========================================
# FUNCIONES DE EVALUACIÓN Y ANÁLISIS
# ==========================================

def evaluate(model, test_loader):
    """Calcula la precisión (accuracy) de forma rápida."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

def train_with_validation(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    """Entrena el modelo y guarda el historial de Loss/Acc."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

    print(f"Iniciando entrenamiento en: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Métricas de la época
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validación
        model.eval()
        dev_loss_acc = 0.0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                dev_loss_acc += criterion(outputs, labels).item()

        avg_dev_loss = dev_loss_acc / len(dev_loader)
        dev_acc = evaluate(model, dev_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f'[Epoch {epoch + 1}] Loss: {avg_train_loss:.3f} | Train Acc: {train_acc:.2f}% | Val Acc: {dev_acc:.2f}%')

    return model, history

def analizar_resultados(model, history, test_loader, classes):
    """Dibuja Loss, Accuracy y Matriz de Confusión."""
    epochs = range(1, len(history['train_loss']) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['dev_loss'], label='Val')
    plt.title('Pérdida (Loss)'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['dev_acc'], label='Val')
    plt.title('Precisión (Accuracy)'); plt.legend()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real'); plt.xlabel('Predicción')

    plt.tight_layout()
    plt.show()

# ==========================================
# CONFIGURACIÓN Y EJECUCIÓN
# ==========================================

data_dir = "dataset3"
NUM_CLASSES = 4

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carga de datos
if not os.path.exists(data_dir):
    os.makedirs(os.path.join(data_dir, "glioma"), exist_ok=True)

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Aplicamos Wrapper para validación (opcional, aquí simplificado)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Definición de Red
class CNN_Modificada(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_Modificada, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.Tanh(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.Tanh(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN_Modificada(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler.step()

# Entrenamiento usando las nuevas funciones
model_entrenado, historial = train_with_validation(
    model, train_loader, val_loader, criterion, optimizer, epochs=7
)

# Análisis visual
analizar_resultados(model_entrenado, historial, val_loader, full_dataset.classes)

# Guardado
torch.save(model_entrenado.state_dict(), "cnn_modificada.pth")
print("\nModelo y gráficas generadas con éxito.")