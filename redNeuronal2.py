# --- LIBRERIAS NECESARIAS PARA EL DESARROLLO RED NEURONAL ---
import torch
from torch import nn, optim
import torch.nn.functional as F  # <--- IMPORTANTE: Necesario para tu función one_hot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# from PIL import Image
# import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
data_dir = "dataset3"
# Ajustamos num_classes dinámicamente más abajo según lo que encuentre en la carpeta

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset y Split
# Asegúrate de que 'dataset3' existe. Si usas la ruta de kaggle, pon la variable path_dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Cálculo dinámico de clases
classes = full_dataset.classes
NUM_CLASSES = len(classes)
print(f"Dataset cargado. {NUM_CLASSES} Clases encontradas: {classes}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Aplicar transformación de validación
val_dataset.dataset.transform = val_transform

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("-" * 30)

# --- 2. DEFINICIÓN DEL MODELO (SimpleNet) ---
class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleNet(NUM_CLASSES)

# --- 3. TUS FUNCIONES INTEGRADAS ---

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = len(test_loader.dataset) # Forma segura de obtener longitud

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy

def train_with_validation(model, train_loader, dev_loader, criterion, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}

    print(f"Iniciando entrenamiento en: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Ajuste para CrossEntropyLoss con one_hot (si labels no son one-hot ya)
            # CrossEntropyLoss normalmente espera índices de clase, no one-hot.
            # Si quieres usar one-hot explícito:
            n_classes = outputs.shape[1]
            labels_one_hot = F.one_hot(labels, num_classes=n_classes).float()
            loss = criterion(outputs, labels_one_hot)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validación (loss)
        model.eval()
        dev_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                n_classes = outputs.shape[1]
                labels_one_hot = F.one_hot(labels, num_classes=n_classes).float()
                dev_loss = criterion(outputs, labels_one_hot)
                dev_running_loss += dev_loss.item()

        avg_dev_loss = dev_running_loss / len(dev_loader)

        # Validación usando la función evaluate (medida de precisión)
        dev_acc = evaluate(model, dev_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f'[Epoch {epoch + 1}] train_loss: {avg_train_loss:.3f} | train_acc: {train_acc:.2f}% | dev_loss: {avg_dev_loss:.3f} | dev_acc: {dev_acc:.2f}%')

    return model, history

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['dev_loss'], label='Validation Loss')
    plt.title('Loss por Época')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['dev_acc'], label='Validation Acc')
    plt.title('Accuracy por Época')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.show()

# --- 4. EJECUCIÓN PRINCIPAL ---

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ENTRENAR USANDO TUS FUNCIONES
model_entrenado, history = train_with_validation(
    model,
    train_loader,
    val_loader, # Pasamos val_loader como 'dev_loader'
    criterion,
    optimizer,
    epochs=10
)

# GRAFICAR RESULTADOS
plot_training_history(history)

# --- 5. MATRIZ DE CONFUSIÓN ---
print("-" * 30)
print("Generando Matriz de Confusión Final...")

model_entrenado.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    # Usamos val_loader porque no tenemos test_loader separado
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_entrenado(inputs)
        _, predicted = torch.max(outputs, 1) # <--- Corregido el error de sintaxis

        # Mover a CPU para numpy
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generar la matriz
cm = confusion_matrix(all_labels, all_preds)

# Visualizar
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Guardar Modelo
torch.save(model_entrenado.state_dict(), "simple_cnn_final.pth")
print("Modelo guardado exitosamente.")