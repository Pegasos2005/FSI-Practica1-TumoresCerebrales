# Código 3
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuración del Dataset ---
data_dir = "dataset3"

# --- 1. Transformaciones ---
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

# --- 2. Carga y División ---
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
classes = full_dataset.classes
NUM_CLASSES = len(classes)
print(f"Dataset cargado. {NUM_CLASSES} Clases: {classes}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Cambio de transformación para validación
val_dataset.dataset.transform = val_transform

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("-" * 30)

# --- 3. Definición del Modelo ---
class CNN_Minimalista(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_Minimalista, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN_Minimalista(NUM_CLASSES)

# =========================================================================
#  ZONA DE FUNCIONES REUTILIZABLES (COPIAR Y PEGAR EN OTRAS IAs)
# =========================================================================

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

            # Cálculo de Loss
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

        # Guardar historial
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(avg_dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f'[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.3f} | Train Acc: {train_acc:.2f}% | Val Acc: {dev_acc:.2f}%')

    return model, history

def analizar_resultados(model, history, test_loader, classes):
    """
    Función 'Todo en Uno': Dibuja Loss, Accuracy y Matriz de Confusión.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    plt.figure(figsize=(18, 5))

    # 1. Gráfica Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['dev_loss'], label='Val')
    plt.title('Pérdida (Loss)'); plt.legend()

    # 2. Gráfica Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['dev_acc'], label='Val')
    plt.title('Precisión (Accuracy)'); plt.legend()

    # 3. Matriz de Confusión
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

# =========================================================================

# --- 4. Configuración y Ejecución ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model_entrenado, historial = train_with_validation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=7
)

analizar_resultados(model_entrenado, historial, val_loader, classes)

# --- 7. Guardar y Predecir ---
model_save_path = "simple_cnn_model.pth"
torch.save(model_entrenado.state_dict(), model_save_path)
print(f"\nModelo guardado en: {model_save_path}")

# Ejemplo de predicción
img_path = os.path.join(data_dir, classes[0], "G_1_BR.jpg")
if os.path.exists(img_path):
    print(f"Probando predicción con: {img_path}")
    img = Image.open(img_path).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(device)

    model_entrenado.eval()
    with torch.no_grad():
        outputs = model_entrenado(x)
        _, pred = torch.max(outputs, 1)
        print("Predicción:", classes[pred.item()])