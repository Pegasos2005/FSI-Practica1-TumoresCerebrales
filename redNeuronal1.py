import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
# Importamos convnext especificamente
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os

# --- LIBRERÍAS DE VISUALIZACIÓN ---
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carga del dataset
data_dir = "dataset3"

# Transformacion (Standard ImageNet)
transformacion_comun = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transformacion_comun)

# ================================Imprime los 4 tipos de tumores (completamente borrable)==============================
classes = full_dataset.classes
NUM_CLASSES = len(classes)
print(f"Dataset cargado. {NUM_CLASSES} Clases encontradas: {classes}")
# =====================================================================================================================

# Dividir dataset (80% train, 20% val)
n_pictures_train = int(0.8 * len(full_dataset))
n_pictures_valid = len(full_dataset) - n_pictures_train
train_dataset, val_dataset = random_split(full_dataset, [n_pictures_train, n_pictures_valid])


# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =========================================================================
#  ZONA DE FUNCIONES REUTILIZABLES
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
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Métricas
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

# --- 2. MODELO (ConvNeXt) ---
weights = ConvNeXt_Tiny_Weights.DEFAULT # Coje los conociemientos d la IA ConvNeXt
model = convnext_tiny(weights=weights) # Crea una red neuronal copiando ConvNeXt

# Modificar la última capa para el número correcto de clases
# ConvNeXt tiene la estructura: model.classifier[2] es la capa lineal final
num_ftrs = model.classifier[2].in_features # Lee cuantas neuronas hay en la última capa
model.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES) # Cambia ese nº d neuronas x las 4 q hacen falta

# Detecta si hay GPU y la usa en caso afirmativo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
model = model.to(device)

# Establece el criterio de entrenamiento y la velocidad de aprendizaje
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- 3. ENTRENAMIENTO (Usando las funciones nuevas) ---
# Ejecutamos el entrenamiento y guardamos el historial
model_entrenado, historial = train_with_validation(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    epochs=7
)

# --- 4. ANÁLISIS VISUAL ---
# Genera las gráficas y la matriz de confusión automáticamente
analizar_resultados(model_entrenado, historial, val_loader, classes)

# --- 5. GUARDAR Y PROBAR ---
torch.save(model_entrenado.state_dict(), "ia_tumors01 ConvNeXt.pth")
print("\nModelo guardado correctamente.")
