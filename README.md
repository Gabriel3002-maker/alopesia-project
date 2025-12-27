# Alopecia Segmentation Model

Modelo de segmentación para detección de alopecia usando PyTorch.

## 🚀 Inicio Rápido

### 1. Activar entorno virtual
```bash
source env/bin/activate
```

### 2. Ejecutar pipeline completo
```bash
python3 main.py
```

## ⚙️ Opciones de Entrenamiento

### Para CPU (recomendado para i5 3ra gen)
```bash
# Entrenamiento rápido (5 épocas, ~10-15 min)
python3 main.py --epochs 5

# Solo entrenar (si ya procesaste datos)
python3 main.py --skip_processing --epochs 5
```

### Si tienes más tiempo
```bash
# Entrenamiento completo (30 épocas, ~1-2 horas en CPU)
python3 main.py --epochs 30
```

## 📁 Estructura del Proyecto

```
alopecia_final/
├── data/                          # Datos procesados
│   ├── test-coco-upload.zip      # Dataset original
│   └── alopecia_dataset/         # Datos aumentados
├── outputs/
│   └── models/                   # Modelos entrenados
│       ├── alopecia_segmentation_model.pth
│       ├── training_loss.png
│       └── predictions_visualization.png
├── main.py                       # Script principal
├── process_dataset.py            # Procesamiento de datos
├── train_model.py               # Entrenamiento
└── requirements.txt             # Dependencias
```

## 💡 Notas para CPU

- **Batch size**: Configurado en 1 para evitar problemas de memoria
- **Épocas por defecto**: 5 (suficiente para ver resultados)
- **Tiempo estimado**: ~2-3 min por época en i5 3ra gen
- El modelo se guarda cada 5 épocas automáticamente

## 🔧 Troubleshooting

Si el proceso es muy lento:
```bash
# Reducir épocas
python3 main.py --epochs 3

# O entrenar con datos ya procesados
python3 main.py --skip_processing --epochs 3
```


Plan de Corrección: Modelo de Detección de Alopecia
Diagnóstico Completo
Problema Crítico Identificado
Diagnóstico Profundo
Review
Diagnóstico Profundo

CAUTION

Desbalance de Clases Severo: El modelo está fallando completamente debido a un desbalance extremo entre píxeles con alopecia (4-10%) vs píxeles normales (90-96%).

Métricas del Problema
Imagen	Ground Truth	Predicción	IoU	Estado
Imagen 1	4.16%	85.67%	0.0485	❌ MALO
Imagen 2	4.02%	54.51%	0.0738	❌ MALO
Imagen 3	10.67%	92.04%	0.1144	❌ MALO
Conclusión: El modelo está prediciendo casi el inverso de lo que debería. No es una simple inversión de máscaras, sino que el modelo no está aprendiendo correctamente.

Causa Raíz
El problema tiene múltiples factores:

Desbalance Extremo de Clases:

Solo 4-10% de píxeles son alopecia
90-96% son píxeles normales
El modelo aprende a predecir "todo es normal" porque minimiza la pérdida
Peso Insuficiente en la Función de Pérdida:

Actualmente: pos_weight=10.0
Necesario: pos_weight=20-50 o usar Focal Loss
Falta de Validación Durante Entrenamiento:

No hay métricas para detectar este problema temprano
No se monitorea IoU/Dice durante entrenamiento
Dataset Pequeño:

Solo 11 imágenes originales
Aunque hay 110 con augmentación, la variedad es limitada
Proposed Changes
1. Mejorar la Función de Pérdida
[MODIFY] 
train_model.py
Cambios principales:

# Opción 1: Aumentar pos_weight dramáticamente
pos_weight = torch.tensor([30.0]).to(device)  # Era 10.0
# Opción 2: Implementar Focal Loss (RECOMENDADO)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
# Opción 3: Dice Loss + BCE (MEJOR)
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]))
    
    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice = 1 - (2. * intersection + 1) / (inputs_sigmoid.sum() + targets.sum() + 1)
        
        return bce + dice
Agregar métricas de evaluación:

IoU (Intersection over Union)
Dice Coefficient
Precisión, Recall, F1-Score
2. Implementar Validación Durante Entrenamiento
[MODIFY] 
train_model.py
Split train/validation:

# Dividir dataset 80/20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
Validación cada epoch:

Calcular IoU en conjunto de validación
Early stopping si IoU no mejora
Guardar mejor modelo (no solo el último)
3. Mejorar Arquitectura del Modelo
[MODIFY] 
train_model.py
Problemas actuales:

Skip connections tienen dimensiones incorrectas (líneas 145, 149)
Modelo muy simple para esta tarea
Mejoras:

# Corregir concatenación en decoder
d1 = self.up1(b)
d1 = torch.cat([d1, e2], dim=1)  # e2, no e1!
d2 = self.up2(d1)
d2 = torch.cat([d2, e1], dim=1)  # e1, no x!
Agregar Batch Normalization y Dropout:

self.enc1 = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
4. Aumentar Datos con Enfoque en Alopecia
[MODIFY] 
process_dataset.py
Aumentar número de augmentaciones:

num_aug = 19  # Era 9, ahora 20x más datos
Agregar augmentaciones específicas para alopecia:

# Agregar crops centrados en regiones de alopecia
iaa.Sometimes(0.3, iaa.CropAndPad(
    percent=(-0.1, 0.1),
    keep_size=True
))
5. Crear Script de Entrenamiento Mejorado
[NEW] 
train_improved.py
Script completo con:

Focal Loss + Dice Loss
Train/Val split
Métricas completas (IoU, Dice, F1)
Early stopping
Learning rate scheduling
Visualizaciones durante entrenamiento
Logging detallado
6. Actualizar Script de Predicción
[MODIFY] 
predict.py
Agregar cálculo de pelos para peluca:

def calculate_hair_needed(mask_binary, image_size_cm, hair_density=120):
    """
    Calcula pelos necesarios para peluca
    
    Args:
        mask_binary: Máscara binaria de alopecia
        image_size_cm: Tamaño real de la cabeza en cm
        hair_density: Pelos por cm² (normal: 100-150)
    """
    # Área en píxeles
    area_pixels = np.sum(mask_binary)
    total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
    
    # Convertir a cm²
    area_cm2 = (area_pixels / total_pixels) * image_size_cm
    
    # Calcular pelos
    hairs_needed = int(area_cm2 * hair_density)
    
    return hairs_needed, area_cm2
Verification Plan
Automated Tests
# 1. Reprocesar dataset con más augmentaciones
source env/bin/activate
python process_dataset.py --base_dir . --zip_file test-coco-upload.zip
# 2. Entrenar con modelo mejorado
python train_improved.py \
    --data_dir data/alopecia_dataset/augmented_dataset \
    --output_dir outputs/models/v0.2 \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
# 3. Validar en imagen de prueba
python predict.py \
    --image test.jpg \
    --model outputs/models/v0.2/alopecia_segmentation_model.pth \
    --threshold 0.5
Manual Verification
Monitorear Entrenamiento:

IoU debe subir progresivamente (objetivo: >0.5)
Dice coefficient debe mejorar (objetivo: >0.6)
Pérdida debe bajar consistentemente
Validar Predicciones:

Probar con imagen 
test.jpg
Verificar que detecta áreas laterales/temporales
Confirmar que NO detecta áreas con cabello denso
IoU en validación debe ser >0.5
Calcular Pelos para Peluca:

Usar área detectada para estimar pelos necesarios
Validar que números tienen sentido (ej: 5000-15000 pelos para alopecia parcial)
Expectativas Realistas
IMPORTANT

Con solo 11 imágenes originales, el modelo tendrá limitaciones. Para producción se recomienda:

Mínimo 100-200 imágenes originales
Imágenes de diferentes tipos de alopecia
Diferentes ángulos, iluminación, tonos de piel
Validación con dermatólogos
Resultados esperados con dataset actual:

IoU: 0.4 - 0.6 (aceptable para prototipo)
Dice: 0.5 - 0.7
Puede funcionar para casos similares a los de entrenamiento
Necesitará más datos para generalizar bien
Próximos Pasos
Implementar mejoras en 
train_model.py
Crear train_improved.py con todas las mejoras
Reentrenar modelo
Validar resultados
Si funciona bien, agregar cálculo de pelos en 
predict.py


# 1. Reprocesar dataset con más augmentaciones
source env/bin/activate
python process_dataset.py --base_dir . --zip_file test-coco-upload.zip
# 2. Entrenar con modelo mejorado
python train_improved.py \
    --data_dir data/alopecia_dataset/augmented_dataset \
    --output_dir outputs/models/v0.2 \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
# 3. Validar en imagen de prueba
python predict.py \
    --image test.jpg \
    --model outputs/models/v0.2/alopecia_segmentation_model.pth \
    --threshold 0.5# alopesia-project
