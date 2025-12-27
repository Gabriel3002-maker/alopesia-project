# Modelo v0.1 - Alopecia Segmentation

## Información del Modelo

- **Versión**: 0.1
- **Fecha**: 2025-12-21
- **Arquitectura**: SimpleUNet
- **Épocas entrenadas**: 5
- **Dataset**: 110 imágenes aumentadas (11 originales)
- **Loss final**: 1.1468

## Archivos

- `alopecia_segmentation_model.pth` - Modelo final
- `model_checkpoint_epoch_5.pth` - Checkpoint época 5
- `predictions_visualization.png` - Visualización de predicciones
- `training_loss.png` - Curva de aprendizaje

## Rendimiento

- Loss inicial (época 1): 1.5986
- Loss final (época 5): 1.1468
- Mejora: ~28%

## Notas

- Entrenado en CPU (i5 3ra gen)
- Batch size: 1
- Learning rate: 0.001
- Pos weight: 10.0 (para balance de clases)
