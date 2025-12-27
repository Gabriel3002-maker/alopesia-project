import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from train_model import SimpleUNet

def load_model(model_path, device='cpu'):
    """Carga el modelo entrenado"""
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Modelo cargado desde: {model_path}")
    return model

def preprocess_image(image_path, img_size=256):
    """Preprocesa una imagen para el modelo"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Guardar tamaño original para redimensionar después
    original_size = img.shape[:2]
    
    # Redimensionar y normalizar
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convertir a tensor
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Añadir batch dimension
    
    return img_tensor, img_rgb, original_size

def predict(model, image_tensor, device='cpu', threshold=0.3):
    """Realiza la predicción"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
    
    # Convertir a numpy
    probs_map = probs.squeeze().cpu().numpy()
    mask_binary = mask.squeeze().cpu().numpy()
    
    return probs_map, mask_binary

def visualize_prediction(image_rgb, probs_map, mask_binary, save_path=None):
    """Visualiza la predicción"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(image_rgb)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Mapa de calor
    im = axes[1].imshow(probs_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Mapa de Confianza')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Máscara binaria
    axes[2].imshow(mask_binary, cmap='gray')
    axes[2].set_title('Predicción (>0.3)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Visualización guardada en: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predicción de Alopecia")
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen para predecir')
    parser.add_argument('--model', type=str, default='outputs/models/v0.1/alopecia_segmentation_model.pth', 
                        help='Ruta al modelo entrenado')
    parser.add_argument('--output', type=str, default=None, 
                        help='Ruta para guardar la visualización (opcional)')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Umbral de confianza (0-1)')
    parser.add_argument('--no-display', action='store_true',
                        help='No abrir la imagen automáticamente')
    
    args = parser.parse_args()
    
    # Verificar que existe la imagen
    if not Path(args.image).exists():
        print(f"❌ Error: No se encontró la imagen {args.image}")
        return
    
    # Verificar que existe el modelo
    if not Path(args.model).exists():
        print(f"❌ Error: No se encontró el modelo {args.model}")
        return
    
    print(f"\n🔍 Procesando imagen: {args.image}")
    print(f"🤖 Usando modelo: {args.model}")
    print(f"📊 Umbral de confianza: {args.threshold}\n")
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    
    # Preprocesar imagen
    image_tensor, image_rgb, original_size = preprocess_image(args.image)
    
    # Predecir
    probs_map, mask_binary = predict(model, image_tensor, device, args.threshold)
    
    # Visualizar
    output_path = args.output
    if output_path is None:
        # Generar nombre automático
        img_name = Path(args.image).stem
        output_path = f"outputs/predictions/{img_name}_prediction.png"
        Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    
    visualize_prediction(image_rgb, probs_map, mask_binary, output_path)
    
    # Estadísticas
    area_detectada = np.sum(mask_binary) / (mask_binary.shape[0] * mask_binary.shape[1]) * 100
    confianza_promedio = np.mean(probs_map[mask_binary > 0]) if np.any(mask_binary > 0) else 0
    
    print(f"\n📈 Estadísticas:")
    print(f"   Área detectada: {area_detectada:.2f}%")
    print(f"   Confianza promedio: {confianza_promedio:.2f}")
    print(f"\n✅ Predicción completada!")
    
    # Abrir imagen automáticamente
    if not args.no_display:
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Linux':
                subprocess.run(['xdg-open', output_path], check=False)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', output_path], check=False)
            elif platform.system() == 'Windows':
                subprocess.run(['start', output_path], shell=True, check=False)
            print(f"🖼️  Abriendo visualización...")
        except Exception as e:
            print(f"⚠️  No se pudo abrir automáticamente. Abre manualmente: {output_path}")

if __name__ == "__main__":
    main()
