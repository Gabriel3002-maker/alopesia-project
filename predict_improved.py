import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Import the improved model
sys.path.append(str(Path(__file__).parent))
from train_improved import ImprovedUNet

def load_model(model_path, device='cpu'):
    """Carga el modelo entrenado mejorado"""
    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Modelo cargado desde: {model_path}")
    return model

def preprocess_image(image_path, img_size=256):
    """Preprocesa una imagen para el modelo"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Guardar tamaño original
    original_size = img.shape[:2]
    
    # Redimensionar y normalizar
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convertir a tensor
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img_rgb, original_size

def predict(model, image_tensor, device='cpu', threshold=0.5):
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

def calculate_hair_needed(mask_binary, head_circumference_cm=56, hair_density=120):
    """
    Calcula pelos necesarios para peluca basado en área de alopecia
    
    Args:
        mask_binary: Máscara binaria de alopecia (256x256)
        head_circumference_cm: Circunferencia de la cabeza en cm (promedio adulto: 56cm)
        hair_density: Pelos por cm² (normal: 100-150 pelos/cm²)
    
    Returns:
        dict con información de pelos necesarios
    """
    # Calcular área en píxeles
    alopecia_pixels = np.sum(mask_binary)
    total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
    alopecia_percentage = (alopecia_pixels / total_pixels) * 100
    
    # Estimar área real de la cabeza
    # Aproximación: área de cabeza ≈ circunferencia² / (4π)
    head_area_cm2 = (head_circumference_cm ** 2) / (4 * np.pi)
    
    # Calcular área de alopecia en cm²
    alopecia_area_cm2 = head_area_cm2 * (alopecia_percentage / 100)
    
    # Calcular pelos necesarios
    hairs_needed = int(alopecia_area_cm2 * hair_density)
    
    return {
        'alopecia_percentage': alopecia_percentage,
        'alopecia_area_cm2': alopecia_area_cm2,
        'hairs_needed': hairs_needed,
        'hair_density': hair_density,
        'head_circumference_cm': head_circumference_cm
    }

def visualize_prediction(image_rgb, probs_map, mask_binary, hair_info, save_path=None):
    """Visualiza la predicción con información de pelos"""
    fig = plt.figure(figsize=(18, 5))
    
    # Crear grid con 4 columnas
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2])
    
    # Imagen original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title('Imagen Original', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Mapa de confianza
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(probs_map, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Mapa de Confianza', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Máscara binaria
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mask_binary, cmap='gray')
    ax3.set_title(f'Detección de Alopecia\n({hair_info["alopecia_percentage"]:.1f}% del área)', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Información de pelos
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    info_text = f"""
INFORMACIÓN PARA PELUCA
{'='*35}

📊 Área Afectada:
   • Porcentaje: {hair_info['alopecia_percentage']:.1f}%
   • Área: {hair_info['alopecia_area_cm2']:.1f} cm²

👤 Parámetros:
   • Circunferencia: {hair_info['head_circumference_cm']} cm
   • Densidad: {hair_info['hair_density']} pelos/cm²

💇 Pelos Necesarios:
   • Total: {hair_info['hairs_needed']:,} pelos

📝 Notas:
   • Densidad normal: 100-150 pelos/cm²
   • Ajustar según grosor del cabello
   • Considerar +10% para reserva
    """
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualización guardada en: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predicción de Alopecia con Cálculo de Pelos")
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen para predecir')
    parser.add_argument('--model', type=str, default='outputs/models/v0.2/best_model.pth', 
                        help='Ruta al modelo entrenado')
    parser.add_argument('--output', type=str, default=None, 
                        help='Ruta para guardar la visualización (opcional)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Umbral de confianza (0-1)')
    parser.add_argument('--head-circumference', type=float, default=56.0,
                        help='Circunferencia de la cabeza en cm (default: 56)')
    parser.add_argument('--hair-density', type=int, default=120,
                        help='Densidad de cabello deseada (pelos/cm², default: 120)')
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
        print(f"💡 Tip: Usa --model para especificar la ruta correcta")
        return
    
    print(f"\n🔍 Procesando imagen: {args.image}")
    print(f"🤖 Usando modelo: {args.model}")
    print(f"📊 Umbral de confianza: {args.threshold}")
    print(f"👤 Circunferencia de cabeza: {args.head_circumference} cm")
    print(f"💇 Densidad objetivo: {args.hair_density} pelos/cm²\n")
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    
    # Preprocesar imagen
    image_tensor, image_rgb, original_size = preprocess_image(args.image)
    
    # Predecir
    probs_map, mask_binary = predict(model, image_tensor, device, args.threshold)
    
    # Calcular pelos necesarios
    hair_info = calculate_hair_needed(
        mask_binary, 
        head_circumference_cm=args.head_circumference,
        hair_density=args.hair_density
    )
    
    # Generar nombre de salida
    output_path = args.output
    if output_path is None:
        img_name = Path(args.image).stem
        output_path = f"outputs/predictions/{img_name}_prediction.png"
        Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    
    # Visualizar
    visualize_prediction(image_rgb, probs_map, mask_binary, hair_info, output_path)
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print(f"📈 RESUMEN DE DETECCIÓN")
    print(f"{'='*60}")
    print(f"  Área con alopecia: {hair_info['alopecia_percentage']:.2f}% ({hair_info['alopecia_area_cm2']:.1f} cm²)")
    print(f"  Pelos necesarios: {hair_info['hairs_needed']:,} pelos")
    print(f"  Confianza promedio: {np.mean(probs_map[mask_binary > 0]):.2f}" if np.any(mask_binary > 0) else "  No se detectó alopecia")
    print(f"{'='*60}")
    print(f"\n✅ Predicción completada!")
    
    # Abrir imagen automáticamente
    if not args.no_display:
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Linux':
                subprocess.run(['xdg-open', output_path], check=False)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', output_path], check=False)
            elif platform.system() == 'Windows':
                subprocess.run(['start', output_path], shell=True, check=False)
            print(f"🖼️  Abriendo visualización...")
        except Exception as e:
            print(f"⚠️  No se pudo abrir automáticamente. Abre manualmente: {output_path}")

if __name__ == "__main__":
    main()
