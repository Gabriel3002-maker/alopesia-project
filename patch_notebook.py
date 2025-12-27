import json

# Read the notebook
with open('/home/sucode/alopecia_final/Test3.ipynb', 'r') as f:
    nb = json.load(f)

# The new code for the model cell (using raw string to handle newlines easily)
new_code = r"""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# ----------------------------
# 1. DATASET PARA ENTRENAMIENTO
# ----------------------------
class AlopeciaDataset(Dataset):
    def __init__(self, aligned_dir, img_size=256):
        self.img_dir = os.path.join(aligned_dir, 'images')
        self.annot_dir = os.path.join(aligned_dir, 'annotations')
        self.img_size = img_size

        # Cargar información del dataset
        info_path = os.path.join(aligned_dir, 'annotations', 'dataset_info.json')
        with open(info_path, 'r') as f:
            self.dataset_info = json.load(f)

        # También cargar COCO filtrado
        coco_path = os.path.join(aligned_dir, 'annotations', 'filtered_coco.json')
        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)

        print(f"Dataset cargado: {len(self.dataset_info)} imágenes, {sum(item['annotation_count'] for item in self.dataset_info)} anotaciones")

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        item = self.dataset_info[idx]
        img_path = os.path.join(self.img_dir, item['aligned_image'])

        # Cargar imagen
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # Normalizar [0, 1]
        img = torch.tensor(img).permute(2, 0, 1).float()

        # Crear máscara
        mask = self._create_mask_from_coco(item['id'], self.img_size)
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

    def _create_mask_from_coco(self, image_id, size):
        mask = np.zeros((size, size), dtype=np.uint8)

        # Get image info
        image_info = None
        for img_info in self.coco_data['images']:
            if img_info['id'] == image_id:
                image_info = img_info
                break
        if image_info is None:
            raise ValueError(f"Image info not found for image_id: {image_id}")

        img_width = image_info['width']
        img_height = image_info['height']

        # Buscar anotaciones
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == image_id:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    for polygon in seg:
                        pts = np.array(polygon).reshape(-1, 2)
                        pts = (pts * size / max(img_width, img_height)).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)

        return mask

# ----------------------------
# 2. MODELO MEJORADO (Sin Sigmoid al final)
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(67, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        # Salida (logits, sin sigmoid)
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        
        d1 = self.up1(b)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x[:, :, :d2.shape[2], :d2.shape[3]]], dim=1)
        d2 = self.dec2(d2)
        
        out = self.output(d2)
        return out # <-- Dejamos los logits crudos

# ----------------------------
# 3. ENTRENAMIENTO (Con pesos para clases desbalanceadas)
# ----------------------------
def train_model():
    aligned_dir = '/content/alopecia_dataset/aligned_dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    dataset = AlopeciaDataset(aligned_dir, img_size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleUNet().to(device)

    # PESO PARA LA CLASE POSITIVA (ALOPECIA)
    # Esto penaliza 10 veces más equivocarse en la alopecia que en el fondo
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    train_losses = []

    print("\n=== COMIENZO DEL ENTRENAMIENTO MEJORADO ===")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 2 == 0:
                 print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'/content/alopecia_dataset/model_checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint guardado: {checkpoint_path}")

    final_model_path = '/content/alopecia_dataset/alopecia_segmentation_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Modelo final guardado: {final_model_path}")

    return model, train_losses

# ----------------------------
# 4. VISUALIZACIÓN MEJORADA (Mapas de calor)
# ----------------------------
def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    device = next(model.parameters()).device

    # Visualizamos 3 filas, 4 columnas
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

    for i in range(num_samples):
        img, true_mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits) # Convertimos logits a prob (0-1)
            pred_mask = (probs > 0.3).float() # Umbral bajado a 0.3

        img_np = img.permute(1, 2, 0).cpu().numpy()
        true_mask_np = true_mask[0].cpu().numpy()
        probs_np = probs[0][0].cpu().numpy()
        pred_mask_np = pred_mask[0][0].cpu().numpy()

        # Imagen
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Imagen {i+1}")
        axes[i, 0].axis('off')

        # Real
        axes[i, 1].imshow(true_mask_np, cmap='gray')
        axes[i, 1].set_title("Real")
        axes[i, 1].axis('off')

        # Calor (Probabilidades)
        im = axes[i, 2].imshow(probs_np, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title("Confianza (Mapa Calor)")
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Predicción binaria
        axes[i, 3].imshow(pred_mask_np, cmap='gray')
        axes[i, 3].set_title("Predicción (>0.3)")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO DE SEGMENTACIÓN")
    try:
        trained_model, losses = train_model()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.title('Curva de Aprendizaje')
        plt.grid(True)

        aligned_dir = '/content/alopecia_dataset/aligned_dataset'
        dataset = AlopeciaDataset(aligned_dir, img_size=256)

        print("\\n🎯 VISUALIZANDO PREDICCIONES DEL MODELO")
        visualize_predictions(trained_model, dataset, num_samples=min(3, len(dataset)))

        print("\\n✅ ¡ENTRENAMIENTO COMPLETADO!")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
"""

# Find the cell that contains 'class SimpleUNet'
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        if "class SimpleUNet" in source_text:
            # Replace the source
            # Split new_code into lines and add \n to each
            lines = new_code.split('\n')
            # Keeping the \n at the end of each line for valid ipynb format is safer
            cell['source'] = [line + '\n' for line in lines]
            cell['outputs'] = [] # Clear old outputs
            cell['execution_count'] = None
            print("Found and patched the model cell.")
            break

# Save to new file
with open('/home/sucode/alopecia_final/Test3_corregido.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
