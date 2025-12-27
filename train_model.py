import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

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
        if not os.path.exists(info_path):
             raise FileNotFoundError(f"Dataset info not found at {info_path}. Did you run process_dataset.py?")

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
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}. Asegúrate de que el archivo existe.")

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
                # Depending on how the data was saved, segmentation might be a list of lists or a single list
                # process_dataset.py saves as [ [x1, y1, x2, y2, ...] ] (list of lists of floats)
                if isinstance(seg, list):
                    for polygon in seg:
                         # polygon should be a list of numbers
                        pts = np.array(polygon).reshape(-1, 2)
                        
                        # Scale points to the target size
                        # Note: pts are in original image coordinates
                        pts = (pts * ([size / img_width, size / img_height])).astype(np.int32)
                        
                        # Ensure points are within bounds
                        pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)

                        # Only fill if there are enough points to form a polygon
                        if len(pts) >= 3:
                            cv2.fillPoly(mask, [pts], 1)

        return mask

# ----------------------------
# 2. MODELO
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
        return out

# ----------------------------
# 3. ENTRENAMIENTO
# ----------------------------
def train_model(args):
    aligned_dir = args.data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    try:
        dataset = AlopeciaDataset(aligned_dir, img_size=256)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return None, []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  # batch_size=1 para CPU

    model = SimpleUNet().to(device)

    # Weighted loss for class imbalance
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    num_epochs = args.epochs
    train_losses = []

    print(f"\n=== COMIENZO DEL ENTRENAMIENTO ({num_epochs} epochs) ===")

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

            # Print every 20 batches to reduce output
            if batch_idx % 20 == 0:
                 progress = (batch_idx / len(dataloader)) * 100
                 print(f"  Epoch {epoch+1}/{num_epochs} [{progress:.1f}%] Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'model_checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint guardado: {checkpoint_path}")

    # Save final
    final_model_path = os.path.join(args.output_dir, 'alopecia_segmentation_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Modelo final guardado: {final_model_path}")

    return model, train_losses

# ----------------------------
# 4. VISUALIZACIÓN
# ----------------------------
def visualize_predictions(model, dataset, args):
    model.eval()
    device = next(model.parameters()).device
    
    num_samples = min(args.visualize_count, len(dataset))
    if num_samples == 0:
        return

    # Use a non-gui backend if needed, or just save images
    # plt.switch_backend('Agg') 

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    print(f"\nGenerando visualizaciones para {num_samples} muestras...")

    for i in range(num_samples):
        img, true_mask = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.3).float()

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

        # Calor
        im = axes[i, 2].imshow(probs_np, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title("Confianza")
        axes[i, 2].axis('off')
        
        # Predicción
        axes[i, 3].imshow(pred_mask_np, cmap='gray')
        axes[i, 3].set_title("Predicción (>0.3)")
        axes[i, 3].axis('off')

    plt.tight_layout()
    viz_path = os.path.join(args.output_dir, 'predictions_visualization.png')
    plt.savefig(viz_path)
    print(f"✅ Visualización guardada en: {viz_path}")
    # plt.show() # Commented out for script execution environment

def main():
    parser = argparse.ArgumentParser(description="Alopecia Segmentation Training")
    parser.add_argument('--data_dir', type=str, default='alopecia_dataset/augmented_dataset', help='Path to augmented dataset')
    parser.add_argument('--output_dir', type=str, default='models', help='Path to save models and results')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--visualize_count', type=int, default=3, help='Number of samples to visualize after training')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    trained_model, losses = train_model(args)

    if trained_model:
        # Plot loss
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(args.output_dir, 'training_loss.png')
        plt.savefig(loss_plot_path)
        print(f"✅ Gráfica de pérdida guardada en: {loss_plot_path}")

        # Visualizations
        dataset = AlopeciaDataset(args.data_dir, img_size=256)
        visualize_predictions(trained_model, dataset, args)

if __name__ == "__main__":
    main()
