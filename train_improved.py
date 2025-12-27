import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# 1. IMPROVED LOSS FUNCTIONS
# ----------------------------
class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance de clases"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return dice

class CombinedLoss(nn.Module):
    """Combinación de Focal Loss + Dice Loss"""
    def __init__(self, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

# ----------------------------
# 2. METRICS
# ----------------------------
def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice Coefficient"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    return dice.item()

# ----------------------------
# 3. IMPROVED U-NET MODEL
# ----------------------------
class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 256 = 128 (up1) + 128 (enc2)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 128 = 64 (up2) + 64 (enc1)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Output
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)  # Corrected: concatenate with e2
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)  # Corrected: concatenate with e1
        d2 = self.dec2(d2)

        out = self.output(d2)
        return out

# ----------------------------
# 4. DATASET (reuse from train_model.py)
# ----------------------------
class AlopeciaDataset(Dataset):
    def __init__(self, aligned_dir, img_size=256):
        self.img_dir = os.path.join(aligned_dir, 'images')
        self.annot_dir = os.path.join(aligned_dir, 'annotations')
        self.img_size = img_size

        info_path = os.path.join(aligned_dir, 'annotations', 'dataset_info.json')
        if not os.path.exists(info_path):
             raise FileNotFoundError(f"Dataset info not found at {info_path}")

        with open(info_path, 'r') as f:
            self.dataset_info = json.load(f)

        coco_path = os.path.join(aligned_dir, 'annotations', 'filtered_coco.json')
        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)

        print(f"Dataset loaded: {len(self.dataset_info)} images, {sum(item['annotation_count'] for item in self.dataset_info)} annotations")

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        item = self.dataset_info[idx]
        img_path = os.path.join(self.img_dir, item['aligned_image'])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        mask = self._create_mask_from_coco(item['id'], self.img_size)
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

    def _create_mask_from_coco(self, image_id, size):
        mask = np.zeros((size, size), dtype=np.uint8)

        image_info = None
        for img_info in self.coco_data['images']:
            if img_info['id'] == image_id:
                image_info = img_info
                break
        if image_info is None:
            raise ValueError(f"Image info not found for image_id: {image_id}")

        img_width = image_info['width']
        img_height = image_info['height']

        for ann in self.coco_data['annotations']:
            if ann['image_id'] == image_id:
                seg = ann['segmentation']
                if isinstance(seg, list):
                    for polygon in seg:
                        pts = np.array(polygon).reshape(-1, 2)
                        pts = (pts * ([size / img_width, size / img_height])).astype(np.int32)
                        pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)

                        if len(pts) >= 3:
                            cv2.fillPoly(mask, [pts], 1)

        return mask

# ----------------------------
# 5. TRAINING FUNCTION
# ----------------------------
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    try:
        dataset = AlopeciaDataset(args.data_dir, img_size=256)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return None

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}\n")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = ImprovedUNet().to(device)
    print(f"Model: ImprovedUNet with BatchNorm and Dropout")
    
    # Loss
    criterion = CombinedLoss(focal_weight=1.0, dice_weight=1.0)
    print(f"Loss: Focal Loss + Dice Loss\n")

    # Optimizer with learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': []
    }
    
    best_iou = 0.0
    patience_counter = 0
    
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_iou_sum = 0
        val_dice_sum = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                
                probs = torch.sigmoid(outputs)
                
                val_loss += loss.item()
                val_iou_sum += calculate_iou(probs, masks)
                val_dice_sum += calculate_dice(probs, masks)
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_sum / len(val_loader)
        avg_val_dice = val_dice_sum / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        history['val_dice'].append(avg_val_dice)

        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val IoU: {avg_val_iou:.4f}")
        print(f"  Val Dice: {avg_val_dice:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_iou)
        
        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ New best model saved! IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-"*80)

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'alopecia_segmentation_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    print(f"✅ Best IoU achieved: {best_iou:.4f}")

    return model, history

# ----------------------------
# 6. VISUALIZATION
# ----------------------------
def plot_training_history(history, output_dir):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['val_dice'], label='Val Dice', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].set_title('Validation Dice Coefficient')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(history['val_iou'], label='IoU', color='green')
    axes[1, 1].plot(history['val_dice'], label='Dice', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Training history plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Improved Alopecia Segmentation Training")
    parser.add_argument('--data_dir', type=str, default='data/alopecia_dataset/augmented_dataset', 
                        help='Path to augmented dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/models/v0.2', 
                        help='Path to save models and results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    model, history = train_model(args)

    if model and history:
        # Plot training history
        plot_training_history(history, args.output_dir)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)

if __name__ == "__main__":
    main()
