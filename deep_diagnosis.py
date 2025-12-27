import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from train_model import SimpleUNet, AlopeciaDataset

# Load model
model_path = 'outputs/models/v0.1/alopecia_segmentation_model.pth'
device = torch.device('cpu')
model = SimpleUNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load dataset
dataset = AlopeciaDataset('data/alopecia_dataset/augmented_dataset', img_size=256)

# Get first few original images
coco_path = 'data/alopecia_dataset/augmented_dataset/annotations/filtered_coco.json'
with open(coco_path, 'r') as f:
    coco_data = json.load(f)

original_images = [img for img in coco_data['images'] if '_aug' not in img['file_name']][:3]
original_ids = [img['id'] for img in original_images]

# Find indices in dataset
indices = []
for img_id in original_ids:
    for idx, item in enumerate(dataset.dataset_info):
        if item['id'] == img_id:
            indices.append(idx)
            break

print(f"Analyzing {len(indices)} original images\n")
print("="*80)

fig, axes = plt.subplots(len(indices), 5, figsize=(20, 4*len(indices)))
if len(indices) == 1:
    axes = axes.reshape(1, -1)

for plot_idx, dataset_idx in enumerate(indices):
    # Get data from dataset (how it's fed to the model during training)
    img_tensor, mask_tensor = dataset[dataset_idx]
    
    # Get prediction
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(device)
        logits = model(img_input)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.3).float()
    
    # Convert to numpy
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    true_mask_np = mask_tensor[0].cpu().numpy()
    probs_np = probs[0][0].cpu().numpy()
    pred_mask_np = pred_mask[0][0].cpu().numpy()
    
    # Calculate statistics
    true_coverage = (np.sum(true_mask_np > 0) / (true_mask_np.shape[0] * true_mask_np.shape[1])) * 100
    pred_coverage = (np.sum(pred_mask_np > 0) / (pred_mask_np.shape[0] * pred_mask_np.shape[1])) * 100
    
    # Calculate overlap (IoU)
    intersection = np.sum((true_mask_np > 0) & (pred_mask_np > 0))
    union = np.sum((true_mask_np > 0) | (pred_mask_np > 0))
    iou = intersection / union if union > 0 else 0
    
    # Calculate inverted IoU (to check if prediction is inverted)
    inverted_pred = 1 - pred_mask_np
    inv_intersection = np.sum((true_mask_np > 0) & (inverted_pred > 0))
    inv_union = np.sum((true_mask_np > 0) | (inverted_pred > 0))
    inv_iou = inv_intersection / inv_union if inv_union > 0 else 0
    
    item_info = dataset.dataset_info[dataset_idx]
    
    print(f"\nImage {plot_idx + 1}: {item_info['aligned_image'][:50]}")
    print(f"  True Mask Coverage: {true_coverage:.2f}%")
    print(f"  Predicted Coverage: {pred_coverage:.2f}%")
    print(f"  IoU (normal): {iou:.4f}")
    print(f"  IoU (inverted): {inv_iou:.4f}")
    
    if inv_iou > iou:
        print(f"  ⚠️  WARNING: Inverted IoU is HIGHER! Prediction appears inverted!")
    
    # Plot
    axes[plot_idx, 0].imshow(img_np)
    axes[plot_idx, 0].set_title(f'Input Image')
    axes[plot_idx, 0].axis('off')
    
    axes[plot_idx, 1].imshow(true_mask_np, cmap='gray')
    axes[plot_idx, 1].set_title(f'Ground Truth\n({true_coverage:.1f}%)')
    axes[plot_idx, 1].axis('off')
    
    axes[plot_idx, 2].imshow(probs_np, cmap='jet', vmin=0, vmax=1)
    axes[plot_idx, 2].set_title('Model Confidence')
    axes[plot_idx, 2].axis('off')
    
    axes[plot_idx, 3].imshow(pred_mask_np, cmap='gray')
    axes[plot_idx, 3].set_title(f'Prediction\n({pred_coverage:.1f}%)\nIoU: {iou:.3f}')
    axes[plot_idx, 3].axis('off')
    
    axes[plot_idx, 4].imshow(inverted_pred, cmap='gray')
    axes[plot_idx, 4].set_title(f'Inverted Pred\nIoU: {inv_iou:.3f}')
    axes[plot_idx, 4].axis('off')

plt.tight_layout()
plt.savefig('outputs/deep_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"✅ Deep diagnosis saved to: outputs/deep_diagnosis.png")
