import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load COCO data
coco_path = 'data/alopecia_dataset/augmented_dataset/annotations/filtered_coco.json'
with open(coco_path, 'r') as f:
    coco_data = json.load(f)

images_dir = Path('data/alopecia_dataset/augmented_dataset/images')

# Get first few original images (non-augmented)
original_images = [img for img in coco_data['images'] if '_aug' not in img['file_name']][:5]

print(f"Total images: {len(coco_data['images'])}")
print(f"Original images: {len([img for img in coco_data['images'] if '_aug' not in img['file_name']])}")
print(f"Visualizing first {len(original_images)} original images\n")

fig, axes = plt.subplots(len(original_images), 3, figsize=(15, 5*len(original_images)))
if len(original_images) == 1:
    axes = axes.reshape(1, -1)

for idx, img_info in enumerate(original_images):
    img_id = img_info['id']
    img_path = images_dir / img_info['file_name']
    
    # Load image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create mask
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    
    # Get annotations for this image
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    print(f"Image {idx+1}: {img_info['file_name']}")
    print(f"  Size: {img_info['width']}x{img_info['height']}")
    print(f"  Annotations: {len(anns)}")
    
    # Draw polygons on mask
    for ann in anns:
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                print(f"    Polygon with {len(pts)} points, area: {ann.get('area', 'N/A')}")
    
    # Calculate mask coverage
    coverage = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
    print(f"  Mask coverage: {coverage:.2f}%\n")
    
    # Original image
    axes[idx, 0].imshow(img_rgb)
    axes[idx, 0].set_title(f'Original: {img_info["file_name"][:30]}')
    axes[idx, 0].axis('off')
    
    # Mask
    axes[idx, 1].imshow(mask, cmap='gray')
    axes[idx, 1].set_title(f'Mask (Coverage: {coverage:.1f}%)')
    axes[idx, 1].axis('off')
    
    # Overlay
    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red for alopecia areas
    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
    axes[idx, 2].imshow(blended)
    axes[idx, 2].set_title('Overlay (Red = Alopecia)')
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/dataset_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: outputs/dataset_visualization.png")
