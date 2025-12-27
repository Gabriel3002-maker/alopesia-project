
import os
import json
import shutil
import zipfile
import glob
from pathlib import Path
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def setup_directories(base_dir):
    """Creates necessary directories and cleans up previous runs."""
    extract_dir = base_dir / 'temp_extract'
    yolo_dir = base_dir / 'yolo_dataset'
    augmented_output_dir = base_dir / 'alopecia_dataset' / 'augmented_dataset'

    # Clean up previous runs if they exist, but be careful
    # shutil.rmtree(extract_dir, ignore_errors=True) # Let's keep extract dir for now or clear it
    if extract_dir.exists():
         shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    (yolo_dir / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels').mkdir(parents=True, exist_ok=True)

    if augmented_output_dir.exists():
        shutil.rmtree(augmented_output_dir)
    (augmented_output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (augmented_output_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    return extract_dir, yolo_dir, augmented_output_dir

def extract_zip(zip_path, extract_dir):
    """Extracts the dataset ZIP file."""
    print(f"📦 Extracting ZIP: {zip_path}")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Extracted to: {extract_dir}")

    # Find JSON
    json_files = list(Path(extract_dir).rglob('*.json'))
    if not json_files:
        raise FileNotFoundError("❌ No JSON found in ZIP")
    json_path = json_files[0]
    print(f"✅ JSON found: {json_path}")

    # Find Images Directory
    images_dir = None
    possible_dirs = ['images', 'img', 'upload', 'data', 'media']
    for dir_name in possible_dirs:
        dir_path = Path(extract_dir) / dir_name
        if dir_path.exists() and any(dir_path.iterdir()):
            images_dir = dir_path
            break
            
    if not images_dir:
        # Search recursively
        for root, dirs, files in os.walk(extract_dir):
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                images_dir = Path(root)
                break
    
    if not images_dir:
        raise FileNotFoundError("❌ No images directory found")
        
    print(f"✅ Images directory: {images_dir}")
    return json_path, images_dir

def load_coco_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded COCO data: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations")
    return data

def convert_to_yolo(coco_data, images_dir, yolo_dir):
    """Converts COCO annotations to YOLO format."""
    print("\n--- Starting YOLO Conversion ---")
    
    class_names = [cat['name'] for cat in coco_data.get('categories', [])]
    class_id_map = {cat['id']: i for i, cat in enumerate(coco_data.get('categories', []))}
    
    # helper to find image file
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG']:
        all_images.extend(list(images_dir.rglob(ext)))

    processed_count = 0
    
    for img_data in coco_data.get('images', []):
        image_id = img_data['id']
        file_name = img_data.get('file_name')
        img_width = img_data.get('width')
        img_height = img_data.get('height')
        
        if not file_name or img_width is None:
            continue
            
        # Find image file
        clean_name = os.path.basename(file_name).split('?')[0]
        image_path = None
        
        # Exact match
        for p in all_images:
            if p.name == clean_name:
                image_path = p
                break
        
        if not image_path:
             # Try partial
             base_name = os.path.splitext(clean_name)[0]
             for p in all_images:
                 if os.path.splitext(p.name)[0] == base_name:
                     image_path = p
                     break
        
        if image_path:
             # Copy image
             ext = image_path.suffix
             dest_img_path = yolo_dir / 'images' / f"{image_id}{ext}"
             shutil.copy(image_path, dest_img_path)
             
             # Create Label File
             label_path = yolo_dir / 'labels' / f"{image_id}.txt"
             with open(label_path, 'w') as f_txt:
                 image_anns = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] == image_id]
                 for ann in image_anns:
                     if 'bbox' in ann:
                         x, y, w, h = ann['bbox']
                         cx = (x + w/2) / img_width
                         cy = (y + h/2) / img_height
                         nw = w / img_width
                         nh = h / img_height
                         
                         cat_id = ann['category_id']
                         if cat_id in class_id_map:
                             class_idx = class_id_map[cat_id]
                             f_txt.write(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
             
             processed_count += 1
             
    # Create dataset.yaml
    with open(yolo_dir / 'dataset.yaml', 'w') as f:
        f.write(f"path: {yolo_dir}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
        
    print(f"✅ Processed {processed_count} images for YOLO.")
    return all_images

def augment_data(coco_data, all_images, output_dir):
    """Generates augmented dataset."""
    print("\n--- Starting Data Augmentation ---")
    
    images_out = output_dir / 'images'
    anns_out = output_dir / 'annotations'
    
    ia.seed(1)
    
    # Augmentation Sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        ),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Sometimes(0.2, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))),
        iaa.Sometimes(0.2, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
        iaa.Sometimes(0.2, iaa.EdgeDetect(alpha=(0.0, 0.7))),
        iaa.Sometimes(0.2, iaa.MotionBlur(k=(3, 7), angle=[-45, 45])),
        iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20)))
    ], random_order=True)

    augmented_coco = {
        "images": [],
        "annotations": [],
        "categories": coco_data.get("categories", []),
        "info": coco_data.get("info", {})
    }
    
    # IDs
    new_img_id = max([x['id'] for x in coco_data['images']]) + 1 if coco_data['images'] else 0
    new_ann_id = max([x['id'] for x in coco_data['annotations']]) + 1 if coco_data['annotations'] else 0
    
    num_aug = 19  # Increased from 9 to generate more training data
    
    for img_info in coco_data.get('images', []):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Find path (reuse logic or map)
        clean_name = os.path.basename(file_name).split('?')[0]
        img_path = None
        for p in all_images:
            if p.name == clean_name:
                img_path = p
                break
        if not img_path:
             base_name = os.path.splitext(clean_name)[0]
             for p in all_images:
                 if os.path.splitext(p.name)[0] == base_name:
                     img_path = p
                     break
                     
        if not img_path:
            print(f"Skipping {file_name}: Not found")
            continue
            
        image = cv2.imread(str(img_path))
        if image is None: 
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        anns = [a for a in coco_data.get('annotations', []) if a['image_id'] == img_id]
        
        # Prepare polygons
        polygons = []
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1, 2)
                    polygons.append(ia.Polygon(pts, label=ann['category_id']))
        
        # Save original image to output directory
        orig_fname = img_info['file_name']
        cv2.imwrite(str(images_out / orig_fname), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Add original to new dataset
        augmented_coco['images'].append(img_info)
        augmented_coco['annotations'].extend(anns)
        
        if not polygons:
            continue
            
        # Augment
        for i in range(num_aug):
            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_image(image)
            
            # Wrap polygons with image shape for imgaug
            from imgaug.augmentables.polys import PolygonsOnImage
            polys_on_img = PolygonsOnImage(polygons, shape=image.shape)
            polys_aug_on_img = seq_det.augment_polygons(polys_on_img)
            polys_aug = polys_aug_on_img.polygons
            
            aug_fname = f"{Path(file_name).stem}_aug{i}{Path(file_name).suffix}"
            
            # Save Image
            cv2.imwrite(str(images_out / aug_fname), cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
            
            # Add Image Info
            current_img_id = new_img_id
            new_img_id += 1
            
            augmented_coco['images'].append({
                "id": current_img_id,
                "file_name": aug_fname,
                "width": img_info['width'],
                "height": img_info['height']
            })
            
            # Add Annotations
            for poly in polys_aug:
                if poly.is_valid and poly.area > 0 and len(poly.exterior) >= 3:
                     # Get bounding box
                     bbox_obj = poly.to_bounding_box()
                     x1, y1, x2, y2 = bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2
                     bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                     
                     # Convert polygon to segmentation format
                     seg_points = np.column_stack([poly.exterior[:, 0], poly.exterior[:, 1]]).flatten().tolist()
                     seg_points = [float(p) for p in seg_points]
                     
                     augmented_coco['annotations'].append({
                         "id": new_ann_id,
                         "image_id": current_img_id,
                         "category_id": int(poly.label),
                         "segmentation": [seg_points],
                         "area": float(poly.area),
                         "bbox": bbox,
                         "iscrowd": 0
                     })
                     new_ann_id += 1
                     
    # Save COCO JSON
    with open(anns_out / 'filtered_coco.json', 'w') as f:
        json.dump(augmented_coco, f, indent=4)
    
    # Create dataset_info.json for training
    dataset_info = []
    for img in augmented_coco['images']:
        img_id = img['id']
        ann_count = sum(1 for ann in augmented_coco['annotations'] if ann['image_id'] == img_id)
        dataset_info.append({
            'id': img_id,
            'aligned_image': img['file_name'],
            'annotation_count': ann_count
        })
    
    with open(anns_out / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"✅ Augmentation Complete. Total Images: {len(augmented_coco['images'])}")
    print(f"✅ Saved filtered_coco.json and dataset_info.json")


import argparse

def main():
    parser = argparse.ArgumentParser(description="Alopecia Dataset Processing")
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory for processing')
    parser.add_argument('--zip_file', type=str, default='test-coco-upload.zip', help='Name of the ZIP file')
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    zip_path = base_dir / args.zip_file
    
    try:
        extract_dir, yolo_dir, aug_dir = setup_directories(base_dir)
        json_path, images_dir = extract_zip(zip_path, extract_dir)
        coco_data = load_coco_data(json_path)
        
        all_images_paths = convert_to_yolo(coco_data, images_dir, yolo_dir)
        augment_data(coco_data, all_images_paths, aug_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
