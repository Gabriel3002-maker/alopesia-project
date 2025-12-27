import os
import shutil
import argparse
from pathlib import Path
import process_dataset
import train_model

def organize_project():
    """Sets up the directory structure and moves files if necessary."""
    base_dir = Path.cwd()
    
    # Define directories
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'outputs'
    models_dir = output_dir / 'models'
    
    # Create them
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Project structure: \n  - Data: {data_dir}\n  - Models: {models_dir}")
    
    # Move ZIP file if it's in the root
    zip_name = 'test-coco-upload.zip'
    root_zip = base_dir / zip_name
    data_zip = data_dir / zip_name
    
    if root_zip.exists():
        print(f"🚚 Moving {zip_name} to {data_dir}...")
        shutil.move(str(root_zip), str(data_zip))
    elif not data_zip.exists():
        print(f"⚠️ Warning: {zip_name} not found in root or data directory.")
    
    return data_dir, models_dir, zip_name

def main():
    parser = argparse.ArgumentParser(description="Alopecia Project Orchestrator")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5 for CPU)')
    parser.add_argument('--skip_processing', action='store_true', help='Skip data processing step')
    args = parser.parse_args()
    
    # 1. Organize Folders
    data_dir, models_dir, zip_name = organize_project()
    
    # 2. Process Data
    augmented_dir = data_dir / 'alopecia_dataset' / 'augmented_dataset'
    
    if not args.skip_processing:
        print("\n" + "="*40)
        print("🏗️  STEP 1: Processing Dataset")
        print("="*40)
        
        # We call the functions from process_dataset directly or via its main if adapted.
        # Since we want to pass specific paths, let's look at how process_dataset works.
        # It uses argparse in main(), but we can import functions.
        
        try:
            extract_dir, yolo_dir, aug_dir = process_dataset.setup_directories(data_dir)
            json_path, images_dir = process_dataset.extract_zip(data_dir / zip_name, extract_dir)
            coco_data = process_dataset.load_coco_data(json_path)
            
            all_images_paths = process_dataset.convert_to_yolo(coco_data, images_dir, yolo_dir)
            process_dataset.augment_data(coco_data, all_images_paths, aug_dir)
            
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            return
    else:
        print("\nSkipping data processing as requested.")

    # 3. Train Model
    print("\n" + "="*40)
    print("🚀 STEP 2: Training Model")
    print("="*40)
    
    # Prepare arguments for train_model
    # train_model expects an 'args' object (Namespace)
    class TrainArgs:
        data_dir = str(augmented_dir)
        output_dir = str(models_dir)
        epochs = args.epochs
        batch_size = 4
        learning_rate = 0.001
        visualize_count = 3
        
    train_args = TrainArgs()
    
    try:
        model, losses = train_model.train_model(train_args)
        
        if model:
            # Visualization
            dataset = train_model.AlopeciaDataset(train_args.data_dir, img_size=256)
            train_model.visualize_predictions(model, dataset, train_args)
            
            # Save loss plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            loss_plot_path = Path(train_args.output_dir) / 'training_loss.png'
            plt.savefig(loss_plot_path)
            print(f"✅ Loss plot saved to: {loss_plot_path}")
            
    except Exception as e:
        print(f"❌ Error in training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
