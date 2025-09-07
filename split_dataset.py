import os
import shutil
import random
import json
from PIL import Image
import config

def validate_image(image_path):
    """Validate if image file is readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def split_dataset():
    """Split dataset into train/val/test"""
    print("🚀 Starting dataset split...")
    
    if not os.path.exists(config.RAW_DATASET_DIR):
        print(f"❌ Dataset directory not found: {config.RAW_DATASET_DIR}")
        print("Please put your dataset in the 'Dataset/raw' folder")
        return False
    
    random.seed(config.RANDOM_SEED)
    
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(config.PROCESSED_DATASET_DIR, split), exist_ok=True)
    
    breed_stats = {}
    total_images = 0
    
    # Process each breed folder
    breed_folders = [f for f in os.listdir(config.RAW_DATASET_DIR) 
                    if os.path.isdir(os.path.join(config.RAW_DATASET_DIR, f))]
    
    print(f"📁 Found {len(breed_folders)} breed folders")
    
    for breed in sorted(breed_folders):
        breed_dir = os.path.join(config.RAW_DATASET_DIR, breed)
        
        # Get valid images
        all_files = os.listdir(breed_dir)
        valid_images = []
        
        for file in all_files:
            file_path = os.path.join(breed_dir, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if validate_image(file_path):
                    valid_images.append(file)
        
        if len(valid_images) < 10:
            print(f"⚠️  Warning: {breed} has only {len(valid_images)} valid images")
        
        # Shuffle and split
        random.shuffle(valid_images)
        n_total = len(valid_images)
        n_train = int(n_total * config.TRAIN_SPLIT)
        n_val = int(n_total * config.VAL_SPLIT)
        
        train_imgs = valid_images[:n_train]
        val_imgs = valid_images[n_train:n_train + n_val]
        test_imgs = valid_images[n_train + n_val:]
        
        # Copy images
        for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_breed_dir = os.path.join(config.PROCESSED_DATASET_DIR, split, breed)
            os.makedirs(split_breed_dir, exist_ok=True)
            
            for img in split_imgs:
                src = os.path.join(breed_dir, img)
                dst = os.path.join(split_breed_dir, img)
                shutil.copy2(src, dst)
        
        breed_stats[breed] = {
            'total': n_total,
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs)
        }
        
        total_images += n_total
        print(f"✅ {breed}: {n_total} images → {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Save statistics
    with open('dataset_stats.json', 'w') as f:
        json.dump({
            'total_breeds': len(breed_folders),
            'total_images': total_images,
            'breed_stats': breed_stats
        }, f, indent=2)
    
    print(f"🎉 Dataset split completed! {len(breed_folders)} breeds, {total_images} images")
    return True

if __name__ == "__main__":
    split_dataset()
