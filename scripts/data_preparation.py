import os
import shutil
from sklearn.model_selection import train_test_split
import cv2

def prepare_dataset(data_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Organize dataset into train/val/test splits
    """
    # Create directories for splits
    splits = ['train', 'val', 'test']
    classes = ['healthy_corals', 'bleached_corals']
    
    for split in splits:
        for cls in classes:
            os.makedirs(f'data/{split}/{cls}', exist_ok=True)
    
    # Split data for each class
    for cls in classes:
        images = os.listdir(os.path.join(data_dir, cls))
        
        # First split: train and temp (val + test)
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio)
        
        # Second split: val and test from temp
        val_size = val_ratio / (1 - train_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_size)
        
        # Copy images to respective directories
        for img in train_imgs:
            shutil.copy(os.path.join(data_dir, cls, img), 
                       os.path.join('data/train', cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(data_dir, cls, img), 
                       os.path.join('data/val', cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(data_dir, cls, img), 
                       os.path.join('data/test', cls, img))
