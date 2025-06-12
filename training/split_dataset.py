import os
import shutil
import random
from pathlib import Path
import yaml

def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_path in file_list:
        base_name = img_path.stem
        label_path = src_lbl_dir / f"{base_name}.txt"
        if label_path.exists():
            shutil.copy(img_path, dst_img_dir / img_path.name)
            shutil.copy(label_path, dst_lbl_dir / label_path.name)
        else:
            print(f"Warning: Label not found for {img_path}")

def main():
    random.seed(42)
    images_dir = Path('images')
    labels_dir = Path('labels')
    train_images_dir = Path('train/images')
    train_labels_dir = Path('train/labels')
    val_images_dir = Path('val/images')
    val_labels_dir = Path('val/labels')

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*.png'))
    print(f"Total images found: {len(image_files)}")

    random.shuffle(image_files)

    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
    copy_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

    with open('data.yaml', 'r') as f:
        data_yaml = yaml.safe_load(f)

    data_yaml['train'] = 'train/images'
    data_yaml['val'] = 'val/images'

    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("Dataset split complete and data.yaml updated.")

if __name__ == "__main__":
    main()
