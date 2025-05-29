import os
import cv2
import numpy as np
import sys

DATASET_FOLDERS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def is_grayscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Check if image has 3 channels (BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Calculate difference between original and converted image
        diff = cv2.subtract(img, bgr)
        # If all pixels are zero, the image is grayscale
        return not np.any(diff)
    # If image has only 1 channel, it's already grayscale
    return len(img.shape) == 2 or img.shape[2] == 1

def process_folder(folder_path):
    images_folder = os.path.join(folder_path, "images")
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return []
    
    non_grayscale_images = []
    total_images = 0
    grayscale_images = 0
    
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            total_images += 1
            image_path = os.path.join(images_folder, filename)
            if not is_grayscale(image_path):
                non_grayscale_images.append(filename)
            else:
                grayscale_images += 1
    
    print(f"Folder: {folder_path}")
    print(f"Total images: {total_images}")
    print(f"Grayscale images (to keep): {grayscale_images}")
    print(f"Non-grayscale images (to remove): {len(non_grayscale_images)}")
    
    return non_grayscale_images

def delete_images_and_labels(folder_path, image_filenames):
    images_folder = os.path.join(folder_path, "images")
    labels_folder = os.path.join(folder_path, "labels")
    
    deleted_images = 0
    deleted_labels = 0
    
    for filename in image_filenames:
        # Delete image
        image_path = os.path.join(images_folder, filename)
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_images += 1
        
        # Delete corresponding label
        # Assuming label has same name but different extension (.txt)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(labels_folder, label_filename)
        if os.path.exists(label_path):
            os.remove(label_path)
            deleted_labels += 1
    
    print(f"Deleted {deleted_images} images and {deleted_labels} label files")

def main():
    if len(sys.argv) > 1:
        folders_to_process = [sys.argv[1]]
    else:
        folders_to_process = DATASET_FOLDERS
    
    for folder in folders_to_process:
        if not os.path.exists(folder):
            print(f"Warning: Folder not found at {folder}, skipping")
            continue
        
        print(f"\nProcessing folder: {folder}")
        non_grayscale_images = process_folder(folder)
        
        if non_grayscale_images:
            user_input = input(f"Do you want to delete {len(non_grayscale_images)} non-grayscale images and their labels? (yes/no): ")
            if user_input.lower() == "yes":
                delete_images_and_labels(folder, non_grayscale_images)
                print("Deletion completed. Only grayscale images remain.")
            else:
                print("No files were deleted.")
        else:
            print("All images are already grayscale. Nothing to delete.")

if __name__ == "__main__":
    main()
