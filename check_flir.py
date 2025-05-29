import os
import sys

DATASET_FOLDERS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def is_flir_image(filename):
    return "FLIR" in filename

def process_folder(folder_path):
    images_folder = os.path.join(folder_path, "images")
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found at {images_folder}")
        return []
    
    non_flir_images = []
    total_images = 0
    flir_images = 0
    
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            total_images += 1
            if not is_flir_image(filename):
                non_flir_images.append(filename)
            else:
                flir_images += 1
    
    print(f"Folder: {folder_path}")
    print(f"Total images: {total_images}")
    print(f"FLIR images (to keep): {flir_images}")
    print(f"Non-FLIR images (to remove): {len(non_flir_images)}")
    
    return non_flir_images

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
        non_flir_images = process_folder(folder)
        
        if non_flir_images:
            user_input = input(f"Do you want to delete {len(non_flir_images)} non-FLIR images and their labels? (yes/no): ")
            if user_input.lower() == "yes":
                delete_images_and_labels(folder, non_flir_images)
                print("Deletion completed. Only FLIR images remain.")
            else:
                print("No files were deleted.")
        else:
            print("All images are already FLIR images. Nothing to delete.")

if __name__ == "__main__":
    main()
