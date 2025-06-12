from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

def visualize_predictions(image_path, results, save_dir, show=False):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = results[0]
    
    vis_img = img.copy()
    
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(masks))]
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            binary_mask = mask.astype(np.uint8)
            
            if binary_mask.shape[:2] != img.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))
            
            color_mask = np.zeros_like(img)
            color_mask[binary_mask > 0] = colors[i]
            
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, 1, color_mask, alpha, 0)
            
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), colors[i], 2)
            
            cv2.putText(vis_img, f"Human: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, os.path.basename(image_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def main():
    model_path = 'runs/segment/human_segmentation/weights/best.pt'
    source = 'val/images'
    conf = 0.25
    iou = 0.7
    max_det = 300
    img_size = 160
    device = ''
    save_dir = 'predictions'
    save_txt = False
    save_conf = False
    save_crop = False
    show = False
    classes = [0]  # Only detect humans
    visualize = False
    augment = False
    agnostic_nms = False
    retina_masks = True
    line_width = None
    
    model = YOLO(model_path)
    
    os.makedirs(save_dir, exist_ok=True)
    
    source_path = Path(source)
    if source_path.is_dir():
        image_paths = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg'))
    else:
        image_paths = [source_path]
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=img_size,
            device=device,
            save=False,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            show=False,
            classes=classes,
            visualize=visualize,
            augment=augment,
            agnostic_nms=agnostic_nms,
            retina_masks=retina_masks,
            line_width=line_width,
        )
        
        output_path = visualize_predictions(image_path, results, save_dir, show)
        print(f"Saved prediction to {output_path}")
    
    print(f"All predictions saved to {save_dir}")

if __name__ == '__main__':
    main()
