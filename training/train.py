from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import torch
import pandas as pd

def main():
    model_path = 'yolov8n-seg.pt'
    epochs = 50
    batch_size = 16
    img_size = 160
    device = ''
    workers = 4
    project = 'runs/segment'
    name = 'human_segmentation'
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    model = YOLO(model_path)
    
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    results = model.val()
    
    print("Training complete. Model saved to:", os.path.join(project, name))
    
    results_path = os.path.join(project, name, 'results.csv')
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path)
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(results_df['epoch'], results_df['train/box_loss'], label='box_loss')
            plt.plot(results_df['epoch'], results_df['train/seg_loss'], label='seg_loss')
            plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='cls_loss')
            plt.plot(results_df['epoch'], results_df['train/dfl_loss'], label='dfl_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision')
            plt.plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall')
            plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP50')
            plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.title('Object Detection Metrics')
            plt.legend()
            
            if 'metrics/mAP50-95(M)' in results_df.columns:
                plt.subplot(2, 2, 3)
                plt.plot(results_df['epoch'], results_df['metrics/mAP50(M)'], label='mAP50 (Mask)')
                plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(M)'], label='mAP50-95 (Mask)')
                plt.xlabel('Epoch')
                plt.ylabel('Metric')
                plt.title('Segmentation Metrics')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(project, name, 'training_results.png'))
            print(f"Training results plot saved to: {os.path.join(project, name, 'training_results.png')}")
        except Exception as e:
            print(f"Error plotting results: {e}")

if __name__ == '__main__':
    main()
