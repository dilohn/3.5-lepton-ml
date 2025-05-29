import os
import torch
from ultralytics import YOLO

def get_optimal_batch_size(model_size):
    if not torch.cuda.is_available():
        return 8
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    if model_size == 'n':
        return int(min(64, max(16, gpu_memory * 4)))
    elif model_size == 's':
        return int(min(48, max(8, gpu_memory * 2)))
    elif model_size == 'm':
        return int(min(32, max(8, gpu_memory * 1.5)))
    elif model_size == 'l':
        return int(min(24, max(4, gpu_memory)))
    elif model_size == 'x':
        return int(min(16, max(2, gpu_memory * 0.5)))
    else:
        return 16

model_size = 'n'  # Choose from: 'n', 's', 'm', 'l', 'x'
epochs = 100
img_size = 640
batch_size = None  # Auto-determine based on GPU memory

model_name = f'yolov8{model_size}.pt'
model = YOLO(model_name)

device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
workers = min(8, os.cpu_count() or 1)

if batch_size is None:
    batch_size = get_optimal_batch_size(model_size)

device = 'cpu'
if device_count > 0:
    device = [i for i in range(device_count)]

results = model.train(
    data='dataset.yaml',
    epochs=epochs,
    patience=20,
    batch=batch_size,
    imgsz=img_size,
    workers=workers,
    device=device,
    optimizer='auto',
    amp=True,
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    pretrained=True,
    verbose=True,
    seed=0,
    exist_ok=True,
    project='runs/train',
    name=f'yolov8{model_size}_thermal_people',
    save=True,
    save_period=10,
    cache=False,
    rect=False,
    resume=False,
    close_mosaic=10,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True,
    plots=True
)

model.export(format='onnx', dynamic=True, simplify=True)
