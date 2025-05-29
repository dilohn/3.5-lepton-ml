import os
import torch
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
workers = min(8, os.cpu_count() or 1)
batch_size = 16 if device_count > 0 else 8

results = model.train(
    data='dataset.yaml',
    epochs=100,
    patience=20,
    batch=batch_size,
    imgsz=640,
    workers=workers,
    device=0 if device_count > 0 else 'cpu',
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
    name='yolov8_thermal_people',
    save=True,
    save_period=-1,
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
