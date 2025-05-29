import os
import torch
import platform
import psutil
from ultralytics import YOLO
import multiprocessing

def get_system_info():
    cpu_count = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    gpu_info = []
    gpu_count = 0
    gpu_memory_total = 0
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory / (1024 ** 3)
            gpu_memory_total += gpu_memory
            gpu_info.append({
                'name': props.name,
                'memory_gb': gpu_memory,
                'compute_capability': f"{props.major}.{props.minor}"
            })
    
    return {
        'platform': platform.system(),
        'cpu_count': cpu_count,
        'ram_gb': ram_gb,
        'gpu_count': gpu_count,
        'gpu_memory_total_gb': gpu_memory_total,
        'gpu_info': gpu_info
    }

def optimize_training_params(system_info, model_size):
    params = {}
    
    if system_info['gpu_count'] > 0:
        params['device'] = [i for i in range(system_info['gpu_count'])]
        
        memory_per_gpu = system_info['gpu_memory_total_gb'] / system_info['gpu_count']
        
        model_memory_map = {
            'n': 1,    # nano model base memory requirement in GB
            's': 2,    # small model
            'm': 4,    # medium model
            'l': 8,    # large model
            'x': 12    # xlarge model
        }
        
        base_memory = model_memory_map.get(model_size, 2)
        available_memory = memory_per_gpu - base_memory
        
        if available_memory > 0:
            batch_multiplier = min(4, available_memory / base_memory)
            base_batch = {
                'n': 16,
                's': 12,
                'm': 8,
                'l': 4,
                'x': 2
            }.get(model_size, 8)
            
            params['batch'] = max(1, int(base_batch * batch_multiplier * system_info['gpu_count']))
        else:
            params['batch'] = 1
            
        params['workers'] = min(8, system_info['cpu_count'])
        params['amp'] = True  # Mixed precision training
    else:
        params['device'] = 'cpu'
        params['batch'] = 8
        params['workers'] = min(4, system_info['cpu_count'])
        params['amp'] = False
    
    return params

def main():
    # Configuration - modify these values as needed
    model_size = 'm'  # Choose from: 'n', 's', 'm', 'l', 'x'
    epochs = 100
    img_size = 640

    system_info = get_system_info()
    print(system_info)
    model = YOLO(f'yolov8{model_size}.pt')

    training_params = optimize_training_params(system_info, model_size)

    results = model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=img_size,
        patience=20,
        optimizer='auto',
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
        rect=True,
        resume=False,
        close_mosaic=10,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        **training_params
    )

    best_weights = model.best if hasattr(model, 'best') else model.last
    model.export(format='onnx', dynamic=True, simplify=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
