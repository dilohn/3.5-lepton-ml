import clr
import sys
import os
import time
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque
import pythoncom

pythoncom.CoInitialize()
from import_clr import *
clr.AddReference("ManagedIR16Filters")
clr.AddReference("LeptonUVC")
from Lepton import CCI
from IR16Filters import IR16Capture, NewIR16FrameEvent, NewBytesFrameEvent

import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = r"best.pt"
MODEL_CONF = 0.5
MODEL_CLASSES = [0]
REPORT_TIME = 3
CALIBRATION_DELAY = 10
CALIBRATION_DURATION = 10

print("Loading YOLOv8 model from best.pt")
model = YOLO(MODEL_PATH)
model.conf = MODEL_CONF
model.classes = MODEL_CLASSES
print("YOLOv8 model loaded successfully")

found_device = None
for dev in CCI.GetDevices():
    print(dev)
    if dev.Name.startswith("PureThermal"):
        found_device = dev
        break
if not found_device:
    print("Couldn't find Lepton device")
    sys.exit(1)

lep = found_device.Open()
lep.sys.RunFFCNormalization()
lep.sys.SetGainMode(CCI.Sys.GainMode.LOW)

incoming_frames = deque(maxlen=1)
def got_a_frame(short_array, width, height):
    incoming_frames.append((height, width, short_array))

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()

tlinear = False
try:
    lep.rad.SetTLinearEnableStateChecked(True)
    tlinear = True
    print("T-Linear mode enabled: pixel values are mapped linearly to temperature")
except:
    print("T-Linear not supported on this device; using raw centi-kelvin values")

def centikelvin_to_celsius(t_ck):
    return (t_ck - 27315) / 100.0

def to_fahrenheit(t_ck):
    c = centikelvin_to_celsius(t_ck)
    return c * 9.0 / 5.0 + 32.0

print(f"Waiting for {CALIBRATION_DELAY} seconds before calibration")
plt.ion()
fig_delay, ax_delay = plt.subplots()
img_delay = None
delay_start = time.time()
while time.time() - delay_start < CALIBRATION_DELAY:
    if not incoming_frames:
        time.sleep(0.1)
        continue
    height, width, raw_buffer = incoming_frames[-1]
    arr = np.fromiter(raw_buffer, dtype=np.uint16).reshape((height, width))
    if img_delay is None:
        img_delay = ax_delay.imshow(arr, cmap=cm.plasma, vmin=arr.min(), vmax=arr.max())
        ax_delay.set_title("Calibration Delay")
        plt.show()
    else:
        img_delay.set_data(arr)
        img_delay.set_clim(vmin=arr.min(), vmax=arr.max())
        fig_delay.canvas.draw()
    plt.pause(0.05)
plt.close(fig_delay)

print("Calibrating for 5 seconds (don't have humans in view)")
plt.ion()
fig_cal, ax_cal = plt.subplots()
img_cal = None
total_sum = 0.0
total_count = 0
cal_start = time.time()
while time.time() - cal_start < CALIBRATION_DURATION:
    if not incoming_frames:
        time.sleep(0.1)
        continue
    height, width, raw_buffer = incoming_frames[-1]
    arr = np.fromiter(raw_buffer, dtype=np.uint16).reshape((height, width))
    total_sum += arr.sum()
    total_count += arr.size
    if img_cal is None:
        img_cal = ax_cal.imshow(arr, cmap=cm.plasma, vmin=arr.min(), vmax=arr.max())
        ax_cal.set_title("Calibration")
        plt.show()
    else:
        img_cal.set_data(arr)
        img_cal.set_clim(vmin=arr.min(), vmax=arr.max())
        fig_cal.canvas.draw()
    plt.pause(0.05)

calibration_mean_ck = total_sum / total_count
cal_mean_c = centikelvin_to_celsius(calibration_mean_ck)
print(f"Calibration done: mean_ck={calibration_mean_ck:.2f}")
print(f"Calibration in °C: mean={cal_mean_c:.2f}°C")

plt.close(fig_cal)

plt.ion()
fig, ax = plt.subplots()
img = None
print("Starting human detection")
last_report_time = time.time()
print("Starting frame capture. Press Ctrl-C to stop")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

try:
    while True:
        if not incoming_frames:
            time.sleep(0.1)
            continue
        height, width, raw_buffer = incoming_frames[-1]
        arr = np.fromiter(raw_buffer, dtype=np.uint16).reshape((height, width))
        normalized_arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_arr = cv2.cvtColor(normalized_arr, cv2.COLOR_GRAY2RGB)
        results = model(
            rgb_arr,
            conf=model.conf,
            classes=MODEL_CLASSES,
            verbose=False,
            show=False
        )
        for child in list(ax.patches):
            child.remove()
        human_mask = np.zeros_like(arr, dtype=bool)
        num_humans = 0
        if len(results) > 0:
            for result in results:
                boxes = result.boxes
                masks = result.masks
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    if int(cls) == 0 and conf >= MODEL_CONF:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             edgecolor='white', facecolor='none', linewidth=2)
                        ax.add_patch(rect)
                        if masks is not None and i < len(masks):
                            mask = masks[i].data[0].cpu().numpy()
                            mask = cv2.resize(mask, (width, height))
                            human_mask |= (mask > 0.5)
                        else:
                            human_mask[y1:y2, x1:x2] = True
                        num_humans += 1
        current_time = time.time()
        if current_time - last_report_time >= REPORT_TIME:
            if num_humans > 0:
                human_vals = arr[human_mask]
                filtered_human_vals = human_vals[human_vals > calibration_mean_ck]
                if len(filtered_human_vals) > 0:
                    human_mean_ck = filtered_human_vals.mean()
                    human_mean_c = centikelvin_to_celsius(human_mean_ck)
                    human_mean_f = to_fahrenheit(human_mean_ck)
                    print(f"Humans Detected: {num_humans}")
                    print(f"Filtered Human Avg: {human_mean_f:.2f} °F / {human_mean_c:.2f} °C")
                else:
                    print(f"Humans Detected: {num_humans}")
                    print("No human pixels above calibration mean; cannot compute filtered human temperature")
                non_human_vals = arr[~human_mask]
                non_human_mean_ck = non_human_vals.mean() if len(non_human_vals) > 0 else 0
                non_human_mean_c = centikelvin_to_celsius(non_human_mean_ck)
                non_human_mean_f = to_fahrenheit(non_human_mean_ck)
                print(f"Non-Human Avg: {non_human_mean_f:.2f} °F / {non_human_mean_c:.2f} °C")
            else:
                non_human_mean_ck = arr.mean()
                non_human_mean_c = centikelvin_to_celsius(non_human_mean_ck)
                non_human_mean_f = to_fahrenheit(non_human_mean_ck)
                print(f"Humans Detected: 0")
                print(f"Scene Avg (no humans): {non_human_mean_f:.2f} °F / {non_human_mean_c:.2f} °C")
            last_report_time = current_time
        if img is None:
            img = ax.imshow(arr, cmap=cm.plasma, vmin=arr.min(), vmax=arr.max())
            ax.set_title("Live Thermal")
            plt.show()
        else:
            img.set_data(arr)
            img.set_clim(vmin=arr.min(), vmax=arr.max())
            fig.canvas.draw()
        plt.pause(0.1)
except KeyboardInterrupt:
    print("\nCapture stopped by user.")
finally:
    capture.StopGraph()
    print("Graph stopped. Exiting.")
