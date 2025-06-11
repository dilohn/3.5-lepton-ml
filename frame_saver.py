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

output_dir = "thermal_frames"
os.makedirs(output_dir, exist_ok=True)

found_device = None
for dev in CCI.GetDevices():
    if dev.Name.startswith("PureThermal"):
        found_device = dev
        break

if not found_device:
    print("Couldn't find Lepton device. Exiting.")
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
    print("T-Linear mode enabled: pixel values are mapped linearly to temperature.")
except:
    print("T-Linear not supported on this device; using raw centi-kelvin values.")

def centikelvin_to_celsius(t_ck):
    return (t_ck - 27315) / 100.0

def to_fahrenheit(t_ck):
    c = centikelvin_to_celsius(t_ck)
    return c * 9.0 / 5.0 + 32.0

plt.ion()
fig, ax = plt.subplots()
img = None

frame_counter = 0

print("Starting frame capture. Press Ctrl-C to stop.")

try:
    while True:
        if not incoming_frames:
            time.sleep(0.1)
            continue

        height, width, raw_buffer = incoming_frames[-1]
        arr = np.fromiter(raw_buffer, dtype=np.uint16).reshape((height, width))

        max_ck = arr.max()
        mean_ck = arr.mean()
        max_c = centikelvin_to_celsius(max_ck)
        mean_c = centikelvin_to_celsius(mean_ck)
        max_f = to_fahrenheit(max_ck)
        mean_f = to_fahrenheit(mean_ck)

        print(f"Max: {max_f:.2f} °F / {max_c:.2f} °C   "
              f"Mean: {mean_f:.2f} °F / {mean_c:.2f} °C")

        frame_counter += 1
        filename = os.path.join(output_dir, f"image_{frame_counter}.png")
        plt.imsave(filename, arr, cmap='plasma')

        if img is None:
            img = ax.imshow(arr, cmap=cm.plasma, vmin=arr.min(), vmax=arr.max())
            plt.colorbar(img, ax=ax)
            ax.set_title("Live Thermal Image")
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
