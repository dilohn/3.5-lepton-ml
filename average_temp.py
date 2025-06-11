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

found_device = None
for dev in CCI.GetDevices():
    if dev.Name.startswith("PureThermal"):
        found_device = dev
        break

if not found_device:
    print("Couldn't find Lepton device")
    sys.exit(1)

lep = found_device.Open()

# Perform a one-time flat-field correction (FFC)
lep.sys.RunFFCNormalization()

# Set gain mode to LOW
lep.sys.SetGainMode(CCI.Sys.GainMode.LOW)

# Prepare a small buffer for incoming raw frames (only keep the latest)
incoming_frames = deque(maxlen=1)

def got_a_frame(short_array, width, height):
    incoming_frames.append((height, width, short_array))

# Set up the capture pipeline
capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()

# Attempt to enable T-Linear (linear temperature output)
tlinear = False
try:
    lep.rad.SetTLinearEnableStateChecked(True)
    tlinear = True
    print("T-Linear mode enabled: pixel values are mapped linearly to temperature.")
except:
    print("T-Linear not supported on this device; using raw centi-kelvin values.")

# Conversion helpers
def centikelvin_to_celsius(t_ck):
    # Convert from centi-kelvin to degrees Celsius
    return (t_ck - 27315) / 100.0

def to_fahrenheit(t_ck):
    # Convert from centi-kelvin to degrees Fahrenheit
    c = centikelvin_to_celsius(t_ck)
    return c * 9.0 / 5.0 + 32.0

plt.ion()
fig, ax = plt.subplots()
img = None

print("Starting frame capture. Press Ctrl-C to stop.")

try:
    while True:
        # Wait until we have at least one frame
        if not incoming_frames:
            time.sleep(0.1)
            continue

        # Grab the most recent raw frame (height, width, rawBuffer)
        height, width, raw_buffer = incoming_frames[-1]
        # Convert rawBuffer (flat sequence of uint16) to a 2D NumPy array
        arr = np.fromiter(raw_buffer, dtype=np.uint16).reshape((height, width))

        # Compute max and average in centi-kelvin
        max_ck = arr.max()
        mean_ck = arr.mean()
        # Convert to Celsius and Fahrenheit
        max_c = centikelvin_to_celsius(max_ck)
        mean_c = centikelvin_to_celsius(mean_ck)
        max_f = to_fahrenheit(max_ck)
        mean_f = to_fahrenheit(mean_ck)

        # Print temperatures
        print(f"Max: {max_f:.2f} °F / {max_c:.2f} °C   "
              f"Mean: {mean_f:.2f} °F / {mean_c:.2f} °C")

        # Display/update the thermal image
        if img is None:
            img = ax.imshow(arr, cmap=cm.plasma, vmin=arr.min(), vmax=arr.max())
            plt.colorbar(img, ax=ax)
            ax.set_title("Live Thermal")
            plt.show()
        else:
            img.set_data(arr)
            img.set_clim(vmin=arr.min(), vmax=arr.max())
            fig.canvas.draw()

        # Small pause to allow the plot to update
        plt.pause(0.1)

except KeyboardInterrupt:
    print("\nCapture stopped by user.")
finally:
    capture.StopGraph()
    print("Graph stopped. Exiting.")
