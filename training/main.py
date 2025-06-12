import os
import sys
import subprocess
import time

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode

def main():
    import split_dataset
    split_dataset.main()
    
    import train
    train.main()
    
    import predict
    predict.main()
    
    print("Trained model saved to: runs/segment/human_segmentation/weights/best.pt")
    print("Prediction results saved to: predictions/")
    
if __name__ == "__main__":
    main()
