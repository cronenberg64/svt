from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import os

root = r'C:\Users\user\Documents\GitHub\Jon\svt\data\DVS128Gesture'
os.makedirs(root, exist_ok=True)
try:
    dataset = DVS128Gesture(root, train=True, data_type='event')
    print("Dataset initialized successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
