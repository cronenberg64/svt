from spikingjelly.activation_based import neuron
import torch

try:
    print("Creating LIFNode...")
    n = neuron.LIFNode()
    print("LIFNode created.")
    print("Printing LIFNode:")
    print(n)
    print("Success!")
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
