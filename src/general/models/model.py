import os
import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Base Custom Model Class
    """
    def __init__(self):
        super().__init__()

    def save(self, name, checkpoint_dir):
        model_save_path = os.path.join(checkpoint_dir, f"{name}.h5")
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, load_file):
        state_dict = torch.load(load_file)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {load_file}")
