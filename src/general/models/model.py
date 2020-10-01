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
        torch.save(self.state_dict(), save_path)
        print("Model saved to {}".format(model_save_path))
