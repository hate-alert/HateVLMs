import torch


class GPUInitializer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def forcibly_use_cpu(self):
        self.device = "cpu"

    def get_device(self):
        return self.device
