# Basic Python
import os
import random

# Data handling
import numpy as np
import pandas as pd

# Image handling
from PIL import Image

# Visualization (optional for now)
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset utilities
from torch.utils.data import Dataset, DataLoader

# Image transforms
from torchvision import transforms

print("All imports successful ✅")
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())