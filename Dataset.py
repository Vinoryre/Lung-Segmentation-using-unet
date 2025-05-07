import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
    This file defines two dataset classes,
    one lungSefDataset is used to load training data, 
    which loads regularized CT image data and binarized lung segmentation mask respectively, 
    and the other LungTestDataset is used to load test data, 
    which only loads regularized CT image data.
'''

# Dataset class for lung segmentation training
class lungSefDataset(Dataset):
    # Initialize the dataset with CT and lung mask directories and optional transforms
    def __init__(self, ct_root, lung_root, transform = None):
        self.ct_root = ct_root # Root directory for CT scans
        self.lung_root = lung_root # Root directory for lung segmentation masks
        self.transform = transform # Optional transform to apply to the data

        # List to store the paths to CT and lung mask pairs
        self.data_list = []

        # Iterate through patients' directories to gather file paths for CT scans and lung masks
        for pid in os.listdir(ct_root):
            ct_pid_dir = os.path.join(ct_root, pid)
            lung_pid_dir = os.path.join(lung_root, pid)
            if not os.path.isdir(ct_pid_dir):
                continue
            for fname in os.listdir(ct_pid_dir):
                ct_path = os.path.join(ct_pid_dir, fname) # Path to CT file
                lung_path = os.path.join(lung_pid_dir, fname) # Path to lung mask file
                if os.path.exists(lung_path):
                    self.data_list.append((ct_path, lung_path)) # Append the pair to the data list

    # Return the total number of items in the dataset
    def __len__(self):
        return len(self.data_list)

    # Fetch a single item from the dataset ( CT scan and corresponding lung mask)
    def __getitem__(self, idx):
        ct_path, lung_path = self.data_list[idx] # Get the paths for CT and lung mask
        ct = np.load(ct_path).astype(np.float32) # Load CT scan as numpy array
        mask = np.load(lung_path).astype(np.float32) # Load lung mask as numpy array

        # Normalize CT scan to the range [0, 1]
        ct_min = np.min(ct)
        ct_max = np.max(ct)
        ct_normalized = (ct - ct_min) / (ct_max - ct_min)

        # Convert the mask to binary (values greater than 0.5 are considered as lung region)
        mask = (mask > 0.5).astype(np.float32)

        # Convert numpy arrays to PyTorch tensors, adding a channel dimension
        ct_normalized = torch.from_numpy(ct_normalized).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Apply optional transformation (if any)
        if self.transform:
            ct_normalized, mask = self.transform(ct, mask)

        # Return CT scan, lung mask, and metadata (min and max values of the CT scan)
        meta_data = {'ct_min': ct_min, 'ct_max': ct_max}
        return ct_normalized, mask, meta_data

# Dataset class for testing (CT scans only)
class LungTestDataset(Dataset):
    # Initialize the test dataset with CT directory and optional transforms
    def __init__(self, ct_root, transform=None):
        self.ct_root = ct_root # Root directory for CT scans
        self.transform = transform # Optional transform to apply to the data

        # List to store the paths to CT scans
        self.data_list = []

        # Iterate through patients' directories to gather CT scan file paths
        for pid in os.listdir(ct_root):
            ct_pid_dir = os.path.join(ct_root, pid)
            if not os.path.isdir(ct_pid_dir):
                continue
            for fname in os.listdir(ct_pid_dir):
                ct_path = os.path.join(ct_pid_dir, fname) # Path to CT file
                self.data_list.append(ct_path) # Append the CT file path to the list

    # Return the total number of items in the dataset
    def __len__(self):
        return len(self.data_list)

    # Fetch a single CT scan from the dataset
    def __getitem__(self, idx):
        ct_path = self.data_list[idx] # Get the path for CT scan
        ct = np.load(ct_path).astype(np.float32) # Load CT scan as numpy array

        # Normalize CT scan to the range [0, 1]
        ct_min = np.min(ct)
        ct_max = np.max(ct)
        ct_normalized = (ct - ct_min) / (ct_max - ct_min)

        # Convert the CT scan to a PyTorch tensor, adding a channel dimension
        ct_normalized = torch.from_numpy(ct_normalized).unsqueeze(0)

        # Apply optional transformation (if any)
        if self.transform:
            ct_normalized = self.transform(ct_normalized)

        # Return CT scan and metadata (min and max values of the CT scans)
        meta_data = {'ct_min': ct_min, 'ct_max': ct_max}
        return ct_normalized, meta_data

