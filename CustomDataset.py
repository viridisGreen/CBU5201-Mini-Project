import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
from PIL import Image

class Genki4kDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, 'file' + "{:04d}".format(index+1) + '.jpg')

        image = io.imread(image_path)
        label_sml = int(self.labels.iloc[index, 0].split()[0])
        label_pose = tuple(float(x) for x in self.labels.iloc[index, 0].split()[1:4])
        label_pose = torch.tensor(label_pose)
        # print(label_pose.shape)

        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            image = self.transform(image)

        return (image, label_sml, label_pose)

class Genki4kDataset_Labeless(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_files[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image









