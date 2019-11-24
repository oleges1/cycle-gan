from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms
import glob
import os
import numpy as np

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_img(path):
    return Image.open(path).convert('RGB')


class FolderDataset(data.Dataset):
    def __init__(self, root):
        paths = []
        for path, subdirs, files in tqdm(os.walk(root)):
            for name in files:
                if name.endswith('.jpg') or name.endswith('.png'):
                    try:
                        _ = Image.open(os.path.join(path, name))
                        paths.append(os.path.join(path, name))
                    except:
                        continue
        self.data = sorted(paths)

        self.transform_image = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize((128, 128)),
            # transforms.RandomResizedCrop(224),
#             transforms.ColorJitter(),
            # transforms.RandomAffine(10),
#             transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        photo = self.data[idx]
        photo = Image.open(photo).convert('RGB')
        return self.transform_image(photo)

class Folder2FolderDataset(data.Dataset):
    def __init__(self, folder1, folder2, phase='train'):
        self.folder1 = FolderDataset(folder1)
        self.folder2 = FolderDataset(folder2)
        self.phase = phase
        if self.phase == 'train':
            self.ind_left = np.arange(int(len(self.folder1) * 0.8))
            self.ind_right = np.arange(int(len(self.folder2) * 0.8))
        else:
            self.ind_left = np.arange(int(len(self.folder1) * 0.8), len(self.folder1))
            self.ind_right = np.arange(int(len(self.folder2) * 0.8), len(self.folder2))

    def __len__(self):
        return min(len(self.ind_left), len(self.ind_right))

    def __getitem__(self, idx):
        left = self.folder1[np.random.choice(self.ind_left)]
        right = self.folder2[np.random.choice(self.ind_right)]
        return left, right
