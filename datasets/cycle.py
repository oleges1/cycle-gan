from os.path import join
from PIL import Image
import os
from torch.utils import data

class CycleDataset(data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, phase):
        """Initialize this dataset class.
        Parameters:
            dataroot - root of data
            phase - train / val
        """
        self.dir_AB = join(dataroot, phase)  # get the image directory
        self.data = sorted([join(self.dir_AB, item) for item in os.listdir(self.dir_AB)])  # get image paths

        self.transform_image = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize(128),
            # transforms.RandomResizedCrop(224),
#             transforms.ColorJitter(),
            # transforms.RandomAffine(10),
#             transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
        """
        # read a image given a random integer index
        AB_path = self.data[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        return self.transform_image(A), self.transform_image(B)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
