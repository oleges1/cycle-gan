from PIL import Image
from torch.utils import data
import glob
import os

gtFine_dir = 'data/cityscapes/gtFine'
leftImg8bit_dir = 'data/cityscapes/leftImg8bit'

def load_img(path):
    return Image.open(path).convert('RGB')

class CityScapes(data.Dataset):
    def __init__(self, phase='train'):
        segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_color.png"
        segmap_paths = glob.glob(segmap_expr)
        segmap_paths = sorted(segmap_paths)

        photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
        photo_paths = glob.glob(photo_expr)
        photo_paths = sorted(photo_paths)

        self.data = list(zip(photo_paths, segmap_paths))

        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            transforms.ColorJitter(),
            # transforms.RandomAffine(10),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        photo, segmap = self.data[idx]
        return self.transform(photo), self.transform(segmap)
