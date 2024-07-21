import os
from pathlib import Path
import numpy as np
import torchvision
from PIL import Image

class ImageFolder(torchvision.datasets.VisionDataset):

    def __init__(self, root, train_list_file, val_list_file, split='train', **kwargs):

        root = Path(root)
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                samples = [self.root.joinpath(line.strip()) for line in f.readlines()]

        self.samples = samples

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        # Convert to numpy array
        sample_array = np.array(sample)
        # Add noise
        noise = np.random.normal(0, 50, sample_array.shape)
        img_with_noise = np.clip(sample_array + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        img_with_noise = Image.fromarray(img_with_noise)

        if self.transforms is not None and with_transform:
            sample, _ = self.transforms(sample, None)
            img_with_noise, _ = self.transforms(img_with_noise, None)
        return sample, img_with_noise


class FFHQ(ImageFolder):
    train_list_file = Path(__file__).parent.joinpath('assets/ffhqtrain.txt')
    val_list_file = Path(__file__).parent.joinpath('assets/ffhqvalidation.txt')

    def __init__(self, root, split='train', **kwargs):
        super().__init__(root, FFHQ.train_list_file, FFHQ.val_list_file, split, **kwargs)
