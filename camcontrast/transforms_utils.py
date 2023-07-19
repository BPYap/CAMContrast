# Partly adapted from https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py

import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])


class TransformsDataset(torch.utils.data.Dataset):
    """ Dataset wrapper for image transformation. Useful for wrapping ConcatDataset, Subset or other Dataset object.
    """

    def __init__(self, dataset, image_transform, target_transform=None):
        self.dataset = dataset
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __bool__(self):
        return self.dataset is not None

    def __getitem__(self, index):
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            image, target = sample
        else:
            image = sample
            target = -1  # used to denote unlabeled dataset

        if self.target_transform:
            if isinstance(image, list):
                raise TypeError("Image list is not supported when `target_transform` is not None.")

            # same seed to ensure random transformations are applied consistently on both image and target
            seed = random.randint(0, 2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)

            if isinstance(target, dict):  # multi-task targets
                for name, target_ in target.items():
                    random.seed(seed)
                    torch.manual_seed(seed)
                    target[name] = self.target_transform(target_)
            else:
                random.seed(seed)
                torch.manual_seed(seed)
                target = self.target_transform(target)
        else:
            image = [self.image_transform(img) for img in image] if isinstance(image, list) \
                else self.image_transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


class ToMask:
    def __call__(self, image):
        if image.mode == 'P':
            np_image = np.array(image)
            if np_image.ndim == 2:
                np_image = np_image[:, :, None]

            tensor = torch.from_numpy(np_image.transpose((2, 0, 1)))
        else:
            tensor = transforms.functional.to_tensor(image)

        return tensor.squeeze().long()
