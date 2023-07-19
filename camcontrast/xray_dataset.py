import os

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import transforms

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
          "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def _process_image(pil_image):
    if pil_image.mode == "RGBA":
        return pil_image.convert('L')
    else:
        return pil_image


# upstream datasets
class ChestXray14PatchDataset(torch.utils.data.Dataset):
    """ Data source: https://nihcc.app.box.com/v/ChestXray-NIHCC
    """

    def __init__(self, root_dir, csv_file, num_views=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            num_views (int): Number of views to be generated from each image.
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = _process_image(Image.open(os.path.join(self.root_dir, f"{row['patch_id']}.png")))
        label = row['image_label']

        images = [image] * self.num_views
        labels = {'patch_id': row['patch_id'], 'image_label': label}

        return images, labels


class ChestXray14HeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, heatmap_dir, csv_file):
        """
        Args:
            image_dir (string): Directory with all the images.
            heatmap_dir (string): Directory with all the heatmaps.
            csv_file (string): Path to the csv file with annotations.
        """
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.data = pd.read_csv(csv_file, index_col='patch_id')
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((14, 14)),
            transforms.ToTensor()
        ])

    def get_patch_and_heatmap(self, patch_id):
        filename = f"{patch_id}.png"
        patch = Image.open(os.path.join(self.image_dir, filename))
        heatmap = Image.open(os.path.join(self.heatmap_dir, filename))

        return patch, heatmap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        patch_id = row.name
        image_label = row["image_label"]

        patch, heatmap = self.get_patch_and_heatmap(patch_id)
        heatmap = self.heatmap_transform(heatmap) if image_label == 1 else torch.zeros(1, 14, 14)

        return patch, heatmap


# downstream datasets
class ChestXray14MultilabelDataset(torch.utils.data.Dataset):
    """ Data source: https://nihcc.app.box.com/v/ChestXray-NIHCC
    """

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.root_dir, row["filename"])
        image = _process_image(Image.open(image_path))
        labels = {c: row[c] for c in LABELS}
        labels["filename"] = row["filename"]
        labels["normal-abnormal"] = row["normal-abnormal"]

        return image, labels


class SIIMDataset(torch.utils.data.Dataset):
    """ Data sources:
            Stage 1 - https://www.kaggle.com/datasets/abhishek/siim-png-images?resource=download
            Stage 2 - https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data
    """

    def __init__(self, image_dir, mask_dir, split_txt):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the segmentation masks.
            split_txt (string): Path to a .txt file with image filenames.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        with open(split_txt, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index].strip()
        image = Image.open(os.path.join(self.image_dir, filename))
        segmentation_mask = Image.open(os.path.join(self.mask_dir, filename))

        return image, segmentation_mask
