import os

import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision.transforms import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ========== upstream datasets ========== #
class OIAODIRDataset(torch.utils.data.Dataset):
    """ Data source: https://github.com/nkicsl/OIA-ODIR
    """

    def __init__(self, root_dir, csv_file, num_views=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the .csv file with annotations.
            num_views (int): Number of views to be generated from each image.
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        filename = row['image']
        image = Image.open(os.path.join(self.root_dir, filename))
        label = row['normal-abnormal']

        images = [image] * self.num_views
        labels = {'filename': filename, 'normal-abnormal': label}

        return images, labels


class OIAODIRPatchDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, csv_file, num_views):
        """
        Args:
            image_dir (string): Directory with all the images.
            csv_file (string): Path to the .csv file with annotations.
            num_views (int): Number of views to be generated from each image.
        """
        self.image_dir = image_dir
        self.num_views = num_views
        self.data = pd.read_csv(csv_file, index_col='patch_id')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        patch_id = row.name

        patch = Image.open(os.path.join(self.image_dir, f"{patch_id}.jpg"))
        labels = {'patch_id': patch_id, 'image_label': row["image_label"]}

        return [patch] * self.num_views, labels


class OIAODIRHeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, heatmap_dir, csv_file):
        """
        Args:
            image_dir (string): Directory with all the images.
            heatmap_dir (string): Directory with all the heatmaps.
            csv_file (string): Path to the .csv file with annotations.
        """
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.data = pd.read_csv(csv_file, index_col='patch_id')
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((14, 14)),
            transforms.ToTensor()
        ])

    def get_patch_and_heatmap(self, patch_id):
        filename = f"{patch_id}.jpg"
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


# ========== downstream datasets ========== #
class IDRiDGradingDataset(torch.utils.data.Dataset):
    """ Data source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    """
    TASK_LABELS = {
        'retinopathy-grade': ('0 (normal)', '1', '2', '3', '4'),
        'macular-edema-risk': ('0 (no risk)', '1', '2')
    }

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

        image_id = row['Image name']
        dr_grade = row['Retinopathy grade']
        edema_risk = row['Risk of macular edema ']

        image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpg"))
        labels = {
            'retinopathy-grade': int(dr_grade),
            'macular-edema-risk': int(edema_risk)
        }

        return image, labels


class REFUGEGradingDataset(torch.utils.data.Dataset):
    """ Data source: https://refuge.grand-challenge.org/
    """
    TASK_LABELS = {
        'glaucoma-grade': ('0 (normal)', '1')
    }

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

        image_id = row['image_id']
        label = row['glaucoma_label']

        image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpg"))

        return image, label


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the segmentation masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_filename, segmentation_filename = self.data[index]

        image = Image.open(os.path.join(self.image_dir, image_filename))
        segmentation_mask = Image.open(os.path.join(self.mask_dir, segmentation_filename))

        return image, segmentation_mask


class DRIVEDataset(SegmentationDataset):
    """ Data source: https://drive.grand-challenge.org/DRIVE/
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.tif).
            mask_dir (string): Directory with all the segmentation masks (.gif).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            image_id, extension = filename.split('.')
            if extension == 'tif':
                self.data.append((filename, f"{image_id.split('_')[0]}_manual1.gif"))


class STAREDataset(SegmentationDataset):
    """ Data source: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.ppm).
            mask_dir (string): Directory with all the segmentation masks (.ah.ppm).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            if len(filename.split('.')) == 2:
                image_id = filename.split('.')[0]
                self.data.append((f"{image_id}.ppm", f"{image_id}.ah.ppm"))


class CHASEDB1Dataset(SegmentationDataset):
    """ Data source: https://blogs.kingston.ac.uk/retinal/chasedb1/
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.jpg).
            mask_dir (string): Directory with all the segmentation masks (.png).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            image_id, extension = filename.split('.')
            if extension == 'jpg':
                self.data.append((filename, f"{image_id}_1stHO.png"))


class IDRiDSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        """ Data source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.filenames = os.listdir(mask_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        mask_filename = self.filenames[index]
        image_filename = "_".join(mask_filename.split("_")[:2]) + ".jpg"

        image = Image.open(os.path.join(self.image_dir, image_filename))
        segmentation_mask = Image.open(os.path.join(self.mask_dir, mask_filename))
        if segmentation_mask.mode == 'RGBA':
            segmentation_mask = segmentation_mask.split()[0]  # extract mask from the red channel
        segmentation_mask = segmentation_mask.point(lambda p: int(p > 0), mode='1')

        return image, segmentation_mask


class REFUGESegmentationDataset(torch.utils.data.Dataset):
    """ Data source: https://refuge.grand-challenge.org/
    """
    TASK_LABELS = {
        'optic disc': ('absent', 'present'),
        'optic cup': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.jpg).
            mask_dir (string): Directory with all the segmentation masks (.bmp).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.data = []
        for filename in os.listdir(image_dir):
            image_id = filename.split('.')[0]
            self.data.append((filename, f"{image_id}.bmp"))

        self.optic_disc_masks = dict()
        self.optic_cup_masks = dict()
        for _, filename in self.data:
            segmentation_mask = Image.open(os.path.join(self.mask_dir, filename))
            self.optic_disc_masks[filename] = segmentation_mask.point(lambda p: 255 if p != 255 else 0)
            self.optic_cup_masks[filename] = segmentation_mask.point(lambda p: 255 if p == 0 else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_filename, segmentation_filename = self.data[index]

        image = Image.open(os.path.join(self.image_dir, image_filename))
        optic_disc_mask = self.optic_disc_masks[segmentation_filename]
        optic_cup_mask = self.optic_cup_masks[segmentation_filename]

        return image, {"optic disc": optic_disc_mask, "optic cup": optic_cup_mask}
