import argparse
import csv
import os

import torch
from torchvision import transforms
from tqdm import tqdm

from camcontrast.fundus_dataset import OIAODIRDataset
from camcontrast.fundus_transforms import OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD
from camcontrast.layer_utils import get_heatmaps
from camcontrast.model import ClassificationModel
from camcontrast.transforms_utils import TransformsDataset

BACKGROUND_COLOR_THRESHOLD = sum([(15 / 255 - OIAODIR_DATASET_MEAN[i]) / OIAODIR_DATASET_STD[i] for i in range(3)])
MEAN = torch.tensor(OIAODIR_DATASET_MEAN).reshape(3, 1, 1)
STD = torch.tensor(OIAODIR_DATASET_STD).reshape(3, 1, 1)

TASK_NAME = 'normal-abnormal'

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_images", type=str, required=True)
    arg_parser.add_argument("--train_labels", type=str, required=True)
    arg_parser.add_argument("--dev_images", type=str, required=True)
    arg_parser.add_argument("--dev_labels", type=str, required=True)
    arg_parser.add_argument("--model_path", type=str, required=True)
    arg_parser.add_argument("--encoder_backbone", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    arg_parser.add_argument("--crop_size", type=int, default=224)
    args = arg_parser.parse_args()

    patch_folder = os.path.join(args.output_folder, "patches")
    heatmap_folder = os.path.join(args.output_folder, "heatmaps")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        os.makedirs(patch_folder)
        os.makedirs(heatmap_folder)

    crop_height = args.crop_size
    crop_width = args.crop_size

    model = ClassificationModel(n_channels=3, num_classes=2)
    model.init(args.encoder_backbone, imagenet_init=False)
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model.to(torch.device("cuda"))
    model.eval()

    dataset = TransformsDataset(
        torch.utils.data.ConcatDataset([
            OIAODIRDataset(args.train_images, args.train_labels, num_views=1),
            OIAODIRDataset(args.dev_images, args.dev_labels, num_views=1)
        ]),
        image_transform=transforms.Compose([
            transforms.CenterCrop(args.crop_size * 2),
            transforms.ToTensor(),
            transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
        ])
    )

    image_labels = dict()
    with torch.no_grad():
        for batch in tqdm(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)), "Computing CAMs..."):
            image = batch[0][0].to(torch.device("cuda"))
            image_id = batch[1]["filename"][0].split('.')[0]
            image_label = batch[1][TASK_NAME][0].item()

            image_height = image.shape[-2]
            image_width = image.shape[-1]

            # get heatmap
            feature_maps, _ = model(image)
            heatmap = get_heatmaps(feature_maps, model, upsample_size=(image_height, image_width))

            # apply background mask
            background_mask = image.sum(dim=1) > BACKGROUND_COLOR_THRESHOLD
            heatmap = heatmap * background_mask

            image = image[0]
            heatmap = heatmap[0]
            assert len(image.shape) == 3
            assert len(heatmap.shape) == 2

            for suffix, (h, w) in [
                ("tl", (0, 0)),
                ("tr", (0, image_width - crop_width)),
                ("bl", (image_height - crop_height, 0)),
                ("br", (image_height - crop_height, image_width - crop_width)),
                ("c", (int((image_height - crop_height + 1) * 0.5), int((image_width - crop_width + 1) * 0.5)))
            ]:
                image_crop = image[:, h: h + crop_height, w: w + crop_width].cpu()
                heatmap_crop = heatmap[h: h + crop_height, w: w + crop_width]

                assert (heatmap_crop.shape[0], heatmap_crop.shape[1]) == (crop_height, crop_width)
                patch_id = f"{image_id}_{suffix}"
                image_labels[patch_id] = image_label

                # unnormalize and save image crop
                image_crop = image_crop * STD + MEAN
                transforms.functional.to_pil_image(image_crop).save(
                    os.path.join(patch_folder, f"{patch_id}.jpg"),
                    quality=100, subsampling=0
                )
                # save heatmap crop
                transforms.functional.to_pil_image(heatmap_crop).save(
                    os.path.join(heatmap_folder, f"{patch_id}.jpg"),
                    quality=100, subsampling=0
                )

    with open(os.path.join(args.output_folder, "labels.csv"), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['patch_id', 'image_label'])
        for patch_id, image_label in image_labels.items():
            csv_writer.writerow([patch_id, image_label])
