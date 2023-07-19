import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.metrics import SegmentationMetrics
from camcontrast.model import SegmentationModel
from camcontrast.train_utils import Callback, JobRunner
from camcontrast.transforms_utils import ToMask
from camcontrast.xray_dataset import SIIMDataset
from camcontrast.xray_transforms import (TRANSFORM_TRAIN_SEG, TRANSFORM_TRAIN_SEG_MASK,
                                         TRANSFORM_EVAL_SEG, TRANSFORM_EVAL_SEG_MASK)


def _get_label_bins(dataset):
    label_bins = defaultdict(list)
    for i in range(len(dataset)):
        _, mask = dataset.__getitem__(i)
        if ToMask()(mask).sum() == 0:
            label_bins['negative'].append(i)
        else:
            label_bins['positive'].append(i)

    return label_bins


def _sample_indices(dataset, ratio):
    label_bins = _get_label_bins(dataset)

    sampled = []
    for label_name, indices in label_bins.items():
        sample_count = int(ratio * len(indices))
        sampled.extend(random.sample(indices, sample_count))

    return sampled


class ModelCallback(Callback):
    def __init__(self):
        self.segmentation_criterion = None
        self.metrics = None

    def init_training_criteria(self):
        pass

    @staticmethod
    def dice_loss(pred_probs, targets):
        smooth = 1.
        pred_probs = pred_probs.view(-1)
        targets = targets.view(-1)
        intersection = (pred_probs * targets).sum()
        total = pred_probs.sum() + targets.sum()

        return 1 - ((2. * intersection + smooth) / (total + smooth))

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        logits = model(inputs).squeeze(1)
        ground_truths = batch[1].to(args.device).float()
        loss = self.dice_loss(torch.sigmoid(logits), ground_truths)

        return loss

    def init_metrics(self):
        self.metrics = SegmentationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        logits = model(inputs)
        ground_truths = batch[1].to(args.device)
        if logits.shape[-2:] != ground_truths.shape[-2:]:
            logits = F.interpolate(
                logits, size=ground_truths.shape[-2:], mode='bilinear', align_corners=False
            )
        predictions = torch.sigmoid(logits.squeeze(1)) >= 0.5

        self.metrics.update(predictions, ground_truths)

    def summarize_metrics(self):
        summary = {"F1 score": self.metrics.f1_scores[1]}

        return summary


def main():
    parser = get_argument_parser(segmentation=True, finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--mask_dir",
        type=str,
        help="Directory with all the segmentation masks."
    )
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        help="Path to a .txt file with filenames of training images."
    )
    parser.add_argument(
        "--eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images."
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        help="Path to a .txt file with filenames of evaluation images."
    )
    parser.add_argument(
        "--labeled_ratio",
        type=float,
        default=1.0,
        help="Ratio of labeled data."
    )
    parser.add_argument(
        "--merge_dataset",
        action='store_true',
        help="Merge the original train and dev datasets into one training dataset."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback()
    model = SegmentationModel(n_channels=1, num_classes=1)
    train_dataset = None
    if args.train_image_dir and args.train_split:
        train_dataset = SIIMDataset(args.train_image_dir, args.mask_dir, args.train_split)
        if args.labeled_ratio != 1.0:
            sampled_indices = _sample_indices(train_dataset, args.labeled_ratio)
            train_dataset = Subset(train_dataset, sampled_indices)
    eval_dataset = None
    if args.eval_image_dir and args.eval_split:
        eval_dataset = SIIMDataset(args.eval_image_dir, args.mask_dir, args.eval_split)
        if args.labeled_ratio != 1.0:
            sampled_indices = _sample_indices(eval_dataset, args.labeled_ratio)
            eval_dataset = Subset(eval_dataset, sampled_indices)
    if args.merge_dataset:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, eval_dataset])

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_SEG, train_transforms_mask=TRANSFORM_TRAIN_SEG_MASK,
        eval_transforms_image=TRANSFORM_EVAL_SEG, eval_transforms_mask=TRANSFORM_EVAL_SEG_MASK
    )


if __name__ == "__main__":
    main()
