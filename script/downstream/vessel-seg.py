import torch
import torch.nn as nn
import torch.nn.functional as F

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import CHASEDB1Dataset, DRIVEDataset, STAREDataset
from camcontrast.fundus_transforms import (TRANSFORM_TRAIN_SEG, TRANSFORM_TRAIN_SEG_MASK,
                                           TRANSFORM_EVAL_SEG, TRANSFORM_EVAL_SEG_MASK)
from camcontrast.metrics import SegmentationMetrics
from camcontrast.model import SegmentationModel
from camcontrast.train_utils import Callback, JobRunner


class ModelCallback(Callback):
    def __init__(self):
        self.segmentation_criterion = None
        self.metrics = None

    def init_training_criteria(self):
        pass

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device).float()
        logits = model(inputs).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ground_truths)

        return loss

    def init_metrics(self):
        self.metrics = SegmentationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device).float()
        logits = model(inputs)
        if logits.shape[-2:] != ground_truths.shape[-2:]:
            logits = F.interpolate(
                logits, size=ground_truths.shape[-2:], mode='bilinear', align_corners=False
            )
        predictions = torch.sigmoid(logits.squeeze(1)) >= 0.5

        self.metrics.update(predictions, ground_truths)

    def summarize_metrics(self):
        summary = {
            "F1 score": self.metrics.f1_scores[1],
            "Jaccard score": self.metrics.jaccard_scores[1]
        }

        return summary


def main():
    parser = get_argument_parser(segmentation=True, finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--chasedb1_train_image_dir",
        type=str,
        help="Directory consisting of training images from CHASE_DB1."
    )
    parser.add_argument(
        "--chasedb1_train_mask_dir",
        type=str,
        help="Directory consisting of training segmentation masks from CHASE_DB1."
    )
    parser.add_argument(
        "--drive_train_image_dir",
        type=str,
        help="Directory consisting of training images from DRIVE."
    )
    parser.add_argument(
        "--drive_train_mask_dir",
        type=str,
        help="Directory consisting of training segmentation masks from DRIVE."
    )
    parser.add_argument(
        "--stare_train_image_dir",
        type=str,
        help="Directory consisting of training images from STARE."
    )
    parser.add_argument(
        "--stare_train_mask_dir",
        type=str,
        help="Directory consisting of training segmentation masks from STARE."
    )
    parser.add_argument(
        "--chasedb1_eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images from CHASE_DB1."
    )
    parser.add_argument(
        "--chasedb1_eval_mask_dir",
        type=str,
        help="Directory consisting of evaluation segmentation masks from CHASE_DB1."
    )
    parser.add_argument(
        "--drive_eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images from DRIVE."
    )
    parser.add_argument(
        "--drive_eval_mask_dir",
        type=str,
        help="Directory consisting of evaluation segmentation masks from DRIVE."
    )
    parser.add_argument(
        "--stare_eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images from STARE."
    )
    parser.add_argument(
        "--stare_eval_mask_dir",
        type=str,
        help="Directory consisting of evaluation segmentation masks from STARE."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback()
    model = SegmentationModel(n_channels=3, num_classes=1)

    train_dataset = None
    if args.chasedb1_train_image_dir and args.drive_train_image_dir and args.stare_train_image_dir:
        chasedb1_train_dataset = CHASEDB1Dataset(args.chasedb1_train_image_dir, args.chasedb1_train_mask_dir)
        drive_train_dataset = DRIVEDataset(args.drive_train_image_dir, args.drive_train_mask_dir)
        stare_train_dataset = STAREDataset(args.stare_train_image_dir, args.stare_train_mask_dir)
        train_dataset = torch.utils.data.ConcatDataset([
            chasedb1_train_dataset, drive_train_dataset, stare_train_dataset
        ])
    eval_dataset = None
    if args.chasedb1_eval_image_dir and args.drive_eval_image_dir and args.stare_eval_image_dir:
        chasedb1_eval_dataset = CHASEDB1Dataset(args.chasedb1_eval_image_dir, args.chasedb1_eval_mask_dir)
        drive_eval_dataset = DRIVEDataset(args.drive_eval_image_dir, args.drive_eval_mask_dir)
        stare_eval_dataset = STAREDataset(args.stare_eval_image_dir, args.stare_eval_mask_dir)
        eval_dataset = torch.utils.data.ConcatDataset([
            chasedb1_eval_dataset, drive_eval_dataset, stare_eval_dataset
        ])

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_SEG, train_transforms_mask=TRANSFORM_TRAIN_SEG_MASK,
        eval_transforms_image=TRANSFORM_EVAL_SEG, eval_transforms_mask=TRANSFORM_EVAL_SEG_MASK
    )


if __name__ == "__main__":
    main()
