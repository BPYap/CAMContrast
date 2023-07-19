import torch
import torch.nn as nn
import torch.nn.functional as F

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import IDRiDSegmentationDataset
from camcontrast.fundus_transforms import (TRANSFORM_TRAIN_SEG, TRANSFORM_TRAIN_SEG_MASK,
                                           TRANSFORM_EVAL_SEG, TRANSFORM_EVAL_SEG_MASK)
from camcontrast.metrics import AUCPRMetrics
from camcontrast.model import SegmentationModel
from camcontrast.train_utils import Callback, JobRunner

LABEL_WEIGHTS = {
    "MA": 6.84,
    "HE": 4.60,
    "EX": 4.82,
    "SE": 5.54
}


class ModelCallback(Callback):
    def __init__(self, task_id):
        self.task_id = task_id
        self.pos_weight = LABEL_WEIGHTS[task_id]

        self.metrics = None

    def init_training_criteria(self):
        pass

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device).float()
        logits = model(inputs).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, ground_truths, pos_weight=torch.tensor([self.pos_weight]).to(args.device)
        )

        return loss

    def init_metrics(self):
        self.metrics = AUCPRMetrics()

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        ground_truths = batch[1].to(args.device)
        logits = model(inputs)
        if logits.shape[-2:] != ground_truths.shape[-2:]:
            logits = F.interpolate(
                logits, size=ground_truths.shape[-2:], mode='bilinear', align_corners=False
            )
        predictions = torch.sigmoid(logits.squeeze(1))
        self.metrics.update(predictions, ground_truths)

    def summarize_metrics(self):
        summary = {
            "AUC-PR score": self.metrics.aupr_score
        }

        return summary


def main():
    parser = get_argument_parser(segmentation=True, finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_mask_dir",
        type=str,
        help="Directory consisting of training segmentation masks."
    )
    parser.add_argument(
        "--eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images."
    )
    parser.add_argument(
        "--eval_mask_dir",
        type=str,
        help="Directory consisting of evaluation segmentation masks."
    )
    parser.add_argument(
        "--task_id",
        choices=["MA", "HE", "EX", "SE"],
        help="Type of lesion segmentation task."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback(args.task_id)
    model = SegmentationModel(n_channels=3, num_classes=1)
    train_dataset = None
    if args.train_image_dir and args.train_mask_dir:
        train_dataset = IDRiDSegmentationDataset(args.train_image_dir, args.train_mask_dir)
    eval_dataset = None
    if args.eval_image_dir and args.eval_mask_dir:
        eval_dataset = IDRiDSegmentationDataset(args.eval_image_dir, args.eval_mask_dir)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_SEG, train_transforms_mask=TRANSFORM_TRAIN_SEG_MASK,
        eval_transforms_image=TRANSFORM_EVAL_SEG, eval_transforms_mask=TRANSFORM_EVAL_SEG_MASK
    )


if __name__ == "__main__":
    main()
