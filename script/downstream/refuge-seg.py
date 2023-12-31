import torch
import torch.nn as nn
import torch.nn.functional as F

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import REFUGESegmentationDataset
from camcontrast.fundus_transforms import (TRANSFORM_TRAIN_SEG, TRANSFORM_TRAIN_SEG_MASK,
                                           TRANSFORM_EVAL_SEG, TRANSFORM_EVAL_SEG_MASK)
from camcontrast.metrics import SegmentationMetrics
from camcontrast.model import JointSegmentationModel
from camcontrast.train_utils import Callback, JobRunner

TASK_NAMES = ["optic disc", "optic cup"]


class ModelCallback(Callback):
    def __init__(self):
        self.task_names = TASK_NAMES

        self.segmentation_criterion = None
        self.per_task_metrics = None

    def init_training_criteria(self):
        pass

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        logits = model(inputs)

        loss = 0.0
        for task_name in self.task_names:
            task_logits = logits[task_name].squeeze(1)
            ground_truths = batch[1][task_name].to(args.device).float()
            loss += nn.functional.binary_cross_entropy_with_logits(task_logits, ground_truths)

        return loss

    def init_metrics(self):
        self.per_task_metrics = {t: SegmentationMetrics(2) for t in self.task_names}

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        logits = model(inputs)

        for task_name in self.task_names:
            ground_truths = batch[1][task_name].to(args.device)
            task_logits = logits[task_name]
            if task_logits.shape[-2:] != ground_truths.shape[-2:]:
                task_logits = F.interpolate(
                    task_logits, size=ground_truths.shape[-2:], mode='bilinear', align_corners=False
                )
            predictions = torch.sigmoid(task_logits.squeeze(1)) >= 0.5

            self.per_task_metrics[task_name].update(predictions, ground_truths)

    def summarize_metrics(self):
        summary = {}
        od_metrics = self.per_task_metrics["optic disc"]
        oc_metrics = self.per_task_metrics["optic cup"]
        summary["[optic disc] F1 score"] = od_metrics.f1_scores[1]
        summary["[optic cup] F1 score"] = oc_metrics.f1_scores[1]
        summary["avg. F1 score"] = 0.5 * (od_metrics.f1_scores[1] + oc_metrics.f1_scores[1])

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
        "--merge_dataset",
        action='store_true',
        help="Merge the original train and dev datasets into one training dataset."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback()
    model = JointSegmentationModel(n_channels=3, num_class_per_task={task_name: 1 for task_name in TASK_NAMES})
    train_dataset = None
    if args.train_image_dir and args.train_mask_dir:
        train_dataset = REFUGESegmentationDataset(args.train_image_dir, args.train_mask_dir)
    eval_dataset = None
    if args.eval_image_dir and args.eval_mask_dir:
        eval_dataset = REFUGESegmentationDataset(args.eval_image_dir, args.eval_mask_dir)
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
