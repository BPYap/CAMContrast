import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.metrics import ClassificationMetrics
from camcontrast.model import ClassificationModel
from camcontrast.train_utils import Callback, JobRunner
from camcontrast.xray_dataset import ChestXray14MultilabelDataset, LABELS
from camcontrast.xray_transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS

TASK_NAMES = LABELS


def _sample_indices(dataset, ratio):
    label_bins = defaultdict(list)
    for i in range(len(dataset)):
        _, labels = dataset.__getitem__(i)
        for label_name in LABELS:
            if labels[label_name] == 1:
                label_bins[label_name].append(i)
        if labels['normal-abnormal'] == 0:
            label_bins['normal'].append(i)

    sampled = []
    for label_name, indices in label_bins.items():
        sample_count = int(ratio * len(indices))
        sampled.extend(random.sample(indices, sample_count))

    return list(set(sampled))


class ModelCallback(Callback):
    def __init__(self):
        self.task_names = TASK_NAMES

        self.classification_criterion = None
        self.per_task_metrics = None

    def init_training_criteria(self):
        pass

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        ground_truths = torch.cat([batch[1][t].to(args.device).view(-1, 1) for t in self.task_names], dim=1)
        _, logits = model(inputs)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, ground_truths.float())

        return loss

    def init_metrics(self):
        self.per_task_metrics = {t: ClassificationMetrics(2) for t in self.task_names}

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        _, logits = model(inputs)
        for i, task_name in enumerate(self.task_names):
            task_logits = logits[:, i]
            probs = torch.sigmoid(task_logits).view(-1, 1)
            ground_truths = batch[1][task_name].to(args.device)
            self.per_task_metrics[task_name].update(torch.cat([1 - probs, probs], dim=1), ground_truths)

    def summarize_metrics(self):
        summary = {
            "mean-auc": np.mean([m.roc_auc_scores[1] for m in self.per_task_metrics.values()])
        }

        return summary


def main():
    parser = get_argument_parser(finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory consisting of all images."
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        help="Path to training labels (.csv file)."
    )
    parser.add_argument(
        "--eval_label_path",
        type=str,
        help="Path to evaluation labels (.csv file)."
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
    model = ClassificationModel(n_channels=1, num_classes=14, output_name="chest-disease")
    train_dataset = None
    if args.train_label_path:
        train_dataset = ChestXray14MultilabelDataset(args.image_dir, args.train_label_path)
        if args.labeled_ratio != 1.0:
            sampled_indices = _sample_indices(train_dataset, args.labeled_ratio)
            train_dataset = Subset(train_dataset, sampled_indices)
    eval_dataset = None
    if args.eval_label_path:
        eval_dataset = ChestXray14MultilabelDataset(args.image_dir, args.eval_label_path)
        if args.labeled_ratio != 1.0:
            sampled_indices = _sample_indices(eval_dataset, args.labeled_ratio)
            eval_dataset = Subset(eval_dataset, sampled_indices)
    if args.merge_dataset:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, eval_dataset])

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
