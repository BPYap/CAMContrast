import torch
import torch.nn as nn

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import IDRiDGradingDataset
from camcontrast.fundus_transforms import TRANSFORM_TRAIN_CLS, TRANSFORM_EVAL_CLS
from camcontrast.metrics import ClassificationMetrics
from camcontrast.model import JointClassificationModel
from camcontrast.train_utils import Callback, JobRunner

DR_TASK_NAME = 'retinopathy-grade'
ME_TASK_NAME = 'macular-edema-risk'
NUM_CLASS_PER_TASK = {DR_TASK_NAME: 5, ME_TASK_NAME: 3}
DR_LABEL_WEIGHT = [413 / 134, 413 / 20, 413 / 136, 413 / 74, 413 / 49]
ME_LABEL_WEIGHT = [413 / 177, 413 / 41, 413 / 195]


class ModelCallback(Callback):
    def __init__(self):
        self.task_names = [DR_TASK_NAME, ME_TASK_NAME]
        self.loss_weights = [DR_LABEL_WEIGHT, ME_LABEL_WEIGHT]

        self.classification_criterion = None
        self.per_task_metrics = None

    def init_training_criteria(self):
        pass

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = batch[0].to(args.device)
        _, logits = model(inputs)
        loss = 0.0
        for task_name, weights in zip(self.task_names, self.loss_weights):
            ground_truths = batch[1][task_name].to(args.device)
            weights = torch.tensor(weights).to(args.device)
            loss += nn.functional.cross_entropy(logits[task_name], ground_truths, weight=weights)

        return loss

    def init_metrics(self):
        self.per_task_metrics = {t: ClassificationMetrics(NUM_CLASS_PER_TASK[t]) for t in self.task_names}

    def update_metrics(self, args, model, batch):
        inputs = batch[0].to(args.device)
        _, logits = model(inputs)
        for task_name in self.task_names:
            probs = torch.softmax(logits[task_name], 1)
            ground_truths = batch[1][task_name].to(args.device)

            self.per_task_metrics[task_name].update(probs, ground_truths)

    def summarize_metrics(self):
        summary = {}
        dr_metrics = self.per_task_metrics[DR_TASK_NAME]
        me_metrics = self.per_task_metrics[ME_TASK_NAME]
        dr_bitmap = torch.stack(dr_metrics.predicted_probs).argmax(dim=0) == dr_metrics.ground_truths
        me_bitmap = torch.stack(me_metrics.predicted_probs).argmax(dim=0) == me_metrics.ground_truths
        joint_bitmap = dr_bitmap & me_bitmap
        num_labels = len(dr_metrics.ground_truths)
        summary["DR accuracy"] = dr_bitmap.sum().item() / num_labels
        summary["ME accuracy"] = me_bitmap.sum().item() / num_labels
        summary["joint accuracy"] = joint_bitmap.sum().item() / num_labels

        return summary


def main():
    parser = get_argument_parser(finetune=True)

    # Add custom arguments
    parser.add_argument(
        "--train_image_dir",
        type=str,
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        help="Path to training labels (.csv file)."
    )
    parser.add_argument(
        "--eval_image_dir",
        type=str,
        help="Directory consisting of evaluation images."
    )
    parser.add_argument(
        "--eval_label_path",
        type=str,
        help="Path to evaluation labels (.csv file)."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback()
    model = JointClassificationModel(n_channels=3, num_class_per_task=NUM_CLASS_PER_TASK)
    train_dataset = None
    if args.train_image_dir and args.train_label_path:
        train_dataset = IDRiDGradingDataset(args.train_image_dir, args.train_label_path)
    eval_dataset = None
    if args.eval_image_dir and args.eval_label_path:
        eval_dataset = IDRiDGradingDataset(args.eval_image_dir, args.eval_label_path)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
