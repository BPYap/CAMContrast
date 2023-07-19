import torch
import torch.nn as nn

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import OIAODIRDataset
from camcontrast.fundus_transforms import TRANSFORM_TRAIN_CONTRASTIVE
from camcontrast.metrics import ClassificationMetrics
from camcontrast.model import ClassificationModel
from camcontrast.train_utils import Callback, JobRunner


class ModelCallback(Callback):
    def __init__(self):
        self.classification_criterion = None
        self.metrics = None

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)
        ground_truths = batch[1]['normal-abnormal'].to(args.device)

        feat_maps, logits = model(inputs)
        labeled_loss = self.classification_criterion(logits, ground_truths)

        return labeled_loss

    def init_metrics(self):
        self.metrics = ClassificationMetrics(2)

    def update_metrics(self, args, model, batch):
        inputs = torch.cat(batch[0]).to(args.device)
        ground_truths = batch[1]['normal-abnormal'].to(args.device)

        _, logits = model(inputs)
        probs = torch.softmax(logits, 1)

        self.metrics.update(probs, ground_truths)

    def summarize_metrics(self):
        summary = {}
        metrics = self.metrics

        summary.update({
            "accuracies": metrics.accuracies,
            "precisions": metrics.precisions,
            "recalls": metrics.recalls,
            "specificities": metrics.specificities,
            "F1 scores": metrics.f1_scores,
            "overall accuracy": metrics.overall_accuracy,
            "roc-auc": metrics.roc_auc_scores[1],
            "precision": metrics.precisions[1],
            "recall": metrics.recalls[1],
            "F1 score": metrics.f1_scores[1],
        })

        return summary


def main():
    parser = get_argument_parser(distributed=True)

    # Add custom arguments
    parser.add_argument(
        "--train_images",
        type=str,
        help="Directory consisting of training images for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="Path to training labels (.csv file) for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--dev_images",
        type=str,
        help="Directory consisting of development images for the OIA-ODIR dataset."
    )
    parser.add_argument(
        "--dev_labels",
        type=str,
        help="Path to development labels (.csv file) for the OIA-ODIR dataset."
    )
    args = parse_and_process_arguments(parser)

    model = ClassificationModel(n_channels=3, num_classes=2)

    dataset_1 = OIAODIRDataset(args.train_images, args.train_labels, num_views=1)
    dataset_2 = OIAODIRDataset(args.dev_images, args.dev_labels, num_views=1)
    train_dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])

    callback = ModelCallback()
    job_runner = JobRunner(args, model, callback)
    job_runner.run(train_dataset, None, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
