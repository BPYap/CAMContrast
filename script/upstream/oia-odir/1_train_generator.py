import torch
import torch.nn as nn
import torchvision.transforms as transforms

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.fundus_dataset import OIAODIRDataset
from camcontrast.fundus_transforms import OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD
from camcontrast.layer_utils import get_cams
from camcontrast.metrics import ClassificationMetrics
from camcontrast.model import ClassificationModel
from camcontrast.train_utils import Callback, JobRunner

TRANSFORM_TRAIN_CLS = transforms.Compose([
    transforms.Resize(448),
    transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_EVAL_CLS = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])


class ModelCallback(Callback):
    def __init__(self, attention_map_loss_weight):
        self.classification_criterion = None
        self.attention_map_loss_weight = attention_map_loss_weight
        self.metrics = None

    def init_training_criteria(self):
        self.classification_criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _normalize(x):
        batch_size = len(x)
        max_ = x.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
        min_ = x.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)

        return (x - min_) / (max_ - min_)

    @staticmethod
    def compute_attention_loss(model, feature_maps, logits, ground_truths):
        batch_size = len(feature_maps)
        normal_cams = get_cams(
            feature_maps=feature_maps,
            classification_layer=model.out["classification"][-1],
            targets=[0] * batch_size,
            upsample_size=None
        )
        normal_cams = ModelCallback._normalize(normal_cams)
        abnormal_cams = get_cams(
            feature_maps=feature_maps,
            classification_layer=model.out["classification"][-1],
            targets=[1] * batch_size,
            upsample_size=None
        )
        abnormal_cams = ModelCallback._normalize(abnormal_cams)

        correct_normal = (logits.argmax(dim=1) == 0) * (ground_truths == 0)
        loss = correct_normal * (1 - normal_cams + abnormal_cams).view(batch_size, -1).mean(dim=1)

        return loss.mean()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)
        ground_truths = batch[1]['normal-abnormal'].to(args.device)

        feat_maps, logits = model(inputs)
        cls_loss = self.classification_criterion(logits, ground_truths)
        attn_map_loss = self.compute_attention_loss(model, feat_maps, logits, ground_truths)

        return cls_loss + self.attention_map_loss_weight * attn_map_loss

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
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="Path to training labels (.csv file)."
    )
    parser.add_argument(
        "--dev_images",
        type=str,
        help="Directory consisting of development images."
    )
    parser.add_argument(
        "--dev_labels",
        type=str,
        help="Path to development labels (.csv file)."
    )
    parser.add_argument(
        "--attention_map_loss_weight",
        type=float,
        default=0.01,
        help="Loss weightage given to the attention map loss."
    )
    args = parse_and_process_arguments(parser)

    model = ClassificationModel(n_channels=3, num_classes=2)

    train_dataset = OIAODIRDataset(args.train_images, args.train_labels, num_views=1)
    eval_dataset = OIAODIRDataset(args.dev_images, args.dev_labels, num_views=1)

    callback = ModelCallback(attention_map_loss_weight=args.attention_map_loss_weight)
    job_runner = JobRunner(args, model, callback)
    job_runner.run(
        train_dataset, eval_dataset,
        train_transforms_image=TRANSFORM_TRAIN_CLS,
        eval_transforms_image=TRANSFORM_EVAL_CLS
    )


if __name__ == "__main__":
    main()
