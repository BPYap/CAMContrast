import torch

from camcontrast.config import get_argument_parser, parse_and_process_arguments
from camcontrast.model import ClassificationModel
from camcontrast.train_utils import Callback, JobRunner
from camcontrast.xray_dataset import ChestXray14PatchDataset
from camcontrast.xray_transforms import TRANSFORM_TRAIN_CONTRASTIVE


class ModelCallback(Callback):
    def __init__(self):
        self.classification_criterion = None

    def init_training_criteria(self):
        self.classification_criterion = torch.nn.CrossEntropyLoss()

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)
        ground_truths = batch[1]['image_label'].to(args.device)

        feat_maps, logits = model(inputs)
        labeled_loss = self.classification_criterion(logits, ground_truths)

        return labeled_loss

    def init_metrics(self):
        raise NotImplementedError

    def update_metrics(self, args, model, batch):
        raise NotImplementedError

    def summarize_metrics(self):
        raise NotImplementedError


def main():
    parser = get_argument_parser(distributed=True)

    # Add custom arguments
    parser.add_argument(
        "--images",
        type=str,
        help="Directory consisting of all images."
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to training labels (.csv file)."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback()
    model = ClassificationModel(n_channels=1, num_classes=2)
    train_dataset = ChestXray14PatchDataset(args.images, args.labels, num_views=1)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
