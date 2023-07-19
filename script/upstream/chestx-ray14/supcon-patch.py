import torch

from SupContrast.losses import SupConLoss
from camcontrast.config import get_argument_parser, parse_and_process_arguments, is_distributed
from camcontrast.layer_utils import GatherLayer
from camcontrast.model import ContrastiveLearningModel
from camcontrast.train_utils import Callback, JobRunner
from camcontrast.xray_dataset import ChestXray14PatchDataset
from camcontrast.xray_transforms import TRANSFORM_TRAIN_CONTRASTIVE


class ModelCallback(Callback):
    def __init__(self, temperature):
        self.contrastive_criterion = None
        self.temperature = temperature

    def init_training_criteria(self):
        self.contrastive_criterion = SupConLoss(temperature=self.temperature, base_temperature=self.temperature)

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)
        labels = batch[1]['image_label'].to(args.device)

        vectors = model(inputs)
        batch_size, num_views = int(len(vectors) / 2), 2
        vectors = torch.cat(torch.chunk(vectors, num_views), dim=1).reshape(batch_size, num_views, -1)
        if is_distributed():
            vectors = torch.cat(GatherLayer.apply(vectors))
            labels = torch.cat(GatherLayer.apply(labels))

        return self.contrastive_criterion(vectors, labels)

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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter to control the flatness of the softmax curve in the training objective."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback(args.temperature)
    model = ContrastiveLearningModel(n_channels=1)
    train_dataset = ChestXray14PatchDataset(args.images, args.labels, num_views=2)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
