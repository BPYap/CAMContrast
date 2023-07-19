import torch

from SupContrast.losses import SupConLoss
from camcontrast.config import get_argument_parser, parse_and_process_arguments, is_distributed
from camcontrast.fundus_dataset import OIAODIRDataset
from camcontrast.fundus_transforms import TRANSFORM_TRAIN_CONTRASTIVE
from camcontrast.layer_utils import GatherLayer
from camcontrast.model import ContrastiveLearningModel
from camcontrast.train_utils import Callback, JobRunner


class ModelCallback(Callback):
    def __init__(self, temperature):
        self.contrastive_criterion = None
        self.temperature = temperature

    def init_training_criteria(self):
        self.contrastive_criterion = SupConLoss(temperature=self.temperature, base_temperature=self.temperature)

    def compute_training_loss(self, args, model, batch, train_info):
        inputs = torch.cat(batch[0]).to(args.device)

        vectors = model(inputs)
        batch_size, num_views = int(len(vectors) / 2), 2
        vectors = torch.cat(torch.chunk(vectors, num_views), dim=1).reshape(batch_size, num_views, -1)
        if is_distributed():
            vectors = torch.cat(GatherLayer.apply(vectors))

        return self.contrastive_criterion(vectors)

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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter to control the flatness of the softmax curve in the training objective."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback(args.temperature)
    model = ContrastiveLearningModel(n_channels=3)
    dataset_1 = OIAODIRDataset(args.train_images, args.train_labels, num_views=2)
    dataset_2 = OIAODIRDataset(args.dev_images, args.dev_labels, num_views=2)
    train_dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])

    job_runner = JobRunner(args, model, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
