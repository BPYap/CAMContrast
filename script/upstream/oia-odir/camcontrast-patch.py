import torch

from SupContrast.losses import SupConLoss
from camcontrast.config import get_argument_parser, parse_and_process_arguments, is_distributed
from camcontrast.fundus_dataset import OIAODIRHeatmapDataset
from camcontrast.fundus_transforms import TRANSFORM_TRAIN_CONTRASTIVE
from camcontrast.layer_utils import GatherLayer
from camcontrast.model import CAMContrastModel
from camcontrast.train_utils import Callback, JobRunner


class ModelCallback(Callback):
    def __init__(self, temperature):
        self.contrastive_criterion = None
        self.temperature = temperature

    def init_training_criteria(self):
        self.contrastive_criterion = SupConLoss(temperature=self.temperature, base_temperature=self.temperature)

    def compute_training_loss(self, args, model, batch, train_info):
        model = model.module if hasattr(model, "module") else model
        images = batch[0].to(args.device)
        heatmaps = batch[1].to(args.device)

        batch_size = len(heatmaps)
        num_views = 2
        image_projections = model.image_projector(model.forward_encoder(images))
        heatmap_projections = model.heatmap_projector(heatmaps)
        vectors = torch.cat([image_projections, heatmap_projections], dim=1)
        vectors = vectors.reshape(batch_size, num_views, -1)
        if is_distributed():
            vectors = torch.cat(GatherLayer.apply(vectors))

        loss = self.contrastive_criterion(vectors)

        return loss

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
        help="Directory consisting of training images."
    )
    parser.add_argument(
        "--train_heatmaps",
        type=str,
        help="Directory consisting of extracted heatmaps."
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="Path to training labels (.csv file)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature parameter to control the flatness of the softmax curve in the InfoNCE objective."
    )

    args = parse_and_process_arguments(parser)

    callback = ModelCallback(args.temperature)
    model = CAMContrastModel(n_channels=3)
    train_dataset = OIAODIRHeatmapDataset(args.train_images, args.train_heatmaps, args.train_labels)

    job_runner = JobRunner(args, model, callback)
    job_runner.run(train_dataset, None, pin_memory=False, train_transforms_image=TRANSFORM_TRAIN_CONTRASTIVE)


if __name__ == "__main__":
    main()
