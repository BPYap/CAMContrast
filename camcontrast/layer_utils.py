import torch
import torch.distributed as dist
import torch.nn as nn

from camcontrast.config import is_distributed


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.

    adapted from https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]

        return grad_out


def get_cams(feature_maps, classification_layer, targets, upsample_size=None):
    num_feat_maps = feature_maps.shape[1]
    # extract weights associated with the target class
    weights = classification_layer.weight[targets, :].reshape((-1, num_feat_maps, 1, 1))
    # each cam is the weighted sum of the feature maps
    cams = (weights * feature_maps).sum(1).unsqueeze(1)
    if upsample_size:
        # upsample each cam to match the target size
        up_sample = nn.Upsample(size=upsample_size, mode='bilinear', align_corners=True)
        cams = up_sample(cams)

    return cams


def get_heatmaps(feature_maps, model, upsample_size):
    if is_distributed():
        model = model.module

    batch_size = len(feature_maps)
    positive_cams = get_cams(
        feature_maps=feature_maps,
        classification_layer=model.out["classification"][-1],
        targets=[1] * batch_size,
        upsample_size=upsample_size
    ).squeeze(dim=1)

    # min-max normalization
    batch_size = positive_cams.shape[0]
    max_ = positive_cams.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
    min_ = positive_cams.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
    heatmaps = (positive_cams - min_) / (max_ - min_)

    return heatmaps
