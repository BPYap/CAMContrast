import torchvision
from torchvision.models._utils import IntermediateLayerGetter

RESNET_CONSTRUCTORS = {
    'resnet-18': lambda **kw: _add_dilation(torchvision.models.resnet.resnet18, **kw),
    'resnet-34': lambda **kw: _add_dilation(torchvision.models.resnet.resnet34, **kw),
    'resnet-50': torchvision.models.resnet.resnet50,
    'resnet-101': torchvision.models.resnet.resnet101,
    'resnet-152': torchvision.models.resnet.resnet152
}

RESNET_OUT_CHANNELS = {
    'resnet-18': 512,
    'resnet-34': 512,
    'resnet-50': 2048,
    'resnet-101': 2048,
    'resnet-152': 2048
}


def _add_dilation(model_fn, **kwargs):
    """Modify BasicBlock in ResNet to support dilation."""
    replace_stride_with_dilation = None
    if "replace_stride_with_dilation" in kwargs:
        replace_stride_with_dilation = kwargs["replace_stride_with_dilation"]
        del kwargs["replace_stride_with_dilation"]
    model = model_fn(**kwargs)
    if replace_stride_with_dilation is not None:
        dilation = prev_dilation = 1
        for dilate, layer in zip(replace_stride_with_dilation, [model.layer2, model.layer3, model.layer4]):
            if dilate:
                dilation *= 2
                layer[0].downsample[0].stride = 1
                layer[0].downsample[0].dilation = (prev_dilation, prev_dilation)
                for block in layer:
                    block.conv1.stride = 1
                    block.conv1.dilation = (prev_dilation, prev_dilation)
                    block.conv1.padding = (prev_dilation, prev_dilation)
                    block.conv2.stride = 1
                    block.conv2.dilation = (dilation, dilation)
                    block.conv2.padding = (dilation, dilation)
                    prev_dilation = dilation

    return model


def get_resnet_encoder(backbone, imagenet_init, use_dilation):
    """
    Args:
        backbone (str): Name of the encoder backbone.
        imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                              pre-trained on ImageNet.
        use_dilation (bool): Whether to replace strides in the last two blocks with dilation.
    """
    if use_dilation:
        replace_stride_with_dilation = [False, True, True]
    else:
        replace_stride_with_dilation = None

    if backbone in RESNET_CONSTRUCTORS:
        resnet = RESNET_CONSTRUCTORS[backbone]
        backbone = resnet(pretrained=imagenet_init, replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low_level', 'layer4': 'out'}
        encoder = IntermediateLayerGetter(backbone, return_layers=return_layers)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    return encoder
