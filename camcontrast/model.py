import torch
import torch.nn as nn
import torch.nn.functional as F

from camcontrast.resnet import get_resnet_encoder, RESNET_OUT_CHANNELS
from camcontrast.unet import UNet, UNetEncoder

AVAILABLE_BACKBONES = [
    'u-net-encoder', 'resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152'
]
AVAILABLE_BACKBONES_SEGMENTATION = {
    'u-net': ['default', 'resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152'],
}
ENCODER_OUT_CHANNELS = {
    **RESNET_OUT_CHANNELS,
    "u-net-encoder": 1024
}


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MLPProjectionNetwork(nn.Module):
    def __init__(self, in_channels, output_dim=128):
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, output_dim)
        )

    def forward(self, x):
        x = self.out(x)

        return F.normalize(x)


class CNNProjectionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, output_dim=128):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=(3, 3), bias=False),
            nn.ReLU(inplace=True)
        )
        self.out = MLPProjectionNetwork(hidden_dim * 4, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x).flatten(1)

        return self.out(x)


class BaseEncoder(nn.Module):
    def __init__(self, n_channels):
        """
        Args:
            n_channels (int): Number of channels in input images.
        """
        super().__init__()

        self.n_channels = n_channels
        self.encoder = None

    def init(self, backbone, imagenet_init):
        """
        Args:
            backbone (str): Name of the encoder backbone.
            imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                                  pre-trained on ImageNet.
        """
        if "resnet" in backbone:
            self.encoder = get_resnet_encoder(backbone, imagenet_init, use_dilation=False)
        elif backbone == "u-net-encoder":
            self.encoder = UNetEncoder(self.n_channels)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'.")

    def forward(self, x):
        if self.encoder is None:
            raise RuntimeError("Model is not initialized. Call `init` to initialize the model.")

        return self.encoder(x)['out']


class JointClassificationModel(BaseEncoder):
    def __init__(self, n_channels, num_class_per_task):
        """
        Args:
            n_channels (int): Number of channels in input images.
            num_class_per_task (dict): Mapping of task name to the number of classes.
        """
        super().__init__(n_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_class_per_task = num_class_per_task

        self.dropout_rate = 0.0
        self.out = nn.ModuleDict()

    def init(self, backbone, imagenet_init):
        super().init(backbone, imagenet_init)

        for task_name, num_classes in self.num_class_per_task.items():
            out = nn.Sequential(
                nn.Linear(ENCODER_OUT_CHANNELS[backbone], num_classes)
            )
            self.out[task_name] = out

    def set_dropout_rate(self, dropout_rate=0.0):
        self.dropout_rate = dropout_rate

    def forward(self, x):
        feat_maps = super().forward(x)
        x = self.avg_pool(feat_maps)
        x = torch.flatten(x, 1)
        # x = F.dropout(x, p=self.dropout_rate, training=self.training)

        logits = dict()
        for task_name, out in self.out.items():
            logits[task_name] = out(F.dropout(x, p=self.dropout_rate, training=self.training))

        return feat_maps, logits


class ClassificationModel(JointClassificationModel):
    def __init__(self, n_channels, num_classes, output_name=None):
        """
        Args:
            n_channels (int): Number of channels in input images.
            num_classes (int): Number of classes.
            output_name (str): Optional name given to the output layer.
        """
        self.output_name = 'classification' if output_name is None else output_name
        super().__init__(n_channels, {self.output_name: num_classes})

    def forward(self, x):
        feat_maps, logits = super().forward(x)

        return feat_maps, logits[self.output_name]


class ContrastiveLearningModel(BaseEncoder):
    def __init__(self, n_channels, proj_heads=1):
        """
        Args:
            n_channels (int): Number of channels in input images.
            proj_heads (int): Number of projection networks.
        """
        super().__init__(n_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projection_networks = None
        self.proj_heads = proj_heads

    def init(self, backbone, imagenet_init):
        super().init(backbone, imagenet_init)

        self.projection_networks = nn.ModuleList([
            MLPProjectionNetwork(ENCODER_OUT_CHANNELS[backbone], 128) for _ in range(self.proj_heads)
        ])

    def forward_encoder(self, x):
        feat_maps = super().forward(x)
        feat_vectors = self.avg_pool(feat_maps).flatten(1)

        return feat_vectors

    def forward(self, x):
        feat_vectors = self.forward_encoder(x)
        if self.proj_heads == 1:
            return self.projection_networks[0](feat_vectors)
        else:
            return [proj(feat_vectors) for proj in self.projection_networks]


class CAMContrastModel(BaseEncoder):
    def __init__(self, n_channels):
        """
        Args:
            n_channels (int): Number of channels in input images.
        """
        super().__init__(n_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.image_projector = None
        self.heatmap_projector = None

    def init(self, backbone, imagenet_init):
        super().init(backbone, imagenet_init)

        self.image_projector = MLPProjectionNetwork(ENCODER_OUT_CHANNELS[backbone], 128)
        self.heatmap_projector = CNNProjectionNetwork(1, 32, 128)

    def forward_encoder(self, x):
        feat_maps = super().forward(x)
        feat_vectors = self.avg_pool(feat_maps).flatten(1)

        return feat_vectors


def _get_out_channels(architecture, encoder_backbone):
    encoder_out_channels, decoder_out_channels = None, None
    if architecture == 'u-net':
        if encoder_backbone == 'u-net-encoder':
            encoder_out_channels = 1024
            decoder_out_channels = 64
        elif encoder_backbone in ['resnet-18', 'resnet-34']:
            encoder_out_channels = RESNET_OUT_CHANNELS[encoder_backbone]
            decoder_out_channels = 16
        elif encoder_backbone in ['resnet-50', 'resnet-101', 'resnet-152']:
            encoder_out_channels = RESNET_OUT_CHANNELS[encoder_backbone]
            decoder_out_channels = 64

    if not encoder_out_channels or not decoder_out_channels:
        raise ValueError(f"Unsupported architecture and backbone '{architecture}', '{encoder_backbone}'.")

    return encoder_out_channels, decoder_out_channels


class BaseEncoderDecoder(nn.Module):
    def __init__(self, n_channels):
        """
        Args:
            n_channels (int): Number of channels in input images.
        """
        super().__init__()
        self.n_channels = n_channels

        self.architecture = None
        self.encoder = None
        self.decoder = None
        self.encoder_out_channels = None
        self.decoder_out_channels = None

    def init(self, architecture, encoder_backbone, imagenet_init):
        """
        Args:
            architecture (str): Name of the encoder-decoder architecture.
            encoder_backbone (str): Name of the encoder backbone.
            imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                                  pre-trained on ImageNet.
        """
        n_channels = self.n_channels

        if architecture == 'u-net':
            encoder_decoder = UNet(
                n_channels,
                bilinear=True,
                encoder_backbone=encoder_backbone,
                imagenet_init=imagenet_init
            )
        else:
            raise ValueError(f"Unknown architecture '{architecture}'")

        self.architecture = architecture
        self.encoder = encoder_decoder.encoder
        self.decoder = encoder_decoder.decoder
        self.encoder_out_channels, self.decoder_out_channels = _get_out_channels(architecture, encoder_backbone)

    def forward_encoder(self, x):
        architecture = self.architecture
        encoder = self.encoder

        if encoder is None:
            raise RuntimeError("Model is not initialized. Call `init` to initialize the model.")

        if architecture == 'u-net':
            # input x is the processed images tensor
            # output xs is a list of feature maps from each Down block
            xs = [x]
            for key, down in encoder.items():
                xs.append(down(xs[-1]))

            return xs[1:]

    def forward_decoder(self, xs):
        architecture = self.architecture
        decoder = self.decoder

        if decoder is None:
            raise RuntimeError("Model is not initialized. Call `init` to initialize the model.")

        if architecture == 'u-net':
            # input xs is a list of feature maps from each Down block
            # output x is the decoded feature maps
            x = xs.pop()
            for up in decoder.values():
                x = up(x, xs.pop())

            return x

class JointSegmentationModel(BaseEncoderDecoder):
    def __init__(self, n_channels, num_class_per_task):
        """
        Args:
            n_channels (int): Number of channels in input images.
            num_class_per_task (dict): Mapping of task name to the number of classes.
        """
        super().__init__(n_channels)

        self.num_class_per_task = num_class_per_task
        self.out = nn.ModuleDict()

    def init(self, architecture, encoder_backbone, imagenet_init):
        super().init(architecture, encoder_backbone, imagenet_init)

        for task_name, num_classes in self.num_class_per_task.items():
            out = OutConv(self.decoder_out_channels, num_classes)
            self.out[task_name] = out

    def forward(self, x):
        # segment images using encoder and decoder
        input_shape = x.shape[-2:]
        feature_maps = self.forward_decoder(self.forward_encoder(x))
        logits = dict()
        for task_name, out in self.out.items():
            temp = out(feature_maps)
            if temp.shape[-2:] != input_shape:
                temp = F.interpolate(temp, size=input_shape, mode='bilinear', align_corners=False)
            logits[task_name] = temp

        return logits


class SegmentationModel(JointSegmentationModel):
    def __init__(self, n_channels, num_classes):
        """
        Args:
            n_channels (int): Number of channels in input images.
            num_classes (int): Number of classes.
        """
        super().__init__(n_channels, {'segmentation': num_classes})

    def forward(self, x):
        return super().forward(x)['segmentation']
