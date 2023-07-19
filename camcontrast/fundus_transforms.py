import torchvision.transforms as transforms

from camcontrast.transforms_utils import ToMask

OIAODIR_DATASET_MEAN = (0.2934, 0.1877, 0.1025)
OIAODIR_DATASET_STD = (0.3126, 0.2145, 0.1388)

TRANSFORM_TRAIN_CONTRASTIVE = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_TRAIN_CLS = transforms.Compose([
    transforms.Resize(350),
    transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_EVAL_CLS = transforms.Compose([
    transforms.Resize(350),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_TRAIN_SEG = transforms.Compose([
    transforms.Resize(514),
    transforms.RandomCrop(514),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_TRAIN_SEG_MASK = transforms.Compose(
    TRANSFORM_TRAIN_SEG.transforms[0:4] + [ToMask()]
)

TRANSFORM_EVAL_SEG = transforms.Compose([
    transforms.Resize(514),
    transforms.ToTensor(),
    transforms.Normalize(OIAODIR_DATASET_MEAN, OIAODIR_DATASET_STD)
])

TRANSFORM_EVAL_SEG_MASK = ToMask()
