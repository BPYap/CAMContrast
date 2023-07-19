import torchvision.transforms as transforms

from camcontrast.transforms_utils import ToMask

CHESTX_RAY14_MEAN = 0.5056
CHESTX_RAY14_STD = 0.252

TRANSFORM_TRAIN_CONTRASTIVE = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(CHESTX_RAY14_MEAN, CHESTX_RAY14_STD)
])

TRANSFORM_TRAIN_CLS = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(CHESTX_RAY14_MEAN, CHESTX_RAY14_STD)
])

TRANSFORM_EVAL_CLS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CHESTX_RAY14_MEAN, CHESTX_RAY14_STD)
])

TRANSFORM_TRAIN_SEG = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(CHESTX_RAY14_MEAN, CHESTX_RAY14_STD)
])

TRANSFORM_TRAIN_SEG_MASK = transforms.Compose(
    TRANSFORM_TRAIN_SEG.transforms[0:4] + [ToMask()]
)

TRANSFORM_EVAL_SEG = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CHESTX_RAY14_MEAN, CHESTX_RAY14_STD)
])

TRANSFORM_EVAL_SEG_MASK = ToMask()
