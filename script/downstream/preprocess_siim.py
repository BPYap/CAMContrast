import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm


def rle2mask(rles, width, height):
    mask = np.zeros(width * height)
    if not (isinstance(rles, str) and rles == "-1"):
        if isinstance(rles, str):
            rles = [rles]
        for rle in rles:
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position + lengths[index]] = 255
                current_position += lengths[index]

    return mask.reshape(width, height).T


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--stage_1_train_dir", type=str, required=True)
    arg_parser.add_argument("--stage_1_test_dir", type=str, required=True)
    arg_parser.add_argument("--stage_2_csv", type=str, required=True)
    arg_parser.add_argument("--val_ratio", type=float, default=0.2)
    args = arg_parser.parse_args()

    output_folder = os.path.join(os.path.dirname(args.stage_2_csv), "processed")
    mask_folder = os.path.join(output_folder, "masks")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(mask_folder)

    train_val_list = os.listdir(args.stage_1_train_dir)
    test_list = os.listdir(args.stage_1_test_dir)
    labels = pd.read_csv(args.stage_2_csv, index_col='ImageId')

    # convert RLE to binary masks
    ignored = set()
    for filename in tqdm(train_val_list + test_list, desc="Converting"):
        image_id = ".".join(filename.split(".")[:-1])
        try:
            mask = rle2mask(labels.loc[image_id]['EncodedPixels'], 1024, 1024)
            mask = mask.astype(np.float32)
            mask = torch.tensor(mask / 255)
        except KeyError:
            ignored.add(filename)
            continue
        mask_filename = os.path.join(mask_folder, f"{image_id}.png")
        transforms.functional.to_pil_image(mask).save(mask_filename, quality=100, subsampling=0)
    print("Number of ignored images:", len(ignored))

    train_val_list = list(set(train_val_list).difference(ignored))
    test_list = list(set(test_list).difference(ignored))
    random.shuffle(train_val_list)
    val_count = int(len(train_val_list) * args.val_ratio)
    val_list = train_val_list[:val_count]
    train_list = train_val_list[val_count:]

    for name, _list in zip(["train", "val", "test"], [train_list, val_list, test_list]):
        with open(os.path.join(output_folder, f"{name}.txt"), 'w') as f:
            f.write("\n".join(_list))
