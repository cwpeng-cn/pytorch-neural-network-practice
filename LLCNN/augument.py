import torchvision
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


def horizontal_flip(img , ref):
    return TF.hflip(img), TF.hflip(ref)


def vertical_flip(img, ref):
    return TF.vflip(img), TF.vflip(ref)


def random_crop(img, ref, crop_size):
    assert crop_size <= img.width and crop_size <= img.height

    max_left = img.height - crop_size
    max_top = img.width - crop_size

    left = 0
    top = 0
    if max_left > 0:
        left = np.random.randint(low=0, high=max_left, size=1)[0]
    if max_top > 0:
        top = np.random.randint(low=0, high=max_top, size=1)[0]
    img_crop = TF.crop(img, left, top, crop_size, crop_size)
    ref_crop = TF.crop(ref, left, top, crop_size, crop_size)
    return img_crop, ref_crop


if __name__ == "__main__":
    img = Image.open("OHAZE/train/hazy/03_outdoor_hazy.jpg")
    ref = Image.open("OHAZE/train/GT/03_outdoor_GT.jpg")
    random_crop(img, ref, img.height - 1)

    print(help(TF.crop))