#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time    : 2019/12/11 13:56 
# @Author  : CongXiaofeng 
# @File    : save.py 
# @Software: PyCharm


import numpy as np
from PIL import Image


def save_image(image_tensor, out_name):
    """
    save a single image
    :param image_tensor: torch tensor with size=(3, h, w)
    :param out_name: path+name+".jpg"
    :return: None
    """
    if len(image_tensor.size()) == 3:
        image_numpy = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
        image_numpy = (image_numpy * 255).astype(np.uint8)
        image = Image.fromarray(image_numpy)
        image.save(out_name)
    else:
        raise ValueError("input tensor not with size (3, h, w)")
    return None

