import numpy as np
import cv2
import os


def gamma_func(image, gamma):
    """
    """
    #     assert isinstance(image, np.array)

    image = image / 255.0

    image = image ** gamma

    image = image * 255

    image = image.astype(np.uint8)

    return image


def batch_gamma(ori_dir, save_dir, gamma_range):
    img_files = os.listdir(ori_dir)

    for img_name in img_files:
        image = cv2.imread(os.path.join(ori_dir, img_name))
        gamma = np.random.uniform(low=gamma_range[0],
                                  high=gamma_range[1],
                                  size=1)[0]
        gamma_image = gamma_func(image, gamma)

        cv2.imwrite(os.path.join(save_dir, img_name), gamma_image)
        print("process done for {}".format(img_name))


if __name__ == "__main__":
    batch_gamma(ori_dir="../data/gamma_dataset/test_clear",
                save_dir="../data/gamma_dataset/test_dark",
                gamma_range=[2.5, 3.5])

    batch_gamma(ori_dir="../data/gamma_dataset/train_clear",
                save_dir="../data/gamma_dataset/train_dark",
                gamma_range=[2.5, 3.5])

    batch_gamma(ori_dir="../data/gamma_dataset/val_clear",
                save_dir="../data/gamma_dataset/val_dark",
                gamma_range=[2.5, 3.5])
