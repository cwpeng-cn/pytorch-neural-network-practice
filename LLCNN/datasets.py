from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
from augument import horizontal_flip, vertical_flip, random_crop


class EnahanceDatasets(Dataset):
    def __init__(self, image_size, data_root, input_dir_name, label_dir_name, h_flip, v_flip, train=True):
        # 定义属性
        self.image_size = image_size
        self.data_dir = data_root
        self.input_dir_name = input_dir_name
        self.label_dir_name = label_dir_name
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.train = train
        if self.train:
            self.prefix = "train_"
        else:
            self.prefix = "val_"

        if not os.path.exists(self.data_dir):
            raise Exception(r"[!] data set does not exist!")

        self.image_file_name = sorted(os.listdir(os.path.join(self.data_dir, self.prefix + self.input_dir_name)))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        img = Image.open(os.path.join(self.data_dir, self.prefix + self.input_dir_name, file_name)).convert('RGB')
        ref = Image.open(os.path.join(self.data_dir, self.prefix + self.label_dir_name, file_name)).convert('RGB')

        img, ref = random_crop(img, ref, self.image_size)

        if self.train:
            if self.h_flip:
                img, ref = horizontal_flip(img, ref)

            if self.v_flip:
                img, ref = vertical_flip(img, ref)

        img = TF.to_tensor(img)
        ref = TF.to_tensor(ref)

        out = {'dark': img, 'clear': ref, "img_name": file_name}

        return out

    def __len__(self):
        return len(self.image_file_name)


# class TestDatasets(Dataset):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.image_size = self.cfg["image_size"]
#         self.data_dir = self.cfg["data_root"]
#
#         if not os.path.exists(self.data_dir):
#             raise Exception(r"[!] data set does not exist!")
#
#         self.image_file_name = sorted(os.listdir(os.path.join(self.data_dir, "test_" + self.cfg["input_dir_name"])))
#
#     def __getitem__(self, item):
#         file_name = self.image_file_name[item]
#         img = Image.open(os.path.join(self.data_dir, "test_" + self.cfg["input_dir_name"], file_name)).convert('RGB')
#         ref = Image.open(os.path.join(self.data_dir, "test_" + self.cfg["label_dir_name"], file_name)).convert('RGB')
#
#         img = TF.resize(img, (self.image_size, self.image_size))
#         ref = TF.resize(ref, (self.image_size, self.image_size))
#
#         img = TF.to_tensor(img)
#         ref = TF.to_tensor(ref)
#
#         out = {'dark': img, 'clear': ref, "img_name": file_name}
#
#         return out
#
#     def __len__(self):
#         return len(self.image_file_name)
#
#
# class ValDatasets(Dataset):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.image_size = self.cfg["image_size"]
#         self.data_dir = self.cfg["data_root"]
#
#         if not os.path.exists(self.data_dir):
#             raise Exception(r"[!] data set does not exist!")
#
#         self.image_file_name = sorted(os.listdir(os.path.join(self.data_dir, "val_" + self.cfg["input_dir_name"])))
#
#     def __getitem__(self, item):
#         file_name = self.image_file_name[item]
#         img = Image.open(os.path.join(self.data_dir, "val_" + self.cfg["input_dir_name"], file_name)).convert('RGB')
#         ref = Image.open(os.path.join(self.data_dir, "val_" + self.cfg["label_dir_name"], file_name)).convert('RGB')
#
#         img = TF.resize(img, (self.image_size, self.image_size))
#         ref = TF.resize(ref, (self.image_size, self.image_size))
#
#         img = TF.to_tensor(img)
#         ref = TF.to_tensor(ref)
#
#         out = {'dark': img, 'clear': ref, "img_name": file_name}
#
#         return out
#
#     def __len__(self):
#         return len(self.image_file_name)
#
#
# class InferDatasets(Dataset):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.image_size = self.cfg["image_size"]
#         self.data_dir = self.cfg["data_root"]
#
#         if not os.path.exists(self.data_dir):
#             raise Exception(r"[!] data set does not exist!")
#
#         self.image_file_name = sorted(os.listdir(os.path.join(self.data_dir, "infer_" + self.cfg["input_dir_name"])))
#
#     def __getitem__(self, item):
#         file_name = self.image_file_name[item]
#         img = Image.open(os.path.join(self.data_dir, "infer_" + self.cfg["input_dir_name"], file_name)).convert('RGB')
#
#         img = TF.resize(img, size=(self.image_size, self.image_size))
#         img = TF.to_tensor(img)
#
#         out = {'dark': img, "img_name": file_name}
#
#         return out
#
#     def __len__(self):
#         return len(self.image_file_name)


if __name__ == "__main__":
    pass
    # cfg_train = read_yaml("cfg/llcnn_train.yaml")
    # train_set = TrainDatasets(cfg_train)
    # print("num of Test set {}".format(len(train_set)))

