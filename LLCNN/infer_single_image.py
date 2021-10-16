import torch
from llcnn_net import LLCNN
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda")
    llcnn = LLCNN().to(device)
    llcnn.load_state_dict(torch.load("results/pth/20.pth"))
    llcnn.eval()
    image_path = "gamma_dataset/val_dark/1350.jpg"

    with torch.no_grad():
        low_light_image = Image.open(image_path)
        low_light_image = low_light_image.resize((256, 256))
        low_light_image = TF.to_tensor(low_light_image).to(device).unsqueeze(0)
        enhanced_image = llcnn(low_light_image)

        result = torch.cat((low_light_image, enhanced_image), dim=3)[0]
        result = result * 255
        result = result.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)

        plt.imshow(result)
        plt.show()
