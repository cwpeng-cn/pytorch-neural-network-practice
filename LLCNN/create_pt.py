import torch
from llcnn_net import LLCNN

device = torch.device("cpu")
llcnn = LLCNN()
llcnn.load_state_dict(torch.load("results/pth/20.pth", map_location=device))
llcnn.eval()
x = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(func=llcnn, example_inputs=x)
traced_script_module.save("20_llcnn.pt")
