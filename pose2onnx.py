import torchvision
import torch.onnx
import torch
import cv2
import numpy as np

### simplebaseline res
from pose_resnet import get_pose_net
net = get_pose_net()
net.cuda()
checkpoint = torch.load('./checkpoint/res50_256.pth') # hw 256x192
net.load_state_dict(checkpoint, strict=False)

net.eval()
np_input = cv2.imread('./data/pose_test_00228.png', 1)
np_input = cv2.resize(np_input, (192, 256))
np_input = np_input.transpose((2,0,1))
np_input = np_input.reshape((1,3,256,192)).astype(np.float32)
dummy_input = torch.from_numpy(np_input).cuda()
# dummy_input = torch.rand(1,3,256,192).cuda()
torch.onnx.export(net, dummy_input, './checkpoint/res50_256.onnx', output_names=['output'])

