# new,1/1/x
import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')


class DepthNorm(nn.Module):
    def __init__(
        self,
        max_depth=0,
        min_depth=0.01,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.scale = 1000.0  # nyuv2 abs.depth

    def forward(self, image):
        
        # 改变视差图分布
        image = 1 / (1 / ( image + image.mean() ) + 1/image.mean())

        # 标准化+三通道
        depth_img=(image-np.mean(image))/np.std(image)
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())
        # 标准化+三通道


def get_depth_transform(args):


    transform = transforms.Compose(
        [
            DepthNorm(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            
        ]
    )

    return transform

def load_and_transform_depth(depth_path, transform):
    depth = opencv_loader(depth_path)
    depth_outputs = transform(depth)
    return {'pixel_values': depth_outputs}
