# new 3 channel
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
        # 确保图像是8位的
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        # 应用颜色映射
        image_colored = cv2.applyColorMap(image_8bit, cv2.COLORMAP_INFERNO)
        
        # 对图像进行标准化处理
        depth_img = (image_colored - np.mean(image_colored)) / np.std(image_colored)
        
        # 将numpy数组转换为torch张量
        normalized_img_tensor = torch.from_numpy(depth_img)
        
        # 由于PyTorch期望的张量形状是 [通道数, 高度, 宽度]，我们需要调整张量的维度
        normalized_img_tensor = normalized_img_tensor.permute(2, 0, 1)
        
        # 转换数据类型
        normalized_img_tensor = normalized_img_tensor.to(torch.get_default_dtype())
        
        # 返回标准化后的图像张量
        return normalized_img_tensor


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
