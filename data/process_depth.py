# new
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
        '''
        depth_img = image / self.scale  # (H, W)   in meters
        depth_img = depth_img.clip(min=self.min_depth)
        if self.max_depth != 0:
            depth_img = depth_img.clip(max=self.max_depth)
            depth_img /= self.max_depth   #  0-1
        else:
            depth_img /= depth_img.max()
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())
        '''
        #标准化+三通道
        # 替换NaN值为有限数值的平均值
        finite_mean = np.nanmean(image[np.isfinite(image)])
        image = np.nan_to_num(image, nan=finite_mean)

        # 计算平均值和标准差时忽略NaN值
        mean_image = np.nanmean(image)
        std_image = np.nanstd(image)

        if std_image != 0:
            depth_img = (image - mean_image) / std_image
        else:
            # 如果标准差为零，避免除以零的操作，直接设置depth_img为0或其他合适的值
            depth_img = np.zeros_like(image)

        # 使用np.nan_to_num确保任何可能产生的NaN值被转换为0
        depth_img = np.nan_to_num(depth_img, nan=0)
        # depth_img=(image-np.mean(image))/np.std(image)
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())
        #标准化+三通道


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
    if "MiDaS" in depth_path or "dpt_beit" in depth_path: # midas 是视差图
        depth = 1/ (depth + (depth.mean()+depth.max())/2)
    depth_outputs = transform(depth)
    return {'pixel_values': depth_outputs}
