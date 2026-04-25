
# new
import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
import numpy as np

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

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
        self.scale = 1000.0  # nyuv2 abs.depth   sun8000

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


       

def get_depth_transform(config):
    config = config.vision_config

    transform = transforms.Compose(
        [
            DepthNorm(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            
        ]
    )

    return transform

'''def load_and_transform_depth(depth_path, transform):
    depth = opencv_loader(depth_path)
    if "MiDaS" in depth_path: # midas 是视差图
        depth = 1/ (depth + (depth.mean()+depth.max())/2)
    depth_outputs = transform(depth)
    return {'pixel_values': depth_outputs}'''
def load_and_transform_depth(depth_path, transform):
    depth = opencv_loader(depth_path)
    if "MiDaS" in depth_path or "dpt_beit" in depth_path: # midas 是视差图
        depth = 1/ (depth + (depth.mean()+depth.max())/2)
    depth_outputs = transform(depth)
    return depth_outputs

class LanguageBindDepthProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindDepthTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_depth_transform(config)
        self.image_processor = load_and_transform_depth
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
