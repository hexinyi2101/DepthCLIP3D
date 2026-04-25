import cv2
import torch

from data.build_datasets import DataInfo
from data.process_depth import get_depth_transform, opencv_loader
from torchvision import datasets
'''
def custom_collate_fn(batch):
    # batch中的每个元素形如(image, label, path)
    images, targets, paths = zip(*[(item[0], item[1], path) for item, path in zip(batch, dataset.imgs)])
    images = torch.stack(images, 0)
    targets = torch.tensor(targets)
    return images, targets, paths

def get_depth_dataset(args):
    data_path = args.depth_data_path
    transform = get_depth_transform(args)
    global dataset  # 确保dataset在custom_collate_fn中可访问
    dataset = datasets.ImageFolder(data_path, transform=transform, loader=opencv_loader)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
        collate_fn=custom_collate_fn,  # 使用自定义的collate_fn
    )

    return DataInfo(dataloader=dataloader, sampler=None)

'''
def get_depth_dataset(args):
    data_path = args.depth_data_path
    transform = get_depth_transform(args)
    dataset = datasets.ImageFolder(data_path, transform=transform, loader=opencv_loader)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
