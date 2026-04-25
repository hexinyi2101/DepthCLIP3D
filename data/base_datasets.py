import contextlib
import io
import json
import logging
import os.path
import random
import re
import time

import pandas as pd

from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX
from .process_video import load_and_transform_video, get_video_transform
from .process_audio import load_and_transform_audio, get_audio_transform
from .process_text import load_and_transform_text
from .process_depth import load_and_transform_depth, get_depth_transform
from .process_thermal import load_and_transform_thermal, get_thermal_transform

from .process_image import load_and_transform_image, get_image_transform   # new


import argparse
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


import random

class VAT_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.video_decode_backend = args.video_decode_backend
        self.num_frames = args.num_frames
        self.text_type = args.text_type
        self.chatgpt = self.text_type == 'polish_mplug'
        self.title = self.text_type == 'raw'
        self.data_root = '/A_Youtube'
        '''
        with open(args.train_data, 'r') as f:
            self.id2title_folder_caps = json.load(f)
        self.ids = list(self.id2title_folder_caps.keys())[:args.train_num_samples]
        '''

        # 打乱，并取前一部分
        with open(args.train_data, 'r', encoding='utf-8') as f:
            self.id2title_folder_caps = json.load(f)
        all_ids = list(self.id2title_folder_caps.keys())
        random.shuffle(all_ids)  # 原地打乱列表
        self.ids = all_ids[:args.train_num_samples]
        
        # 更新为随机选择一个值
        for key in self.ids:
            if isinstance(self.id2title_folder_caps[key], list):
                self.id2title_folder_caps[key] = random.choice(self.id2title_folder_caps[key])

        self.clip_type = args.clip_type

        self.num_mel_bins = args.num_mel_bins
        self.target_length = args.target_length
        self.audio_sample_rate = args.audio_sample_rate
        self.audio_mean = args.audio_mean
        self.audio_std = args.audio_std

        # self.audio_error_file = open('./audio_error_id.txt', 'w')

        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.video_transform = get_video_transform(args)
        self.audio_transform = get_audio_transform(args)
        self.depth_transform = get_depth_transform(args)
        self.thermal_transform = get_thermal_transform(args)
        self.image_transform = get_image_transform(args)


    def __len__(self):
        return len(self.ids)
        # return self.id2title_folder_caps.shape[0]

    def __getitem__(self, idx):
        id = self.ids[idx]
        # folder = self.id2title_folder_caps[id]['folder']
        folder = 1
        try:
            text_output = self.get_text(id)
            input_ids, attention_mask = text_output['input_ids'], text_output['attention_mask']
            if self.clip_type == 'dl':
                matched_modality = self.get_depth(id, folder)
                image_modality = self.get_image(id, folder)  # 假设有一个类似的函数来处理RGB图像  
                       
            # if self.clip_type == 'vl':
            #     matched_modality = self.get_video(id, folder)
            # elif self.clip_type == 'al':
            #    matched_modality = self.get_audio(id, folder)
            # elif self.clip_type == 'dl':
            #     matched_modality = self.get_depth(id, folder)
            # elif self.clip_type == 'tl':
            #     matched_modality = self.get_thermal(id, folder)
            return matched_modality['pixel_values'], input_ids, attention_mask,image_modality['pixel_values']
        except Exception as error_msg:
            logging.info(f"Failed at {id} with \"{error_msg}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def get_video(self, id, folder):
        video_path = opj(self.data_root, folder, f'{id}.mp4')
        video = load_and_transform_video(video_path, self.video_transform,
                                         video_decode_backend=self.video_decode_backend, num_frames=self.num_frames)
        return video

    def get_audio(self, id, folder):
        '''
        audio_path = opj(self.data_root, folder, f'{id}.mp3')
        if os.path.exists(audio_path):
            pass
        else:
            audio_path = audio_path[:-4] + '.m4a'
            if os.path.exists(audio_path):
                pass
            else:
                audio_path = audio_path[:-4] + '.wav'
                if not os.path.exists(audio_path):
                    # self.audio_error_file.write(audio_path[:-4] + '\n')
                    raise FileNotFoundError(f'Not found audio file at \'{audio_path[:-4]}\' with .mp3 .m4a .wav')
            # AudioSegment.from_file(audio_path).export(audio_path[:-4] + '.mp3', format='mp3')
            # audio_path = opj(self.data_root, folder, f'{id}.mp3')
        audio = load_and_transform_audio(audio_path, self.audio_transform)
        '''

        audio_path = opj(self.data_root, folder+'_ffmpeg_mp3', f'{id}.mp3')
        audio = load_and_transform_audio(audio_path, self.audio_transform)


        return audio

    

    def get_text(self, id):
        text = self.id2title_folder_caps[id]
        text_output = load_and_transform_text(text, self.tokenizer, title=self.title)
        return text_output

    def get_image(self, depth_path,folder):

        # 使用转换后的路径来加载和转换RGB图像
        if "nyuv2_full" in depth_path:
            image = load_and_transform_image(depth_path.replace("nyuv2_full/depth","nyuv2_full/rgb"), self.image_transform)
        elif "depthsplit" in depth_path:
            image = load_and_transform_image(depth_path.replace("depthsplit","rgbsplit").replace("png","jpg"), self.image_transform)
        elif "scannet166" in depth_path:
            image = load_and_transform_image(depth_path.replace("scannet166/depth","scannet166/rgb").replace("png","jpg"), self.image_transform)
        elif "objectdepth" in depth_path:
            image = load_and_transform_image(depth_path.replace("objectdepth","objectrgb").replace("depth_","rgb_").replace("png","jpg"), self.image_transform)
        elif "downstream_datasets" in depth_path and "sun" in depth_path:
            image = load_and_transform_image(depth_path.replace("_depth","_rgb").replace("png","jpg").replace("_d_","_rgb_"), self.image_transform)
        elif "ILSVRC2013" in depth_path:
            image = load_and_transform_image(depth_path.replace("_depth","_rgb").replace("-dpt_beit_large_512.png",".JPEG"), self.image_transform)
        elif "sampleobj" in depth_path:
            image = load_and_transform_image(depth_path.replace('sampleobj/depth', 'sampleobj/render_images_split_100').replace('-dpt_beit_large_512.png', '.png'), self.image_transform)
        elif "sunrgbd" in depth_path:
            if "_denoised" in depth_path:
                directory, filename = os.path.split(depth_path)
                prefix = ''.join(filter(lambda x: not x.isdigit(), filename.split('_')[0]))
                new_directory = directory.replace("train_fill_depth", "train_class_depth_unified") + "/" + prefix
                new_filename = filename.replace("rgb__", "d__").replace("_denoised",'')
                depth_path = os.path.join(new_directory, new_filename)
            fullname = depth_path
            parts = depth_path.split('/')
            subdir_name = parts[-2]
            filename = parts[-1]
            depth_path = filename.replace('d_', 'rgb_').replace('.png', '.jpg')
            if "train" in fullname:
                base_target_path = "/media/omnisky/sdb/grade2020/liangguang/hxy/depth/mydata/sunrgbd/mychange/train_rgb"
            else:
                base_target_path = "/media/omnisky/sdb/grade2020/liangguang/hxy/depth/mydata/sunrgbd/mychange/test_rgb"
            depth_path = os.path.join(base_target_path, depth_path)
            image = load_and_transform_image(depth_path, self.image_transform)
        elif "imagenet_train" in depth_path:
            new_base_path = "/media/omnisky/sdb/grade2020/liangguang/hxy/depth/mydata/imagenet/train"
            depth_path_relative = depth_path.split('imagenet_train')[-1]
            image_path = depth_path_relative.replace("-dpt_beit_large_512.png", ".jpg")

            rgb_image_path = os.path.join(new_base_path, image_path.strip('/'))

            image = load_and_transform_image(rgb_image_path, self.image_transform)            
        elif "place365" in depth_path:
            new_base_path = "/media/omnisky/sdb/grade2020/liangguang/hxy/depth/mydata/place365/data_large"
            parts = depth_path.split('/')
            subsubdir_name = parts[-3]
            subdir_name = parts[-2]
            filename = parts[-1]
            image_path = filename.replace("-dpt_beit_large_512.png", ".jpg")
            # print(parts[-4])  
            if parts[-4] == "place365":
                rgb_image_path = os.path.join(new_base_path,subsubdir_name, subdir_name, image_path)
            else:
                rgb_image_path = os.path.join(new_base_path,parts[-4],subsubdir_name, subdir_name, image_path)
            image = load_and_transform_image(rgb_image_path, self.image_transform)
        elif "only_rgb_depth_images" in depth_path:
            parts = depth_path.split('/')
            subdir_name = parts[-2]
            filename = parts[-1]
            rgb_filename = filename.replace("_depth0001", "")
            rgb_path = depth_path.replace("/depth/", "/rgb/").replace(filename, rgb_filename)
            image = load_and_transform_image(rgb_path, self.image_transform)
        return image



    def get_depth(self, id, folder):
        '''
        depth_folder = opj(self.data_root, folder, f'{id}_depth_f8glpn_folder')
        # random_id = random.randint(0, 7)
        random_id = 3
        depth_path = os.path.join(depth_folder, f'{random_id}.png')
        depth = load_and_transform_depth(depth_path, self.depth_transform)
        '''
        depth = load_and_transform_depth(id, self.depth_transform)
        
        return depth

    def get_thermal(self, id, folder):
        thermal_folder = opj(self.data_root, folder, f'{id}_thermal_f8_folder')
        # random_id = random.randint(0, 7)
        random_id = 3
        thermal_path = os.path.join(thermal_folder, f'{random_id}.jpg')
        thermal = load_and_transform_thermal(thermal_path, self.thermal_transform)
        return thermal





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument('--num_frames', default=8, type=float, help='')
    parser.add_argument('--workers', default=10, type=int, help='')
    args = parser.parse_args()

    args.cache_dir = 'D:\Omni-modal-hf'
    args.num_frames = 8
    args.clip_type = 'vl'
    args.num_mel_bins = 128
    args.target_length = 1024
    args.audio_sample_rate = 16000
    args.audio_mean = 1
    args.audio_std = 1
    args.rank = 0
    args.batch_size = 16

    train_dataset = VAT_dataset(args)
    load = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    for samples in tqdm((load)):
        matched_modality, input_ids, attention_mask = samples
        # print(video.shape, text.shape)