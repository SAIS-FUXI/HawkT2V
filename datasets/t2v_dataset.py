import csv
import random
import cv2
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import torch
from torch.utils.data import Dataset, DataLoader
import time

class VideoImageDataset(Dataset):
    def __init__(self, args, transform=None, image_transform=None, fps=24, seconds=2, frames_per_second=8, max_load_frames=100, max_try_times=999, debug=False):
        self.use_image_num = args.use_image_num
        self.use_video_num = args.use_video_num
        self.max_load_frames = max_load_frames
        self.max_try_times = max_try_times
        self.video_data = self._load_data(args.video_json, path_key='video')
        self.image_data = self._load_data(args.image_json, path_key='image')
        self.transform = transform
        self.image_transform = image_transform
        self.fps = args.dataset_config.fps
        self.seconds = args.dataset_config.seconds
        self.frames_seconds = fps * seconds + 1
        
        self.frames_per_second = args.dataset_config.frames_per_second
        self.sampled_frames = seconds * frames_per_second + 1
        self.debug = debug

        # transform
        self.image_size = args.image_size


    def _load_data(self, filename, path_key=None):
        if filename.endswith('.json'):
            return self._load_json_data(filename, path_key=path_key)
        else:
            return self._load_csv_data(filename)

    def _load_csv_data(self, csv_file):
        data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # Skip the header
            for row in reader:
                data.append((row[0], row[1]))  # (path, caption)
        return data

    def _load_json_data(self, json_file, path_key='video'):
        data = []
        with open(json_file, 'r') as f:
            content = json.load(f)
            for it in content:
                if not path_key in it or not 'caption' in it:
                    continue
                data.append((it[path_key], it['caption'])) 
        return data

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):

        # load multiple videos
        video_index_list = [random.randint(0, len(self.video_data)-1) for _ in range(self.use_video_num)]

        video_caption_list = []
        video_data_list = []
        for idx in video_index_list:
            video_path, video_caption = self.video_data[idx]
            
            # Try to load video
            video = None
            for try_time in range(self.max_try_times):  
                try:
                    video = self._load_video(video_path)

                    video_caption_list.append(video_caption)
                    video_data_list.append(video)
                    break
                except Exception:
                    print("failed to load video : ",try_time , video_path)
                    idx = random.randint(0, len(self.video_data) - 1)
                    video_path, video_caption = self.video_data[idx]
        videos_tensor = torch.stack(video_data_list, dim=0)

        # Load multiple images
        images = []
        image_captions = []
        if self.use_image_num >0:
            try_time = 0
            while len(images) < self.use_image_num and try_time < self.max_try_times * 3:  # Allow multiple attempts
                try_time += 1
                image_path, image_caption = random.choice(self.image_data)
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.image_transform:
                        image = self.image_transform(image)
                    images.append(image)
                    image_captions.append(image_caption)
                except Exception:
                    print("failed to load image : ", try_time, image_path)
                    continue
        
        # Concatenate video and images
        if video is not None and images:
            images_tensor = torch.stack(images)
            if self.debug:
                print("videos_tensor.shape = ", videos_tensor.shape, "images_tensor.shape = ", images_tensor.shape)
        else:
            pass

        return {
            'videos': videos_tensor,
            'video_captions': video_caption_list,
            'images': images_tensor,
            'image_captions': image_captions
        }

    def _load_video(self, path):
        start_time = time.time()
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0 
        while True:
            ret, frame = cap.read()
            if frame_count > self.max_load_frames:
                break
            frame_count += 1 
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Ensure at least enough frames for 2 seconds
        num_frames = len(frames)
        if num_frames < self.frames_seconds:
            frames.extend([frames[-1]] * (self.frames_seconds - num_frames))

        # Randomly sample 2 seconds worth of frames
        start_idx = random.randint(0, max(0, num_frames - self.frames_seconds))
        sample_frames = frames[start_idx:start_idx + self.frames_seconds]

        # Uniformly sample 16 frames
        indices = np.linspace(0, self.frames_seconds - 1, self.sampled_frames).astype(int)
        sampled_frames = [Image.fromarray(sample_frames[i]) for i in indices]

        if self.debug:
            print(elapsed_time, ", path = ", path, "num_frames = ", num_frames, "start_idx = ", start_idx, "len(sampled_frames) = ", len(sampled_frames), "indices = ", indices)
        if self.transform:
            sampled_frames = [self.transform(frame) for frame in sampled_frames]

        return torch.stack(sampled_frames)


if __name__ == '__main__':

    import argparse  
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use-image-num", type=int, default=4)
    parser.add_argument("--use-video-num", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--video_json", type=str, default='samples/video_data.json')
    parser.add_argument("--image_json", type=str, default='samples/image_data.json')
    parser.add_argument("--video_folder", type=str, default='')


    config = parser.parse_args()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VideoImageDataset(config, transform=transform, image_transform=transform, fps=24, debug=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # num_workers=16 

    # Test the dataloader
    for i, batch in enumerate(dataloader, start=1):
        print(i, batch['videos'].shape, len(batch['video_captions']), batch['images'].shape, len(batch['image_captions']))