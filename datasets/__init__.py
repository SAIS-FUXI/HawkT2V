from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from datasets import video_transforms
from .t2v_dataset import VideoImageDataset


def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 't2v':
        transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC, max_size=None),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return VideoImageDataset(args, transform=transform, image_transform=transform, debug=False)
    
    else:
        raise NotImplementedError(args.dataset)