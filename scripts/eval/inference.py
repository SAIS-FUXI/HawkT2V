import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append('.')
from tqdm import tqdm
import io
import os
import math
import argparse
import pickle as pkl
import imageio
import json
# import wandb
import html
from typing import Callable, List, Optional, Tuple, Union
import inspect
import re
import cv2
import urllib.parse as ul
import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from collections import defaultdict
import pickle

from datasets import get_dataset
from models import get_models
from models.clip import TextEmbedder
from models.text_encoder import T5Encoder
from diffusion import create_diffusion


from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import T5EncoderModel, T5Tokenizer
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir)
# from download import find_model
from utils import save_video_grid
from diffusers.utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from scripts.eval.pipeline_hawkgen import HawkT2VPipeline
if is_bs4_available():
    from bs4 import BeautifulSoup
if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import random 
import einops

from models.video_vae import CausualVAEVideo

def get_added_info(video_file, device='cpu'):
    video_data_info = defaultdict(list)
    assert os.path.isfile(video_file), f"{video_file} not exist."
    with open(video_file, 'rb') as f:
        data = pickle.load(f)
    origin_resolution = data["origin_resolution"][0] #（h, w）
    ori_height = origin_resolution[0]
    ori_width = origin_resolution[1]
    ori_aspect_ratio = ori_height/ori_width
    t_frames=data["t_frames"]

    ori_frame_length = data["origin_frames"][0]
    crop_top_left = list(data["crop_top_left"][0])

    video_data_info["ori_height"].append(ori_height)
    video_data_info["ori_width"].append(ori_width)
    video_data_info["ori_aspect_ratio"].append(ori_aspect_ratio)
    video_data_info["crop_top_left"].append(crop_top_left)

    video_data_info["t_frames"].append(t_frames)
    print(f'video_data_info t_frames is {t_frames}', flush=True)
    video_data_info["target_height"].append(256)
    video_data_info["target_width"].append(256)

    data_info = {}
    for key, _ in video_data_info.items():
        data_info[key] = torch.tensor(video_data_info[key]).unsqueeze(0).to(device)

    return data_info

def parse_added_info(info_list, device='cpu'):
    video_data_info = defaultdict(list)
    video_data_info["ori_height"].append(info_list[0])
    video_data_info["ori_width"].append(info_list[1])
    video_data_info["ori_aspect_ratio"].append(info_list[2])
    video_data_info["crop_top_left"].append([info_list[3],info_list[4]])

    video_data_info["t_frames"].append(info_list[5])
    video_data_info["target_height"].append(info_list[6])
    video_data_info["target_width"].append(info_list[7])

    data_info = {}
    for key, _ in video_data_info.items():
        data_info[key] = torch.tensor(video_data_info[key]).unsqueeze(0).to(device)

    return data_info


def set_added_info(device='cpu', condition_type='high_resolution'):
    video_data_info = defaultdict(list)
    if condition_type == 'high_resolution':
        video_data_info["ori_height"].append(720)
        video_data_info["ori_width"].append(1280)
        video_data_info["ori_aspect_ratio"].append(9.0/16)
        video_data_info["crop_top_left"].append([0, 150])

    elif condition_type == 'low_resolution':
        video_data_info["ori_height"].append(240)
        video_data_info["ori_width"].append(384)
        video_data_info["ori_aspect_ratio"].append(10.0/16)
        video_data_info["crop_top_left"].append([0, 50])

    else:
        raise("condition_type should be high_resolution or low_resolution")
    
    video_data_info["t_frames"].append(17) #25,33, not 5,7,9
    video_data_info["target_height"].append(512)
    video_data_info["target_width"].append(512)
    data_info = {}
    for key, _ in video_data_info.items():
        data_info[key] = torch.tensor(video_data_info[key]).unsqueeze(0).to(device)

    return dict(data_info)

def init_video_vae(vae_config, vae_model_path):
    videovae_config = OmegaConf.load(vae_config)
    args.fps_ds = videovae_config.fps_ds
    videovae = CausualVAEVideo(ddconfig=videovae_config.ddconfig, embed_dim=videovae_config.embed_dim)
    checkpoint = torch.load(vae_model_path)
    msg = videovae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("vae load from {}".format(vae_model_path))
    print(msg)
    for p in videovae.parameters():
        p.requires_grad = False
    t_downsampling = 2**(sum([s==2 for s in videovae_config.ddconfig.temporal_stride]))
    assert t_downsampling in [4, 8]
    videovae.patch_size = (t_downsampling, 8, 8)    # TODO: vae spatial downsampling is fixed to 8
    videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_video")
    return videovae

def decode_latents(vae, latents, decode_length=7):
    print("vae.scaling_factor = ", vae.scaling_factor)
    latents = latents / vae.scaling_factor
    # video_length = latents.shape[2]
    video_length = (decode_length-1)*4 + 1
    B, _, T, H, W = latents.shape
    video_recons = vae.decode(latents[:, :, :decode_length, :, :]) #TODO: only decode the first 3 seconds
    video_recons = torch.clamp(video_recons, -1.0, 1.0)
    video_recons = (video_recons + 1.0) / 2.0
    video_recons = (video_recons * 255).type(torch.uint8)
    return einops.rearrange(video_recons, "b c f h w -> b f h w c", f=video_length)

def gen_single_sample(videogen_pipeline, prompt, args, samples_dir, train_ts, step, 
                      negative_prompt="",
                      prompt_embeds=None, 
                      negative_prompt_embeds=None,
                      output_type='video',
                      added_cond_kwargs=None):
    print('Processing the ({}) prompt'.format(prompt))
    videos = videogen_pipeline(prompt=prompt if prompt_embeds is None else None,
                               negative_prompt=negative_prompt,
                               video_length=args.video_length,
                               height=args.validation.image_size[0],
                               width=args.validation.image_size[1],
                               num_inference_steps=args.validation.num_sampling_steps,
                               guidance_scale=args.validation.guidance_scale,
                               enable_temporal_attentions=args.validation.enable_temporal_attentions,
                               num_images_per_prompt=1,
                               mask_feature=True, # False
                               enable_vae_temporal_decoder=args.validation.enable_vae_temporal_decoder,
                               prompt_embeds=prompt_embeds,
                               negative_prompt_embeds=negative_prompt_embeds,
                               output_type=output_type,
                               added_cond_kwargs=added_cond_kwargs,
                               ).video
    if output_type == 'latents':    
        videos = decode_latents(videogen_pipeline.vae, videos, decode_length=args.video_length).cpu()
    
    # try:
    if True:
        if args.video_length > 1:
            file_prefix = prompt.replace(' ', '_')[:120]
            imageio.mimwrite(samples_dir + '/' + file_prefix + f'{train_ts}_{step}_webv_imageio.mp4',
                             videos[0], fps=8, quality=9)  # highest quality is 10, lowest is 0
            print('saved in path {}'.format(samples_dir + '/' + file_prefix + f'{train_ts}_{step}_webv_imageio.mp4'))
        else:
            file_prefix = prompt.replace(' ', '_')[:120]
            cv2.imwrite(samples_dir + '/' + file_prefix + f'{train_ts}_{step}_webv_imageio.jpg',
                             videos[0][0].numpy()[:, :, ::-1])  # highest quality is 10, lowest is 0
            print('saved in path {}'.format(samples_dir + '/' + file_prefix + f'{train_ts}_{step}_webv_imageio.jpg'))

    # except:
    #     print('Error when saving {}'.format(prompt))
    return videos

def gen_validation_samples(videogen_pipeline, args, samples_dir, train_ts, step):
    video_grids = []
    for prompt in args.validation.text_prompt:
        videos = gen_single_sample(videogen_pipeline, prompt, args, samples_dir, train_ts, step)
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    to_save_path = samples_dir + "/" + f'{train_ts}_{step}_merged_.mp4'
    imageio.mimwrite(to_save_path, video_grids, fps=8,
                     quality=5)
    print('saved in path {}'.format(to_save_path))

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    global logger
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")


    if rank == 0:
        
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        num_frame_string = 'F' + str(args.video_length) + 'S' + str(args.frame_interval)
        dir_name = f"ckpt_{checkpoint_name}-{num_frame_string}-{args.validation.sample_method}-cfg_{args.validation.guidance_scale}-samplingsteps{args.validation.num_sampling_steps}-seed{args.global_seed}" 
        samples_dir = f"{args.results_dir}/samples/{dir_name}-ema"
        print(f'samples_dir is {samples_dir}', flush=True)
        os.makedirs(samples_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        test_list_name = os.path.splitext(os.path.basename(args.input_valid_list))[0]
        samples_dir = f"{samples_dir}/{test_list_name}"
        os.makedirs(samples_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        OmegaConf.save(args, os.path.join(samples_dir, 'config.yaml'))
        tb_writer = None
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    output_type = 'video'
    if args.extras == 78 and hasattr(args, 'vae_config'):
        vae = init_video_vae(args.vae_config, args.vae_model_path).eval().to(device)
        output_type = 'latents'
    elif args.extras == 78:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="sd-vae-ft-mse").to(device)

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    # set distributed training
    model = model.to(device)
    model.eval()  
    # import ipdb; ipdb.set_trace();
    
    if args.extras == 78 and (args.dataset == 't2v' or args.dataset == 't2v_offline'):
        #text_encoder = TextEmbedder(args.pretrained_model_path, dropout_prob=0.1).to(device)

        t5_encoder = T5Encoder(from_pretrained=args.pretrained_text_model_path,
                                    model_max_length=200,
                                    shardformer=True)
        text_encoder = t5_encoder.t5.model
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        tokenizer = t5_encoder.t5.tokenizer
    

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if args.extras == 78 and args.dataset == 't2v':
        text_encoder.requires_grad_(False)



    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    test_ts = int(start_time)


    validation_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_path,
                                              subfolder="scheduler",
                                              beta_start=args.validation.beta_start,
                                              beta_end=args.validation.beta_end,
                                              beta_schedule=args.validation.beta_schedule,
                                              variance_type=args.validation.variance_type)

    # import ipdb; ipdb.set_trace();
    validation_pipeline = HawkT2VPipeline(vae=vae,
                                 text_encoder=text_encoder,
                                 tokenizer=tokenizer,
                                 scheduler=validation_scheduler,
                                 transformer=model).to(device)

 
    with open(args.input_valid_list, 'r') as ins:
        prompt_list = json.load(ins)

    pre_scaling_factor = vae.scaling_factor
    for step, video_data in enumerate(prompt_list):
        with torch.no_grad():
            prompt = video_data['caption']
            ref_video_path = ''
            # if 'video' in video_data:
            #     ref_video_path = video_data['video']
            print(f'ref_video_path is {ref_video_path}', flush=True)
            if args.offline_text_feat and args.offline_text_feat_dir:
                video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
                feat_path = os.path.join(args.offline_text_feat_dir, f'{video_name}.pkl')
                with open(feat_path, 'rb') as ins1:
                    prompt_embeds = pkl.load(ins1)['embeddings'][0]['t_embedding']
                    prompt_embeds = torch.from_numpy(prompt_embeds)

                with open(args.neg_text_embedding_path, 'rb') as ins2:
                    negative_prompt_embeds = pkl.load(ins2)['null_embeddings']
                    negative_prompt_embeds = torch.from_numpy(negative_prompt_embeds)

                gen_single_sample(validation_pipeline, 
                                  prompt=prompt, 
                                  args=args, 
                                  samples_dir=samples_dir, 
                                  train_ts=test_ts, 
                                  step=train_steps, 
                                  negative_prompt=None,
                                  prompt_embeds=prompt_embeds,
                                  negative_prompt_embeds=negative_prompt_embeds,
                                  output_type=output_type)
            elif ref_video_path == '':
                if args.HawkT2V.use_additional_conditions:
                    data_info = set_added_info(device=device)
                    gen_single_sample(validation_pipeline, prompt, args, samples_dir, test_ts, train_steps, output_type=output_type, added_cond_kwargs=data_info)
            else:
                vae.scaling_factor = pre_scaling_factor
                if args.HawkT2V.use_additional_conditions:
                    video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
                    feat_path = os.path.join(args.offline_text_feat_dir, f'{video_name}.pkl')
                    data_info = get_added_info(feat_path, device=device)
                    # data_info = set_added_info(device=device)
                    gen_single_sample(validation_pipeline, prompt, args, samples_dir, test_ts, train_steps, output_type=output_type, added_cond_kwargs=data_info)
                else:
                    gen_single_sample(validation_pipeline, prompt, args, samples_dir, test_ts, train_steps, output_type=output_type)

                video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
                feat_path = os.path.join(args.offline_text_feat_dir, f'{video_name}.pkl')
                if not os.path.exists(feat_path):
                    continue 
                with open(feat_path, 'rb') as ins1:
                    vae_embeds = pkl.load(ins1)['embeddings'][0]['vae_latent']
                    vae_embeds = torch.from_numpy(vae_embeds).unsqueeze(0).to(device)
                    # import ipdb; ipdb.set_trace();
                    vae.scaling_factor = 1.0
                    videos = decode_latents(vae, vae_embeds, decode_length=vae_embeds.shape[2]).cpu()
                    file_prefix = prompt.replace(' ', '_')[:50]
                    imageio.mimwrite(samples_dir + '/' + file_prefix + f'{test_ts}_{train_steps}_origin_webv_imageio.mp4',
                            videos[0], fps=8, quality=9)  # highest quality is 10, lowest is 0
                    print('saved in path {}'.format(samples_dir + '/' + file_prefix + f'{test_ts}_{train_steps}_origin_webv_imageio.mp4'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))